

# --------------------------------------------------------------------------------------------------------------
#  Dependencies
# --------------------------------------------------------------------------------------------------------------

# external
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document 
from pydantic import BaseModel, Field
from dotenv import load_dotenv 

# internal
from rag import ConversationalAgenticRAG
from configs import PodagentConfigs
from utils import get_clean_chunks, load_pdf_content

# built in
from typing import Annotated, List, Dict, Optional, Literal, AnyStr
import os 
 








## loading secret keys
load_dotenv()

 




 
#------------------------------------------------------------------------------------------
# Confiurations
#------------------------------------------------------------------------------------------

# COMMON_LLM = "gemma-3-12b" 
COMMON_LLM = PodagentConfigs.COMMON_LLM
COMMON_TOKENS_SIZE = 32000



#------------------------------------------------------------------------------------------
# LLM instance: Gemini
#------------------------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model = COMMON_LLM,
    verbose = False,
    max_tokens = COMMON_TOKENS_SIZE,
    temperature = 0.3 
)












#------------------------------------------------------------------------------------------
# Agent State
#------------------------------------------------------------------------------------------

class ChapterContentLoaderAgentState(BaseModel):

    # ----------------------- input -------------------------------------
    user_query: Annotated[str, Field(..., title="user query", description="exact user query")]

    # -------------------- internal state -------------------------------
    fetched_chapter_name_or_number: Optional[Dict[str,str|int|None]]
    message: Optional[str] = ""
    chapter_name: Optional[str] = "" 
    chapter_number: Optional[str|int] = 0
    chapter_page_number: Optional[int] = 0
    table_of_contents_page_no: Optional[int] = 0

    # ------------------------- output -----------------------------------
    chapter_content: Optional[str] = ""






#------------------------------------------------------------------------------------------
# 
#          Nodes development starts from here
# 
#------------------------------------------------------------------------------------------





# ------------------------------------- Node 1 --------------------------------------------


#------------------------------------------------------------------------------------------
# LLM Output parser schema
#------------------------------------------------------------------------------------------
 
class FetchChapterNameParser(BaseModel):
    chapter_name: Annotated[Optional[str], Field(..., title="chapter name")] = None 
    chapter_number: Annotated[Optional[int], Field(..., title="chapter number")] = None




#------------------------------------------------------------------------------------------
# Node 
#------------------------------------------------------------------------------------------

def fetch_chapter_name(state: ChapterContentLoaderAgentState):
    user_query: str = state.model_dump()['user_query']

    parser = PydanticOutputParser(pydantic_object=FetchChapterNameParser)

    template = PromptTemplate(
        name="fetch_chapter_name_prompt_template",
        template="""
You are a good extractor. You have to fetch chapter name or chapter number from the given user_query.
Your response must be a json only, its format: {format_instructions}

user_query: {user_query}
        """,
        input_variables=['user_query'],
        partial_variables={'format_instructions':parser.get_format_instructions()},
        validate_template=True
    )

    chain = template | llm | parser 

    result: ChapterContentLoaderAgentState = chain.invoke({'user_query':user_query})

    return { "fetched_chapter_name_or_number": result.model_dump() }

 

 

#------------------------------------------------------------------------------------------
# Orchestrator function 
#------------------------------------------------------------------------------------------

def orchestrator(
        state: ChapterContentLoaderAgentState
    ) -> Literal["ch_found","ch_not_found"]:

    """ it will orchestrate the route depends on chapter name/numbe found or not """

    _state: dict = state.model_dump()
    ch_name = _state['fetched_chapter_name_or_number']['chapter_name']
    ch_number = _state['fetched_chapter_name_or_number']['chapter_number']

    if ch_name or ch_number:
        # no chapter found
        return "ch_found"
    
    return "ch_not_found"


 



#------------------------------------------------------------------------------------------
# Node 2
#------------------------------------------------------------------------------------------

def update_msg_chapter_not_found(state: ChapterContentLoaderAgentState):
    """Updating message in state when chapter name and chapter serial number not found.

    Args:
        state (ChapterContentLoaderAgentState): agent state

    """

    message = "No chapter name or number found, user have to provide it first."

    return { "message": message }

 





# ----------------------------------------- Node 3 ----------------------------------------

#------------------------------------------------------------------------------------------
# Loading RAG for this node
#------------------------------------------------------------------------------------------

rag = ConversationalAgenticRAG(PodagentConfigs.pdf_path)

rag.load_vector_store()





#------------------------------------------------------------------------------------------
# LLM Output parser schema 
#------------------------------------------------------------------------------------------
 
class FetchChapterPageNumberParserSchema(BaseModel):
    
    chapter_name: Annotated[
        Optional[str], 
        Field(..., title="chapter name", description="same as mention in the retrieved book structure chunks, or given chapter_name")
    ]

    chapter_number: Annotated[
        Optional[str], 
        Field(..., title="chapter number", description="same as input chapter_number if not given then chapter serial number of given chapter_name")
    ]

    starting_page_number: Annotated[
        int | None, 
        Field(..., title="page number where given `chapter_name` or `chapter_number` begins ")
    ]

    table_of_contents_page_no: Annotated[
        int,
        Field(..., title="table of contents page number", description="page number of page table of contents")
    ]

    first_chapter_page_no: Annotated[
        int,
        Field(..., description="page number of first chapter")
    ]







#------------------------------------------------------------------------------------------
# Node 4
#------------------------------------------------------------------------------------------

def fetch_chapter_page_number(state: ChapterContentLoaderAgentState):

    ## fetching state
    _state = state.model_dump()

    ## query for extracting "table of contents" from RAG
    query = """ Table of Contents, Contents, Index, Chapter """

    ## extracting docs from rag
    extracted_chunks = rag.fetch_docs(query=query, top_k=5)
    # merging chunks
    chunks_merged: str = get_clean_chunks(extracted_chunks)
    
    # output parser
    parser = PydanticOutputParser(pydantic_object=FetchChapterPageNumberParserSchema)

    # prompt template 
    prompt = PromptTemplate(
        input_variables=[
            "chapter_name",
            "chapter_number"
        ],
        partial_variables={
            "retrieved_chunks":chunks_merged,
            "output_format": parser.get_format_instructions()
        },
        validate_template=True,
        template="""
You are a precise information extraction assistant.

TASK:
You are given structured chunks extracted from a book's
Table of Contents or Index pages.

Your job is to determine:
1. The starting page number of the given chapter.
2. If chapter_number is given so find chapter name.
3. If chapter_name is given so find chapter number.
4. The page number of table of contents.

IMPORTANT RULES (STRICT):
- Use ONLY the information explicitly present in the provided chunks.
- Do NOT infer or guess page numbers.
- Do NOT hallucinate missing values.

INPUTS:
- Chapter Name: {chapter_name}
- Chapter Number: {chapter_number}

RETRIEVED BOOK STRUCTURE CHUNKS:
-------------------------------
{retrieved_chunks}
-------------------------------

OUTPUT FORMAT INSTRUCTIONS:
-------------------------------
{output_format}
-------------------------------
"""
    )

    # extracting chapter's page number
    chain = prompt | llm | parser

    result: FetchChapterPageNumberParserSchema = chain.invoke(_state['fetched_chapter_name_or_number'])
    result_as_dict: dict = result.model_dump()

    response = {
        "chapter_name": result_as_dict['chapter_name'],
        "chapter_number": result_as_dict['chapter_number'],
        "chapter_page_number": result_as_dict['starting_page_number'],
        "table_of_contents_page_no": result_as_dict['table_of_contents_page_no']
    }

    return response 


 






#------------------------------------------------------------------------------------------
# Node 5
#------------------------------------------------------------------------------------------
 
def fetch_content_from_x_to_y_page(state: ChapterContentLoaderAgentState):

    _state = state.model_dump()

    # extracting pages
    chapter_page_number, table_of_contents_page_no = _state['chapter_page_number'], _state['table_of_contents_page_no']
    starting_page_no = chapter_page_number + table_of_contents_page_no + 2

    # loading pdf selective pages
    pages_content = load_pdf_content(
        PodagentConfigs.pdf_path,
        _from = starting_page_no, 
        to = starting_page_no + 7 # next 7 pages of chapter
    )

    return { 'chapter_content': get_clean_chunks(pages_content, plain_text=True) }


 


 



#------------------------------------------------------------------------------------------
# Building Agent
#------------------------------------------------------------------------------------------
 
# ------------------- defining graph -------------------------
graph = StateGraph(ChapterContentLoaderAgentState)

# ------------------- adding nodes ---------------------------
graph.add_node("fetch_ch_name", fetch_chapter_name)
graph.add_node("fetch_ch_page_no", fetch_chapter_page_number)
graph.add_node("load_ch_content", fetch_content_from_x_to_y_page)
graph.add_node("update_msg_in_state", update_msg_chapter_not_found)


# ------------------ connecting edges ------------------------
graph.set_entry_point("fetch_ch_name")
graph.add_conditional_edges(
    source="fetch_ch_name",
    path=orchestrator,
    path_map={
        "ch_found":"fetch_ch_page_no",
        "ch_not_found": "update_msg_in_state" 
    }
)
graph.add_edge("fetch_ch_page_no", "load_ch_content")
graph.add_edge("load_ch_content", END)
graph.add_edge("update_msg_in_state",END)


# ------------------- graph compilation -----------------------
workflow = graph.compile()








#------------------------------------------------------------------------------------------
# loader function: helps agent to be use in another file
#------------------------------------------------------------------------------------------
 
def load_chapter_content_loader_agent():

    return workflow 















#------------------------------------------------------------------------------------------
# Test function
#------------------------------------------------------------------------------------------
 
def test_agent():
    initial_state = ChapterContentLoaderAgentState(
        user_query="generate a quiz of 13 chapter",
        fetched_chapter_name_or_number=None
    )
    
    response = workflow.invoke(initial_state)
    
    print(response['chapter_content'])

 


 









if __name__ == "__main__":

    test_agent()



 


 



