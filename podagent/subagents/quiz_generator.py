 
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
 






 
#------------------------------------------------------------------------------------------
# LLM instance: Gemini
#------------------------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    verbose = False,
    max_tokens = 32000,
    temperature = 0.7
)






#------------------------------------------------------------------------------------------
# Schema and state
#------------------------------------------------------------------------------------------

 
class MCQ(BaseModel):
    question: Optional[str] = None 
    options: Optional[List[str]] = []
    right_option: Optional[str] = None


class GenerateQuizAgentState(BaseModel):

    chapter_content: Annotated[str, Field(..., title="content", description="fetched chapter content")]
    number_of_questions: int = 5
    quiz: Optional[List[MCQ]] = None








#------------------------------------------------------------------------------------------
# Template
#------------------------------------------------------------------------------------------

template_content = """
You are an AI quiz generation agent.

Your task is to generate EXACTLY **{number_of_questions} multiple-choice questions (MCQs)** 
using ONLY the provided chapter content.

====================
STRICT INSTRUCTIONS
====================
1. Use only facts, definitions, or statements explicitly present in the chapter content.
2. Do NOT use outside knowledge or assumptions.
3. If information is missing or unclear, do not invent it.
4. Each MCQ must contain:
   - One clear question
   - Exactly 4 options
   - Exactly 1 correct option
5. The correct option must be directly supported by the chapter content.
6. Keep the language simple, factual, and unambiguous.
7. Question level will be basic.
8. Try to be short options.

====================
OUTPUT EXPECTATION
====================
Return the result in a structure compatible with the following models:

- List[MCQ]
- Each MCQ contains:
  - question: string
  - options: list of strings (length = 4)
  - right_option: string (must exactly match one option)

Do NOT include explanations, markdown, comments, or extra text.


====================
OUTPUT STRUCTURE
====================
{output_structure}


====================
CHAPTER CONTENT
====================
{chapter_content}


Now you can start you work.
"""






#------------------------------------------------------------------------------------------
# Output parser schema
#------------------------------------------------------------------------------------------


class GenerateQuizOutputParserSchema(BaseModel):
    mcqs: Annotated[
        List[MCQ],
        Field(..., title="MCQs", description="List of mcqs")
    ]







#------------------------------------------------------------------------------------------
# Node: it will generate mcqs
#------------------------------------------------------------------------------------------


def generate_quiz_node(state: GenerateQuizAgentState):

    _state = state.model_dump()
    chapter_content: str = _state['chapter_content']
    number_of_questions: int = _state['number_of_questions']

    parser = PydanticOutputParser(pydantic_object=GenerateQuizOutputParserSchema)

    template = PromptTemplate(
        input_variables=['chapter_content','number_of_questions'],
        partial_variables={"output_structure": parser.get_format_instructions()},
        template=template_content,
        validate_template=True 
    )

    # building chain
    chain = template | llm | parser 

    response = chain.invoke({
        'chapter_content':chapter_content,
        'number_of_questions':number_of_questions
    })

    return { "quiz": response.model_dump() }









#------------------------------------------------------------------------------------------
# Building sub-agent
#------------------------------------------------------------------------------------------

 
# ----------------- graph ------------------------
graph = StateGraph(GenerateQuizAgentState)


# ------------- adding nodes ---------------------
graph.add_node("generate", generate_quiz_node)


# ----------- connecting edges -------------------
graph.set_entry_point("generate")
graph.add_edge("generate", END)


# ---------------- comilation --------------------
workflow = graph.compile()




 


 


#------------------------------------------------------------------------------------------
# function that use to load this agent
#------------------------------------------------------------------------------------------

def load_quiz_generator_agent():

    return workflow

 



