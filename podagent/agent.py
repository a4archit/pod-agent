#------------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------------

# external
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# local
from rag import ConversationalAgenticRAG
from configs import PodagentConfigs
from subagents.chapter_content_loader import load_chapter_content_loader_agent, ChapterContentLoaderAgentState
from subagents.quiz_generator import load_quiz_generator_agent, GenerateQuizAgentState, MCQ

# built-in
from typing import List, Optional, Literal, Annotated, Dict
import sys





## loading secret keys
load_dotenv()







#------------------------------------------------------------------------------------------
# Loading sub agents
#------------------------------------------------------------------------------------------

chapter_content_loader_subagent = load_chapter_content_loader_agent()

quiz_generator_subagent = load_quiz_generator_agent()









#------------------------------------------------------------------------------------------
# Confiurations
#------------------------------------------------------------------------------------------

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

class PodagentSchema(BaseModel):

    messages: Annotated[List[BaseMessage], add_messages]
    fetched_docs: Optional[str] = ""

    chapter_content: Annotated[
        Optional[str],
        Field(
            ..., 
            title="chapter content", 
            description="Content of chapter, if it has some value other than `None` it \
                means `load_chapter_content` sub agent is called."
        )
    ] = None 

    quiz: Annotated[
        Optional[List[MCQ]],
        Field(
            ...,
            title="quiz",
            description="Generated quiz, if it has some value other than `None` it \
                means `generate_quiz` subagent is called." 
        )
    ] = None 









#------------------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------------------

def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    """
    It will return the last HumanMessage from the provided messages list
    
    :param messages: list of messages
    :type messages: List[BaseMessage]
    :return: last human message
    :rtype: HumanMessage
    """


    for msg in messages[::-1]:
        if isinstance(msg, HumanMessage):
            return msg











#------------------------------------------------------------------------------------------
# Orchestrator: template
#------------------------------------------------------------------------------------------

orchestrator_template = """
You are an expert orchestration agent responsible for routing tasks.

Your goal is to select EXACTLY ONE next action based on:
- The current agent state
- The user query
- The defined routing rules
- The Agent's answer


====================
AVAILABLE SUB AGENTS
====================
`load_chapter_content`  → Loads the content of a chapter  
`generate_quiz`         → Generates a quiz from chapter content  

====================
AGENT CURRENT STATE
====================
{agent_state}

====================
STATE-BASED RULES (MANDATORY)
====================
1. If `quiz` in agent state has any value other than None:
   - You MUST return: `end`

2. Else if `chapter_content` in agent state has any value other than None:
   - You are allowed ONLY these options:
     ["generate_quiz", "end"]

3. Else (both `chapter_content` and `quiz` are None):
   - You may choose from:
     ["load_chapter_content", "end"]

4. If you are satisfy with the agent's answer with respect to query then:
    - You must return: `end`

Violating these rules is NOT allowed.

====================
USER QUERY
====================
{user_query}


====================
AGENT ANSWER
====================
{agent_answer}

====================
DECISION INSTRUCTIONS
====================
- Select ONLY ONE option.
- Your selection MUST be one of:
  ["generate_quiz", "load_chapter_content", "end"]
- If no sub-agent is required, return `end`.
- Do NOT explain your reasoning.
- Do NOT include markdown.
- Do NOT hallucinate or infer missing state.

====================
OUTPUT FORMAT (STRICT)
====================
{output_format}

====================
BEGIN DECISION
====================
"""



#------------------------------------------------------------------------------------------
# Orchestrator: Output parser
#------------------------------------------------------------------------------------------

class OrchestratorOutputParserSchema(BaseModel):

    next: Literal[
        'load_chapter_content',
        'generate_quiz',
        'end'
    ]



#------------------------------------------------------------------------------------------
# Orchestrator: function
#------------------------------------------------------------------------------------------

def orchestrator(
        state: PodagentSchema
    ) -> Literal["generate_quiz","load_chapter_content","end"]:

    print(" ((orchestrator)) ", end="")

    # extracting agent state
    agent_state = state.model_dump()
    # extracting agent answer
    agent_ans = state.messages[-1].content
    # deleting messages and fetched_docs from agent state
    del agent_state['messages'], agent_state['fetched_docs']
    # updating value of chapter content if it has 
    # - to reduce tokens while passing it into orchesreator's llm
    if len(str(agent_state['chapter_content'])) > 20: # it means it has any other value than `None`
        agent_state['chapter_content'] = agent_state['chapter_content'][:50]
    if len(str(agent_state['quiz'])) > 20: 
        agent_state['quiz'] = str(agent_state['quiz'])[:50]

    # fetching all messages
    messages = state.messages
    # fetching user query
    user_query = get_last_human_message(messages).content
    # building parser
    parser = PydanticOutputParser(pydantic_object=OrchestratorOutputParserSchema)

    template = PromptTemplate(
        template = orchestrator_template,
        validate_template = True,
        input_variables = ['user_query'],
        partial_variables = {
            'output_format':parser.get_format_instructions(),
            'agent_state': agent_state,
            'agent_answer': agent_ans
        } 
    )

    # chaining all components
    # chain = template | llm | parser 

    prompt = template.invoke({'user_query': user_query})
    llm_result = llm.invoke(prompt)
    parser_result = parser.invoke(llm_result)

    # getting result
    result = parser_result # chain.invoke({})

    print(" -> ", end="")

    # # ------------------------ testing --------------------
    # print(result.model_dump()['next']) 
    # print("\n\nPrompt: \n", prompt)
    # sys.exit(1)

    return result.model_dump()['next']












#------------------------------------------------------------------------------------------
# Node: Main agent node
#------------------------------------------------------------------------------------------

def agent_chat_node(state: PodagentSchema):
    messages = state.model_dump()['messages']
    print(" [agent] ", end="")
    # print(f"\n\nAgent chat node [come in] -> messages: {messages}")
    response = llm.invoke(f"{messages}, \n\nretrieved docs: {state.fetched_docs} ")

    final_response = {
        "messages": messages + [response]
    }

    # print(f"\n\nreturning from agent node -> {final_response}")
    print(" -> ", end="")
    return final_response




#------------------------------------------------------------------------------------------
# Node: RAG node
#------------------------------------------------------------------------------------------


rag = ConversationalAgenticRAG(PodagentConfigs.pdf_path)
rag.load_vector_store()

def retriever(state: PodagentSchema) -> dict:
    """
    use to get information from knowledge base
    
    :param query: query that use to retrieve docs
    :type query: str
    :return: retrieved docs
    :rtype: dict
    """
    print(" [retriever] ", end="")
    query = get_last_human_message(state.messages).content
    # print(f"\n\nQuery = {query}\n")
    retrieved_docs = rag.fetch_docs(query=query, conversational=False)

    merged = ""
    for doc in retrieved_docs:
        merged += doc.page_content

    print(" -> ", end="")

    return { "fetched_docs": merged }

    




#------------------------------------------------------------------------------------------
# Node: Connecting subagent [chapter_content_loader]
#------------------------------------------------------------------------------------------

def chapter_content_loader_subagent_node(state: PodagentSchema):

    print(" [subagent[ch_content_loader]] ", end="")

    # fetching user query from state
    messages = state.messages
    user_query = get_last_human_message(messages).content 

    # setting up initial state
    initial_state = ChapterContentLoaderAgentState(
        user_query = user_query,
        fetched_chapter_name_or_number=None 
    )

    # calling subagent
    final_state = chapter_content_loader_subagent.invoke(initial_state)

    print(" -> ", end="")

    return { 'chapter_content': final_state['chapter_content'] }









#------------------------------------------------------------------------------------------
# Node: Connecting subagent [quiz_generator]
#------------------------------------------------------------------------------------------

def quiz_generator_subagent_node(state: PodagentSchema):

    print(" [subagent[quiz_generator]] ", end="")

    # fetching chapter content from state
    chapter_content = state.model_dump()['chapter_content']

    # setting up initial state
    initial_state = GenerateQuizAgentState(
        chapter_content= chapter_content,
        number_of_questions=5 
    )

    # calling subagent
    final_state = quiz_generator_subagent.invoke(initial_state)

    print(" -> ", end="")

    return { 'quiz': final_state.model_dump()['quiz'] }








#------------------------------------------------------------------------------------------
# Workflow
#------------------------------------------------------------------------------------------

# ------------------------- graph instance -----------------------
graph = StateGraph(PodagentSchema)

# adding nodes
graph.add_node("agent", agent_chat_node)
graph.add_node("retriever", retriever)
graph.add_node("quiz_generator", quiz_generator_subagent_node)
graph.add_node("ch_content_loader", chapter_content_loader_subagent_node)


# ------------------------- connecting edges ---------------------
graph.set_entry_point("retriever")
graph.add_edge("retriever","agent")
graph.add_conditional_edges(
    source="agent",
    path=orchestrator,
    path_map={
        'generate_quiz': 'quiz_generator', 
        'load_chapter_content': 'ch_content_loader',
        'end': END
    }
)
graph.add_edge("quiz_generator","agent")
graph.add_edge("ch_content_loader", "agent")


# ----------------------- extracting workflow ---------------------
workflow = graph.compile()


# # ------------------------- saving workflow diagram ----------------
# png_bytes = workflow.get_graph().draw_mermaid_png()

# with open("podagent.png", "wb") as f:
#     f.write(png_bytes)









#------------------------------------------------------------------------------------------
# Test agent
#------------------------------------------------------------------------------------------

def test_agent():

    prompt = "how many chapters in the book"
    # prompt = "how many states in india"
    initial_state = PodagentSchema(messages=[HumanMessage(content=prompt)])
    response = workflow.invoke(initial_state)

    # print(response)

    print(f"\n\nAI) {response['messages'][-1].content}")








if __name__ == "__main__":

    # from subagents.chapter_content_loader import test_agent as t2

    # t2()

    test_agent()

    # print(workflow.get_graph().draw_mermaid())

    # initial_state = PodagentSchema(messages=[HumanMessage(content="hello")])

    # print(initial_state)









