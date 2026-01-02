#------------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------------

# external
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from dotenv import load_dotenv

# local
from rag import ConversationalAgenticRAG
from configs import PodagentConfigs

# built-in
from typing import List, Optional, Literal, Annotated, Dict






## loading secret keys
load_dotenv()







#------------------------------------------------------------------------------------------
# Confiurations
#------------------------------------------------------------------------------------------

COMMON_LLM = "gemini-2.5-flash-lite"
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
# Schema
#------------------------------------------------------------------------------------------

class PodagentSchema(BaseModel):

    messages: Annotated[List[BaseMessage], add_messages]
    fetched_docs: Optional[str] = ""













#------------------------------------------------------------------------------------------
# Tools
#------------------------------------------------------------------------------------------

rag = ConversationalAgenticRAG(PodagentConfigs.pdf_path)

@tool
def retriever(query: str) -> dict:
    """
    use to get information from knowledge base
    
    :param query: query that use to retrieve docs
    :type query: str
    :return: retrieved docs
    :rtype: dict
    """

    retrieved_docs = rag.fetch_docs(query=query, conversational=False)

    merged = ""
    for doc in retrieved_docs:
        merged += doc.page_content

    # print(f"\n\nMerged: {merged}")

    return str({ "fetched_docs": merged })

    











#------------------------------------------------------------------------------------------
# Tools binding
#------------------------------------------------------------------------------------------

tools = [retriever]

llm_with_tools = llm.bind_tools(tools)













#------------------------------------------------------------------------------------------
# Nodes
#------------------------------------------------------------------------------------------

def agent_chat_node(state: PodagentSchema):
    messages = state.model_dump()['messages']
    print(f"\n\nAgent chat node [come in] -> messages: {messages}")
    response = llm_with_tools.invoke(messages)
    # final_response = {
    #     "messages": messages + [{
    #         "role": "assistant",
    #         "content": response.content
    #     }]
    # }

    final_response = {
        "messages": messages + [response]
    }

    print(f"\n\nreturning from agent node -> {final_response}")
    return final_response





tool_node = ToolNode(tools)


def should_use_tool(state: PodagentSchema):
    last_message = state.model_dump()["messages"][-1]
    # print(last_message)
    try:
        if last_message['additional_kwargs']['function_call']:
            print(f"\n\ncalling tools, here last message is: {last_message}")
            return "tools"
        else:
            "end"
        # return "tools" if last_message.function_call else "end"
    except AttributeError:
        print("\n\nno-function_call exists")
        return "end"
    
    # def should_use_tool(state: AgentState):
    # last_message = state["messages"][-1]
    # print(last_message)
    # return "tools" if last_message.additional_kwargs.function_call else "end"

    
    





#------------------------------------------------------------------------------------------
# Workflow
#------------------------------------------------------------------------------------------

# graph instance
graph = StateGraph(PodagentSchema)

# adding nodes
graph.add_node("agent", agent_chat_node)
graph.add_node("tools", tool_node)

# connecting edges
# graph.add_edge(START, "agent")
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent", 
    should_use_tool, 
    {
        "tools": "tools",
        "end": END
    }
)

# graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
# graph.add_edge("agent", END)


# extracting workflow
workflow = graph.compile()







#------------------------------------------------------------------------------------------
# Test agent
#------------------------------------------------------------------------------------------

def test_agent():

    prompt = "what is the conclusion of Auto Park King System"
    # prompt = "how many states in india"
    initial_state = PodagentSchema(messages=[HumanMessage(content=prompt)])
    response = workflow.invoke(initial_state)

    print(response)

    print(f"\n\nAI) {response['messages'][-1].content}")








if __name__ == "__main__":

    test_agent()

    # initial_state = PodagentSchema(messages=[HumanMessage(content="hello")])

    # print(initial_state)









