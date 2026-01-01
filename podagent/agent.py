#------------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------------

# external
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from dotenv import load_dotenv

# local
# built-in
from typing import List, Optional, Literal, Annotated, Dict






## loading secret keys
load_dotenv()







#------------------------------------------------------------------------------------------
# Confiurations
#------------------------------------------------------------------------------------------

COMMON_LLM = "gemini-2.5-flash"
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





#------------------------------------------------------------------------------------------
# Nodes
#------------------------------------------------------------------------------------------

def agent_chat_node(state: PodagentSchema):
    messages = state.model_dump()['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}







#------------------------------------------------------------------------------------------
# Workflow
#------------------------------------------------------------------------------------------

# graph instance
graph = StateGraph(PodagentSchema)

# adding nodes
graph.add_node("agent", agent_chat_node)

# connecting edges
graph.add_edge(START, "agent")
graph.add_edge("agent", END)


# extracting workflow
workflow = graph.compile()







#------------------------------------------------------------------------------------------
# Test agent
#------------------------------------------------------------------------------------------

def test_agent():

    prompt = "hi"
    initial_state = PodagentSchema(messages=[HumanMessage(content=prompt)])
    response = workflow.invoke(initial_state)

    print(response)









if __name__ == "__main__":

    test_agent()

    # initial_state = PodagentSchema(messages=[HumanMessage(content="hello")])

    # print(initial_state)









