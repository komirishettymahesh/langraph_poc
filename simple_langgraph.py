import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages

load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT') 


llm = AzureChatOpenAI(
    model='gpt-4.1-mini',
    azure_endpoint=AZURE_API_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def superbot(state:State):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(State)

graph.add_node("Superbot",superbot)
graph.add_edge(START, "Superbot")
graph.add_edge("Superbot", END)

graph_builder = graph.compile()

#response = graph_builder.invoke({"messages": 'Can you help me write a poem about LangGraph?'})

#print(response)

for event in graph_builder.stream({"messages": 'Can you help me write a poem about LangGraph?'}, stream_mode="values"):
    print(event)

