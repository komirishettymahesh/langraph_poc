from typing import Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper 
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ReAct-Agent_v2'


arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

tools = [arxiv, wiki]


llm = AzureChatOpenAI(
    model='gpt-4.1-mini',
    azure_endpoint=AZURE_API_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION
)




class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def make_default_graph():
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    
    tool_node = ToolNode([add])
    model_with_tools = llm.bind_tools(tools)
    
    
    graph_workflow = StateGraph(State)
    
    def call_model(state:State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    graph_workflow.add_node("Agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    
    graph_workflow.add_edge(START, "Agent")
    graph_workflow.add_conditional_edges('Agent', tools_condition)
    graph_workflow.add_edge('tools', 'Agent')
    
    builder = graph_workflow.compile()
    
    return builder

agent = make_default_graph()

