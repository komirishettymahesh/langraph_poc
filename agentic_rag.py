import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Agentic-RAG'

llm = AzureChatOpenAI(
    model='gpt-4.1-mini',
    azure_endpoint=AZURE_API_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION
)

urls = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph"
]

docs = [WebBaseLoader(urls).load() for url in urls]

doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_split = text_splitter.split_documents(doc_list)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=AZURE_API_ENDPOINT)

vector_store = Chroma.from_documents(doc_split, embeddings, collection_name='langgraph_db')

retriver = vector_store.as_retriever()

langgraph_retriver_tool = create_retriever_tool(
    retriver,
    "retriver_vector_db_blog",
    "Search and run information about langgraph"
)

urls = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/tools"
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

doc_split = text_splitter.split_documents(docs_list)

vector_store_lc = Chroma.from_documents(doc_split, embeddings, collection_name='langgraph_db')

retriver_langchain = vector_store_lc.as_retriever()

langchain_retriver_tool = create_retriever_tool(
    retriver_langchain,
    "retriver_vector_langchain_blog",
    "Search and run information about langchain"
)

tools = [langgraph_retriver_tool, langchain_retriver_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def agent(state:AgentState):
    """
    Invokes the agent model to generate a response based on the current state. Given the question, it will decide to retrieve using the retriever tool, or simply end
    
    Args:
        state(messages): The current state 
        
    Returns:
        dict: The updated state with the agent response appended to messages    
    """
    print('---Call Agent---')
    messages = state['messages']
    llm_with_models = llm.bind_tools(tools)
    response = llm_with_models.invoke(messages)
    return {'messages': [response]}

def grade_documents(state:AgentState) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question
    
    Args: 
        state (messages): the current state 
    Returns: 
        str: A decision for whether the documents are relevant ot not"""
    
    print("---Check Relevance---")
    
    class grade(BaseModel):
        """
        Binary score for relevance check
        """
        
        binary_score: str = Field(description="Relavance score 'yes' or 'no'")
        
    llm_with_tool = llm.with_structured_output(grade)
    
    prompt = PromptTemplate(
        template = """You are a grader assessing relevance of a retrieved document to a user question.\n
        Here is the retrived document: \n\n {context} \n\n
        Here is the user question: {question} \n 
        If the document contains keyword(s) or semantic meaing related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question""",
        input_variables = ["context", "question"],
    )
    
    chain = prompt | llm_with_tool 
    
    messages = state['messages']
    last_message = messages[-1]
    
    question = messages[0].content 
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score 
    
    if score == 'yes':
        print('---Decision: Docs Relevant---')
        return "generate"
    else:
        print("---Decision: Docs not relevant---")
        print(score)
        return "rewrite"
    
    
def generate(state:AgentState):
    """
    Generate answer
    
    Args:
        state(messages): the current state 
        
    Returns:
        dict: The updated message 
    """
    print("--Generate--")
    messages = state['messages']
    question = messages[0].content 
    last_message = messages[-1]
    docs = last_message.content 
    
    '''prompt = hub.pull('rlm/rag-prompt')
    
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
        
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke({'context': docs, 'question': question}) '''
    
    return {'messages': [docs]}
    
    
def rewrite(state:AgentState):
    
    print("---Transform query---")
    messages = state['messages']
    question = messages[0].content
    
    msg = [
        HumanMessage(content=f"""\n
                     Look at the input and try to reason aboyt the underlying semantic intent/meaning.\n
                     Here is the initial question: 
                     \n---------\n
                     {question}
                     \n--------\n
                     Formulate an improved question:""")
    ]
    
    response = llm.invoke(msg)
    return {'messages': [response]}


workflow = StateGraph(AgentState)

workflow.add_node('agent', agent)
workflow.add_node("retriver", ToolNode(tools))
workflow.add_node('generate',generate)
workflow.add_node('rewrite', rewrite)


workflow.add_edge(START, 'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        "tools": "retriver",
        END:END
    }
)
workflow.add_conditional_edges(
    "retriver",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

response = graph.invoke({"messages": "what is langgraph"})

final_output = response["messages"][-1].content
print(final_output)