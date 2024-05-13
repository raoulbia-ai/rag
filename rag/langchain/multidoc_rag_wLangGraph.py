"""
source ~/.bashrc
"""
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
# from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentType, initialize_agent

from pprint import pprint
import os
import glob
# from rag.load_keys import *0000000000000000000

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document

from langgraph.graph import END, StateGraph

import logging, os# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(funcName)s - %(message)s')

# Load API Key
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
mistral_api_key = os.getenv("MISTRAL_API_KEY")


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class DocumentInput(BaseModel):
    question: str = Field()


mistral_model = "mistral-large-latest" # "open-mixtral-8x22b" 
llm = ChatMistralAI(model=mistral_model, temperature=0)
embeddings = MistralAIEmbeddings()

# Create Files list of dicts
directory_path = "/teamspace/uploads"
pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
files = []
for pdf_file in pdf_files:
    base_name = os.path.basename(pdf_file)
    name, _ = os.path.splitext(base_name)
    files.append({
        "name": name,
        "path": pdf_file,
    })
# print(files)

# Wrap retrievers in a Tool
tools = []

def wrap_retrievers_in_tool(file, retriever, llm):
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )


# Document Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)


# Prompt 
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
    """
    question : str
    generation : str
    documents : List[str]


### Nodes 

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    if not os.path.exists("faiss_index"):
        print("Creating 'faiss_index'")
        for file in files:
            loader = PyPDFLoader(file["path"])
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            
            index = FAISS.from_documents(docs, embeddings)
            index.save_local("faiss_index")
            retriever = index.as_retriever()
            wrap_retrievers_in_tool(file, retriever, llm)     
    else:
        print("Loading from disc 'faiss_index'")
        index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = index.as_retriever()
        for file in files:
            wrap_retrievers_in_tool(file, retriever, llm)


    question = state["question"]
    print(question)

    # Transform the question into a vector using the same embedding model
    # question_vector = embeddings.embed_documents([question])

    # Retrieval
    documents = retriever.invoke(question)
    print(len(documents))
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    logging.info(f"Prompt Hub prompt: {prompt}")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        print(grade)
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}
    

### Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"

### Build Graph ###

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
    },
)
workflow.add_edge("generate", END)

# Compile
G = workflow.compile()

from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod
with open('output_v4.png', 'wb') as f:
    f.write(G.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))

if __name__ == "__main__":
    
    inputs = {"question": "What are the names of the inspected cetnres?"}
    for output in G.stream(inputs):
        for key, value in output.items():
            print(key)
            print(value)
            pprint(f"Finished running: {key}:")
            break
    # pprint(value["generation"])