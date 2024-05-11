"""
https://python.langchain.com/v0.1/docs/integrations/toolkits/document_comparison_toolkit/

agent types: 
- https://python.langchain.com/v0.1/docs/modules/agents/agent_types/
- https://levelup.gitconnected.com/giving-mistral-7b-access-to-tools-with-langchain-agents-8daf3d1fe741

source ~/.bashrc

"""

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
# from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentType, initialize_agent

import os
import glob
from rag.load_keys import *

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


class DocumentInput(BaseModel):
    question: str = Field()

# OpenAI
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# embeddings = OpenAIEmbeddings()

# Mistral
mistral_model = "mistral-large-latest" # "open-mixtral-8x22b" 
llm = ChatMistralAI(model=mistral_model, temperature=0)
embeddings = MistralAIEmbeddings()



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

print(type(tools))
print(len(tools))

    
agent = initialize_agent(
    # agent=AgentType.OPENAI_FUNCTIONS,
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
)

agent({"input": "list the names of all the centers. fomat as buullet points"})

# testing
# retriever = index.as_retriever()
# docs = retriever.invoke("fire regulations")
# print(docs)