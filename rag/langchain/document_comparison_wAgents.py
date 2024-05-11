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


# """
# If you want to import load_keys.py from any Python script in any subdirectory of the root, you can add the root directory to your PYTHONPATH.
# nano ~/.bashrc
# Add the following line at the end of the file, replacing /path/to/your/root/directory with the actual path to your root directory:
# export PYTHONPATH="${PYTHONPATH}:/teamspace/studios/this_studio"
# Save and close the file.
# Source your .bashrc file to apply the changes: source ~/.bashrc
# make sure to cd into rag
# """
# try:
#     from rag.load_keys import *
#     print("config module is available")
# except ImportError as e:
#     print(e)




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

    
# testing
# retriever = index.as_retriever()
# docs = retriever.invoke("fire regulations")
# print(docs)

agent = initialize_agent(
    # agent=AgentType.OPENAI_FUNCTIONS,
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
)

agent({"input": "list the names of all the centers. fomat as buullet points"})