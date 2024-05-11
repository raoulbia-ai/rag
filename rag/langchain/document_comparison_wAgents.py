from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

import os
import glob
from config.config import directory_path

class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

tools = []

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

print(files)

# for file in files:
#     loader = PyPDFLoader(file["path"])
#     pages = loader.load_and_split()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     docs = text_splitter.split_documents(pages)
#     embeddings = OpenAIEmbeddings()
#     retriever = FAISS.from_documents(docs, embeddings).as_retriever()

#     # Wrap retrievers in a Tool
#     tools.append(
#         Tool(
#             args_schema=DocumentInput,
#             name=file["name"],
#             description=f"useful when you want to answer questions about {file['name']}",
#             func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
#         )
#     )