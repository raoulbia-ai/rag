import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
HF_API_KEY = os.environ["HUGGINGFACEHUB_API_TOKEN"]
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")