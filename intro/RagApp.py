from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator

import os
from dotenv import load_dotenv

load_dotenv()


llm = GoogleGenerativeAI(
  model="gemini-2.0-pro-exp-02-05",
  google_api_key=os.getenv("GEMINI_API_KEY"),
)

try:
    loader = TextLoader("./climate.txt")
except Exception as e:
    print("error",e)


text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))

index_creator = VectorstoreIndexCreator(embedding=embedding, text_splitter=text_splitter)
store = index_creator.from_loaders([loader])

while True:
    human_message = input("How can I help you today? ")
    response = store.query(human_message, llm=llm,)
    print(response)