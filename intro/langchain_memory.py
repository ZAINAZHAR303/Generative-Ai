from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
  model="gemini-2.0-pro-exp-02-05",
  google_api_key=os.getenv("GEMINI_API_KEY"),
)

memory  = ConversationBufferWindowMemory(k = 2)
chain = ConversationChain(llm=llm, memory=memory)

while True: 
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = chain.invoke(user_input)
    print("Final==>>",response)