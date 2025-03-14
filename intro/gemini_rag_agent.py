from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader

import os
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

# Set the path to the service account key file
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_credentials_path is None:
    raise EnvironmentError("GOOGLE_APPLICATIONS_CREDENTIALS_PATH environment variable not set.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    google_api_key=os.getenv("GEMINI_API_KEY") )

search = TavilySearchResults(tavily_api_key = os.getenv("TAVILY_API_KEY"))

# Set up retry strategy
retry_strategy = Retry(
    total=3,  # Number of retries
    backoff_factor=1,  # A delay between retries
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
    raise_on_status=False,
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# Modify the loader to use the session with retries
loader = WebBaseLoader("https://www.techloset.com/", session=http)

docs = loader.load()
documents =  RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200).split_documents(docs)

vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "techloset_search",
    "Search for information about Techloset. For any questions about Techloset Solutions, you must use this tool!",
)

tools = [search,retriever_tool]
prompt =  hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agents_executer = AgentExecutor(agent = agent, verbose=True)

while True:
    agents_executer.invoke({"input" : input("How i cna help you today? : ")})