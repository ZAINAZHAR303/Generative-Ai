from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# === TOOLS ===
def get_weather_tool(city: str) -> str:
    """Get the weather of a city."""
    print("get_weather_tool input_data:", city)
    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=36717966e1d6486e88452016240107&q={city}"
        )
        response.raise_for_status()
        weather_data = response.json()
        return f"The weather in {city} is {weather_data['current']['condition']['text']} with a temperature of {weather_data['current']['temp_c']}°C."
    except Exception as e:
        return f"Error fetching weather data: {e}"

def calculate_square(number: int) -> str:
    """Calculates the square of a number."""
    return f"The square of {number} is {number ** 2}."

# Define tools properly
tools = [
    Tool(
        name="get_weather_tool",
        func=get_weather_tool,
        description="Get the weather of a city.",
    ),
    Tool(
        name="calculate_square",
        func=calculate_square,
        description="Calculates the square of a number.",
    ),
]

# === CUSTOM AGENT ===
# Custom prompt avoids system messages
# Custom prompt avoids system messages


prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
You are a helpful AI assistant. Answer the user’s question or perform the task using the available tools if necessary.

User's input: {input}

{agent_scratchpad}
"""
)

# Create the agent
custom_agent = create_openai_functions_agent(llm, tools, prompt)

# Agent Executor: Decides to use tools or LLM
agent_executor = AgentExecutor(
    agent=custom_agent,
    tools=tools,
    verbose=True
)


while True:
 inp = input("you :  ")
 if inp == "exit":
        break
 weather_output = agent_executor.invoke({"input" : inp})
 print("Weather Output:", weather_output)

# # Calculation request (triggers calculate_square)
# calc_output = agent_executor.invoke({"input": "Calculate the square of 8."})
# print("Calculation Output:", calc_output)

# # General query (handled by LLM, no tool needed)
# general_output = agent_executor.invoke({"input": "Tell me a fun fact about space."})
# print("General Output:", general_output)
