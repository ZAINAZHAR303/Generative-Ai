from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
import requests
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
  model="gemini-2.0-pro-exp-02-05",
  google_api_key=os.getenv("GEMINI_API_KEY"),
)



# Manually invoke the LLM


@tool
def get_weather_tool(city: str) -> str:
    """Get the weather of a city."""
    print("get_weather_tool input_data:", city)
    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=36717966e1d6486e88452016240107&q={city}"
        )
        response.raise_for_status()  # Raise an error for bad responses
        weather_data = response.json()
        return f"The weather in {city} is {weather_data['current']['condition']['text']} with a temperature of {weather_data['current']['temp_c']}°C."
    except Exception as e:
        return f"Error fetching weather data: {e}"


city = "Faisalabad"
weather = get_weather_tool(city)
prompt = f"Tell me about the weather: {weather}"

# Run the LLM directly
output = llm.invoke(prompt)
print("Output:", output)
# # Initialize the agent with the tool
# agent = initialize_agent(
#     tools=[get_weather_tool],
#     agent=AgentType.OPENAI_FUNCTIONS,
#     llm=llm,
#     verbose=True,
#     max_iterations =1  # Add this to handle parsing errors gracefully
# )

# # Invoke the agent
# output = agent.run("Find the weather of the city Faisalabad.")
# print("Output:", output)