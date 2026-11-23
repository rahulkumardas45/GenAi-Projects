import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


model = init_chat_model(
     "google_genai:gemini-3.6-flash",
     api_key = api_key

)

#tavily api key
tavily_api_key = os.getenv("TAVILY_API_KEY")


from langchain_tavily import TavilySearch
from pprint import pprint
skils_demand_tools = TavilySearch(
     max_results=5,
     search_depth ="advanced",
     tavily_api_key = tavily_api_key
)


rapid_api_key = os.getenv('RAPID_API_KEY')

#creating search tools

import requests
from langchain.tools import tool


@tool
def search_jobs(skill:str, location:str) -> list:
  """
  search for jobs based on the skill and location
  """

  print("calling jobs search tools")
  print(f"search jobs in {skill} in {location}")
  url = "https://jsearch.p.rapidapi.com/search-v2"

  querystring = {
      "query":f"{skill} jobs in {location}",
      "num_pages":"1",
      "country":"in",
      "job_employment_type":"FULLTIME,INTERN",

      }

  headers = {
    "x-rapidapi-key": rapid_api_key,
    "x-rapidapi-host": "jsearch.p.rapidapi.com",
    "Content-Type": "application/json",
  }

  response = requests.get(url, headers=headers, params=querystring)
  data = response.json()
  print(data)
  jobs = data.get("data", {}).get("jobs", [])
  print(f"found {len(jobs)} jobs")
  # pprint(jobs)
  result=[]
  for job in jobs:
    result.append({
        "job_title":job.get("job_title", ""),
        "company_name":job.get("employer_name", ""),
        "location":job.get("job_city", ""),
        "apply_link":job.get("job_apply_link", "")
    }
    )
  return result

#crating an Agent
system_prompt ="""You are a Skill-to-Career Mapping assistant that helps students understand skill demand and find matching job opportunities.

You have access to these tools:
- skill_demand_tool: Search for industry demand, salary insights, and career trends
- search_jobs: Find actual job listings requiring specific skills

Help the student by researching the skill they ask about and finding relevant opportunities.

Present results in a clean, readable format with clear sections and proper spacing. Include all job details with apply links. Don't use markdown format."""

#adding memory to the agents

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()


from langchain.agents import create_agent

agent = create_agent(
    model = model,
    tools = [skils_demand_tools, search_jobs],
    checkpointer = checkpointer,
    system_prompt=system_prompt,
    debug=True
)

config = {"configurable":{"thread_id":"1"}}

user_query = "What's the demand for sde jobs in the industry and show me related job openings in India"


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_query
        }
    ]
}, config=config)

print(response["messages"][-1].content[0]['text'])