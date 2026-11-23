import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage

# take enviroment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


system_msg=SystemMessage(content="You are a helpful assistant")
human_msg=HumanMessage(content="what is aiagents")
messages=[system_msg, human_msg]

#initailized model
model=init_chat_model(
     "google_genai:gemini-3.6-flash",
     api_key=api_key
 )


# invoke the model
response = model.invoke(messages)
print(response.content[0]["text"])