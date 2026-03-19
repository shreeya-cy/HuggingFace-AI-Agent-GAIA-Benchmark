import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY", "").strip()
if not api_key:
	raise RuntimeError("TAVILY_API_KEY is not set.")

tavily_client = TavilyClient(api_key=api_key)
response = tavily_client.search("Mercedes Sosa discography between 2000 and 2009")
print(response)