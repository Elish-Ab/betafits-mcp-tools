from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from langchain_core.output_parsers import StrOutputParser


open_router = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

if __name__ == "__main__":
    response = open_router | parser 
    print(response.invoke("Hello..."))


