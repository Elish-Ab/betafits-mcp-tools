from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

load_dotenv()


# define model configuration
config = {
    "model": "gemini-2.5-flash-lite",
    "temperature": 0.3,
    "max_output_tokens": 6000,
    "top_p": 0.8,
    "top_k": 40,
}

# initialize gemini Chat Model
gemini = ChatGoogleGenerativeAI(
    model=config["model"],
    temperature=config["temperature"],
    max_output_tokens=config["max_output_tokens"],
    top_p=config["top_p"],
    top_k=config["top_k"],
)


# function declaration for gemini response
def get_gemini_response(prompt: str) -> AIMessage:
    response = gemini.invoke(prompt)
    return response


if __name__ == "__main__":
    test_prompt = "Explain Agentic AI"
    print(get_gemini_response(test_prompt).content)
