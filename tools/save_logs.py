from langchain_core.tools import StructuredTool
from supabase_rag import SUPABASE_CLIENT_
from pydantic import BaseModel, Field


class SaveLogInput(BaseModel):
    log_info: str = Field(description="The log message to save.")
    origin: str = Field(description="The origin/source of the log message.")


def save_log_(log_info: str, origin: str) -> str:
    """
    This function stores workflow logs
    Args:
        log_info: string for event happened
        origin: from which part of the workflow

    Returns:
        Status message indicating success or failure
    """
    try:
        response = (
            SUPABASE_CLIENT_.table("logs")
            .insert({"log_info": log_info, "origin": origin})
            .execute()
        )
        if response.data is not None:
            print("Log saved successfully.")
            return f"Log saved. data: {response.data}"
        else:
            print(f"Failed to save log. Status code: {response.status_code}")
            return "Log not saved."
    except Exception as e:
        print(f"Error saving log: {e}")
        return f"Error saving log: {e}"


save_log_tool = StructuredTool.from_function(
    func=save_log_,
    name="save_log",
    description="Saves a log message to the Supabase database.",
    return_direct=True,
    args_schema=SaveLogInput,
)

if __name__ == "__main__":
    print(save_log_tool.args_schema)
    print("invoke dummy tool call")
    # save_log_tool.invoke({"log_info": "Test log message", "origin": "UnitTest"})
