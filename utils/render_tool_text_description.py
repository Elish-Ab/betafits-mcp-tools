from langchain_core.tools import render_text_description_and_args
from tools import store_to_rag, query_rag

tools = [store_to_rag, query_rag]


def render_tool_args_description(tools: list) -> str:
    rendered_tools = render_text_description_and_args(tools)
    return rendered_tools


if __name__ == "__main__":
    description = render_tool_args_description(tools)
    print(description)
