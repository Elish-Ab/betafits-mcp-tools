from models.gemini_client import gemini
#from pcf_parser.models import open_router
from langchain_core.tools import StructuredTool
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
import json
import os

output_parser = StrOutputParser()
class SummarizedPCFParserOutput(BaseModel):
    pcf_id: str
    meeting_id: str
    relevance_score: float = 0.0
    action_items: Optional[list] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    summary: str
    

# Load prompts from config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    PROMPTS = CONFIG.get("prompts", {})


def summarize_text_(text: str) -> str:
    """Simple text summarization (legacy function for backward compatibility)."""
    # Load prompt template from config
    prompt_config = PROMPTS.get("summarize_text", {})
    prompt_template = prompt_config.get(
        "template", "Summarize the following text:\n\n{text}\n\nSummary:"
    )

    prompt = prompt_template.format(text=text)
    response = gemini | output_parser
    #response = open_router | output_parser
    return response.invoke(prompt)


def summarize_pcf_meeting(
    meeting_transcript: str,
    pcf_id: str,
    pcf_name: str,
    pcf_type: str,
    pcf_summary: str,
    meeting_id: Optional[str] = None,
    project_name: Optional[str] = None,
    component_name: Optional[str] = None,
    feature_name: Optional[str] = None,
) -> Dict:
    """
    Summarize the parts of a meeting transcript relevant to a specific PCF.
    """

    # Build PCF context
    pcf_context = (
        f"- PCF Name: {pcf_name}\n"
        f"- PCF Type: {pcf_type}\n"
        f"- PCF ID: {pcf_id}"
    )
    if project_name:
        pcf_context += f"\n- Project: {project_name}"
    if component_name:
        pcf_context += f"\n- Component: {component_name}"
    if feature_name:
        pcf_context += f"\n- Feature: {feature_name}"

    pcf_context += f'\n- PCF Summary: """{pcf_summary}"""'

    # Load prompt template
    prompt_config = PROMPTS.get("summarize_pcf_meeting", {})
    prompt_template = prompt_config.get("template", "")

    prompt = prompt_template.format(
        meeting_transcript=meeting_transcript,
        pcf_context=pcf_context,
        pcf_id=pcf_id,
        meeting_id=meeting_id or "N/A",
        timestamp=datetime.now().isoformat(),
    )

    # FIX: Use PydanticOutputParser instead of BaseModel
    parser = PydanticOutputParser(pydantic_object=SummarizedPCFParserOutput)
    chain = gemini | parser

    try:
        # Response is now a Pydantic object, NOT raw JSON string
        result: SummarizedPCFParserOutput = chain.invoke(prompt)

        print("\n[DEBUG STRUCTURED OUTPUT] ===============================")
        print(result.summary)
        print("[DEBUG END] =============================================\n")

        return {
            "pcf_id": result.pcf_id,
            "meeting_id": result.meeting_id,
            "summary": result.summary,
            "relevance_score": result.relevance_score,
            "action_items": result.action_items,
            "timestamp": result.timestamp,
        }

    except Exception as e:
        print(f"❌ Error generating summary for {pcf_name}: {e}")
        # Re-raise so the workflow node fails instead of saving "Error generating summary"
        raise e

class SummarizeInput(BaseModel):
    text: str = Field(description="The text to be summarized.")


class SummarizePCFMeetingInput(BaseModel):
    meeting_transcript: str = Field(description="Full meeting transcript text")
    pcf_id: str = Field(description="PCF record ID")
    pcf_name: str = Field(description="Name of the PCF")
    pcf_type: str = Field(description="Type (Project/Component/Feature)")
    pcf_summary: str = Field(description="Description/summary of the PCF")
    meeting_id: Optional[str] = Field(None, description="Optional meeting identifier")
    project_name: Optional[str] = Field(None, description="Optional project name")
    component_name: Optional[str] = Field(None, description="Optional component name")
    feature_name: Optional[str] = Field(None, description="Optional feature name")


summarize_tool = StructuredTool.from_function(
    func=summarize_text_,
    name="summarize_text",
    description="Summarizes the provided text into a concise summary.",
    return_direct=True,
    args_schema=SummarizeInput,
)

summarize_pcf_meeting_tool = StructuredTool.from_function(
    func=summarize_pcf_meeting,
    name="summarize_pcf_meeting",
    description="Summarize the parts of a meeting transcript relevant to a specific PCF with structured output including relevance score and action items.",
    return_direct=True,
    args_schema=SummarizePCFMeetingInput,
)


if __name__ == "__main__":
    # Test legacy summarizer
    test_text = (
        "Pathways is a new AI architecture developed by Google that allows a single model to "
        "handle many tasks at once, learn new tasks quickly, and understand complex concepts. "
        "Unlike traditional models that are trained for specific tasks, Pathways can generalize "
        "across different domains and modalities, making it more versatile and efficient. It uses "
        "a combination of techniques such as sparse activation, modularity, and multi-task learning "
        "to achieve this. Pathways aims to create AI systems that are more aligned with human-like "
        "learning and reasoning capabilities."
    )
    # print("Legacy summarizer:")
    # print(summarize_tool.invoke({"text": test_text}))

    # Test new PCF meeting summarizer
    print("\n\nPCF Meeting Summarizer:")
    test_result = summarize_pcf_meeting(
        meeting_transcript="We discussed the new transformation engine for handling census data conversions...",
        pcf_id="rec123",
        pcf_name="Transformation Engine",
        pcf_type="Component",
        pcf_summary="Universal file transformation system for Megacensus",
        meeting_id="mtg_001",
    )
    print(json.dumps(test_result, indent=2))
