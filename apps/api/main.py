"""FastAPI entrypoint for local tool testing."""
from __future__ import annotations

import asyncio
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from lib.context_engine import DEFAULT_PCF_TABLE
from services.code_generator.run import run as run_code_generator
from services.code_reviewer.run import run as run_code_reviewer
from services.pcf_parser.run import run as run_pcf_parser
from workflows.langgraph.orchestrator.graph import workflow

load_dotenv()

app = FastAPI(title="Betafits MCP Tools API")


class CodeGeneratorRequest(BaseModel):
    message: str = Field(..., min_length=1)
    persist: bool = True
    repo_name: Optional[str] = None
    pcf_record_id: Optional[str] = None
    pcf_table: str = DEFAULT_PCF_TABLE
    context_files: Optional[List[str]] = None


class CodeReviewerRequest(BaseModel):
    message: str = Field(..., min_length=1)
    persist: bool = True
    pcf_record_id: Optional[str] = None
    pcf_table: str = DEFAULT_PCF_TABLE


class PcfParserRequest(BaseModel):
    message: Optional[str] = ""
    pcf_record_id: Optional[str] = None
    pcf_table: str = DEFAULT_PCF_TABLE
    transcript: Optional[str] = None
    meeting_record_id: Optional[str] = None
    top_k: int = 5
    record_type: str = "Meeting"


class BrainRequest(BaseModel):
    message: Optional[str] = ""
    pcf_record_id: Optional[str] = None
    pcf_table: str = DEFAULT_PCF_TABLE
    context_files: Optional[List[str]] = None
    repo_id: Optional[str] = None
    repo_name: Optional[str] = None
    record_id: Optional[str] = None
    meeting_record_id: Optional[str] = None
    transcript: Optional[str] = None
    source_table: Optional[str] = None
    source_record_id: Optional[str] = None
    repo_url: Optional[str] = None
    repo_github_id: Optional[str] = None


@app.get("/")
async def root(
    record_id: Optional[str] = Query(default=None, alias="recordId"),
    pcf_record_id: Optional[str] = Query(default=None, alias="pcfRecordId"),
    pcf_table: str = Query(default=DEFAULT_PCF_TABLE, alias="pcfTable"),
    message: Optional[str] = Query(default="", alias="message"),
    repo_id: Optional[str] = Query(default=None, alias="repoId"),
    repo_name: Optional[str] = Query(default=None, alias="name"),
    pcf_id: Optional[str] = Query(default=None, alias="pcf_id"),
    pcf: Optional[str] = Query(default=None, alias="pcf"),
    source_table: Optional[str] = Query(default=None, alias="sourceTable"),
    source_record_id: Optional[str] = Query(default=None, alias="sourceRecordId"),
    repo_url: Optional[str] = Query(default=None, alias="repoUrl"),
    repo_github_id: Optional[str] = Query(default=None, alias="repoGithubId"),
) -> dict:
    resolved_record_id = pcf_record_id or pcf_id or record_id
    resolved_pcf_table = pcf_table
    if pcf:
        resolved_pcf_table = pcf

    if resolved_record_id or message:
        initial_state = {
            "message": message or "",
            "next_node": None,
            "code_generator_output": None,
            "pcf_parser_output": None,
            "error": None,
            "pcf_record_id": resolved_record_id,
            "pcf_table": resolved_pcf_table,
            "repo_id": repo_id,
            "repo_name": repo_name,
            "record_id": record_id,
            "source_table": source_table,
            "source_record_id": source_record_id,
            "repo_url": repo_url,
            "repo_github_id": repo_github_id,
        }
        return await workflow.ainvoke(initial_state)
    return {"status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/code-generator")
def code_generator(request: CodeGeneratorRequest) -> dict:
    return run_code_generator(
        request.message,
        use_chain=True,
        persist=request.persist,
        repo_name=request.repo_name,
        pcf_record_id=request.pcf_record_id,
        pcf_table=request.pcf_table,
        context_files=request.context_files,
    )


@app.post("/tools/code-reviewer")
def code_reviewer(request: CodeReviewerRequest) -> dict:
    return run_code_reviewer(
        request.message,
        persist=request.persist,
        pcf_record_id=request.pcf_record_id,
        pcf_table=request.pcf_table,
    )


@app.post("/tools/pcf-parser")
def pcf_parser(request: PcfParserRequest) -> dict:
    if (
        not request.message
        and not request.pcf_record_id
        and not request.transcript
        and not request.meeting_record_id
    ):
        raise HTTPException(
            status_code=400,
            detail="message, pcf_record_id, transcript, or meeting_record_id is required",
        )
    return run_pcf_parser(
        message=request.message or "",
        pcf_record_id=request.pcf_record_id,
        pcf_table=request.pcf_table,
        transcript=request.transcript,
        meeting_record_id=request.meeting_record_id,
        top_k=request.top_k,
        record_type=request.record_type,
    )


async def _run_brain_workflow(initial_state: dict) -> None:
    """
    Runs the workflow in the background.

    IMPORTANT:
    - We catch exceptions here because Airtable will no longer be waiting for the response.
    - Replace print() with your logger if you have one.
    """
    try:
        await workflow.ainvoke(initial_state)
    except Exception as e:
        print("brain workflow failed:", repr(e))


@app.post("/brain")
async def brain(request: BrainRequest, background_tasks: BackgroundTasks) -> dict:
    if (
        not request.message
        and not request.pcf_record_id
        and not request.meeting_record_id
        and not request.transcript
        and not request.repo_id
        and not request.repo_name
        and not request.repo_url
        and not request.repo_github_id
        and not request.source_record_id
    ):
        raise HTTPException(
            status_code=400,
            detail="message, pcf_record_id, meeting_record_id, transcript, repo context, or source_record_id is required",
        )

    initial_state = {
        "message": request.message or "",
        "next_node": None,
        "code_generator_output": None,
        "pcf_parser_output": None,
        "error": None,
        "pcf_record_id": request.pcf_record_id,
        "pcf_table": request.pcf_table,
        "repo_id": request.repo_id,
        "repo_name": request.repo_name,
        "record_id": request.record_id,
        "meeting_record_id": request.meeting_record_id,
        "transcript": request.transcript,
        "source_table": request.source_table,
        "source_record_id": request.source_record_id,
        "repo_url": request.repo_url,
        "repo_github_id": request.repo_github_id,
    }
    if request.context_files:
        initial_state["context_files"] = request.context_files

    # Generate a job_id so Airtable can log "sent" deterministically.
    job_id = str(uuid.uuid4())

    # BackgroundTasks expects a sync callable. We run the async work in a fresh loop when needed.
    def _kickoff() -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_run_brain_workflow(initial_state))
        else:
            loop.create_task(_run_brain_workflow(initial_state))

    background_tasks.add_task(_kickoff)

    # Return immediately so Airtable doesn't time out.
    # 202 is the most semantically correct, but returning 200 also works.
    return {"status": "accepted", "job_id": job_id}
