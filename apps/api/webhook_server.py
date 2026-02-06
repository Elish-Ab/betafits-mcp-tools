# moved from pcf parser repo
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from services.pcf_parser.run import run as run_pcf_parser_service

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PCF Parser Webhook")

class TriggerPayload(BaseModel):
    record_id: str

def process_meeting_task(record_id: str):
    """Background task to fetch transcript and run workflow."""
    try:
        logger.info(f"Triggering workflow for meeting record: {record_id}")
        result = run_pcf_parser_service(meeting_record_id=record_id, top_k=5, record_type="Meeting")
        
        # 4. Optional: Update Airtable status?
        # We could add a 'Parser Status' field and update it here.
        logger.info(f"Workflow complete for {record_id}. Result: {result}")
        
    except Exception as e:
        logger.error(f"Error processing record {record_id}: {e}", exc_info=True)

@app.post("/trigger-parse")
async def trigger_parse(payload: TriggerPayload, background_tasks: BackgroundTasks):
    """Endpoint triggered by Airtable Automation."""
    logger.info(f"Received trigger for record_id: {payload.record_id}")
    background_tasks.add_task(process_meeting_task, payload.record_id)
    return {"status": "accepted", "message": f"Processing record {payload.record_id} in background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
