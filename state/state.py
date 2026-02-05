from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class PCFParserState(TypedDict):
    transcript: str
    top_k: int
    type: str
    pcf_records: List[Dict[str, Any]]
    mapped_pcfs: List[Dict[str, Any]]
    created_parser_records: List[Dict[str, Any]]
    rag_result: Any
    graphiti_result: Any
    errors: List[str]
