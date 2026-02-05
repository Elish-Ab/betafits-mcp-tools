import logging
from .gemini_client import gemini
from .open_router_client import open_router

__all__ = ["gemini","open_router"]

"""
Models package initialization.
This file imports and exposes the Gemini model for easy access.
"""


logging.getLogger(__name__).addHandler(logging.NullHandler())
