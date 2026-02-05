"""Custom exceptions for BetaFit workflows."""


class BetaFitError(Exception):
    """Base exception for BetaFit application."""

    pass


class LangGraphError(BetaFitError):
    """Errors related to LangGraph workflows."""

    pass


class AgentExecutionError(LangGraphError):
    """Errors during agent execution."""

    pass


class ConfigError(BetaFitError):
    """Configuration-related errors."""

    pass


class ToolExecutionError(BetaFitError):
    """Errors during tool execution."""

    pass


class ValidationError(BetaFitError):
    """Data validation errors."""

    pass


class AirtableError(ToolExecutionError):
    """Errors related to Airtable operations."""

    pass


class SupabaseError(ToolExecutionError):
    """Errors related to Supabase operations."""

    pass


class GraphitiError(ToolExecutionError):
    """Errors related to Graphiti knowledge graph operations."""

    pass


class LLMError(BetaFitError):
    """Errors related to LLM API calls."""

    pass
