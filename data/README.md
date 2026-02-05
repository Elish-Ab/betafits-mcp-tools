# Configuration Guide

## Overview
This directory contains `config.json`, which serves as the **single source of truth** for all tool descriptions and LLM prompts used throughout the BetaFit codebase.

## Why Config-Driven?
- ✅ **Easy Updates**: Change prompts without touching code
- ✅ **Version Control**: Track prompt changes through git
- ✅ **Consistency**: All tools and prompts in one place
- ✅ **Experimentation**: A/B test different prompts easily

## Structure

### `config.json`

```json
{
  "tools": {
    "tool_name": {
      "name": "tool_name",
      "description": "What this tool does",
      "return_description": "What it returns"
    }
  },
  "prompts": {
    "prompt_name": {
      "template": "The actual prompt with {variables}",
      "description": "What this prompt is for",
      "variables": ["list", "of", "variables"]
    }
  }
}
```

## How to Update Prompts

### Example 1: Update the Summarization Prompt

**Current prompt** (in `config.json`):
```json
"summarize_text": {
  "template": "Summarize the following text:\n\n{text}\n\nSummary:",
  "description": "Simple text summarization without context",
  "variables": ["text"]
}
```

**To change it**, just edit the `template` field:
```json
"summarize_text": {
  "template": "Provide a concise summary of this text in 2-3 sentences:\n\n{text}\n\nConcise Summary:",
  "description": "Simple text summarization without context",
  "variables": ["text"]
}
```

✅ **That's it!** No code changes needed. The change takes effect immediately on next run.

### Example 2: Update the Re-ranking Prompt

The re-ranking prompt is more complex with multiple variables:

```json
"rerank_with_llm": {
  "template": "You are an expert...\n\nUSER QUERY SUMMARY:\n{query_summary}\n\nCANDIDATES:\n{candidate_texts}...",
  "description": "LLM-based re-ranking of search results",
  "variables": ["query_summary", "candidate_texts"]
}
```

**To modify scoring criteria**, edit the template:
- Change the **SCORING GUIDELINES** section
- Adjust score thresholds (e.g., HIGH RELEVANCE from 0.8-1.0 to 0.85-1.0)
- Add new examples
- Modify output format

### Example 3: Add a New Prompt

1. Add to `config.json`:
```json
"prompts": {
  "your_new_prompt": {
    "template": "Your prompt with {variable1} and {variable2}",
    "description": "What this prompt does",
    "variables": ["variable1", "variable2"]
  }
}
```

2. In your code:
```python
# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    PROMPTS = CONFIG.get("prompts", {})

# Use the prompt
prompt_config = PROMPTS.get("your_new_prompt", {})
prompt_template = prompt_config.get("template", "")
prompt = prompt_template.format(variable1="value1", variable2="value2")
```

## Available Prompts

### 1. `extract_key_topics`
- **Purpose**: Extract main topics from meeting transcripts
- **Variables**: `transcript`
- **Used in**: `tools/semantic_match.py` → `extract_key_topics_from_transcript()`

### 2. `rerank_with_llm`
- **Purpose**: Re-rank search results for accuracy
- **Variables**: `query_summary`, `candidate_texts`
- **Used in**: `tools/semantic_match.py` → `rerank_with_llm()`

### 3. `summarize_text`
- **Purpose**: Simple text summarization
- **Variables**: `text`
- **Used in**: `tools/summarize.py` → `summarize_text_()`

### 4. `summarize_pcf_meeting`
- **Purpose**: Summarize meeting in context of a PCF
- **Variables**: `meeting_transcript`, `pcf_context`, `pcf_id`, `meeting_id`, `timestamp`
- **Used in**: `tools/summarize.py` → `summarize_pcf_meeting()`

## Best Practices

### 1. Test Before Deploying
Always test prompt changes with sample data before using in production.

### 2. Document Changes
Use git commit messages to explain why you changed a prompt:
```bash
git commit -m "prompts: improve extract_key_topics to focus on action items"
```

### 3. Keep Variables Clear
Use descriptive variable names: `{meeting_transcript}` not `{text}`

### 4. Version Important Prompts
For critical prompts, consider keeping old versions commented:
```json
"rerank_with_llm": {
  "template": "New improved prompt...",
  "description": "v2: Added strict scoring criteria (2025-10-29)",
  "variables": ["query_summary", "candidate_texts"],
  "_old_template_v1": "Previous version..."
}
```

### 5. Format for Readability
Use `\n` for line breaks in JSON strings:
```json
"template": "Line 1\n\nLine 2 with blank line above\nLine 3"
```

## Troubleshooting

### Prompt not updating?
1. Check JSON syntax is valid
2. Ensure server/process restarted after config change
3. Verify variable names match between template and code

### Variables not being replaced?
Check that `.format()` call includes all variables:
```python
prompt = prompt_template.format(
    meeting_transcript=transcript,  # ✅ All variables provided
    pcf_context=context
)
```

### JSON parsing errors?
- Escape special characters: `\"` for quotes inside strings
- Use single `\n` for newlines, not `\\n` in JSON
- Validate JSON: https://jsonlint.com/

## Examples of Good Prompts

### Clear Instructions
```
"Task: Summarize the meeting focusing on:
1. Key decisions
2. Action items
3. Blockers"
```

### Structured Output
```
"Output your answer as JSON:
{
  \"summary\": \"...\",
  \"score\": 0.85
}"
```

### Examples in Prompt
```
"Examples:
- Query: 'Invoice processing' → Invoice Extractor (0.95)
- Query: 'Invoice processing' → Generic automation (0.4)"
```

## Need Help?
- Check existing prompts in `config.json` for examples
- Test prompts manually in ChatGPT/Claude first
- Ask team for prompt engineering best practices
