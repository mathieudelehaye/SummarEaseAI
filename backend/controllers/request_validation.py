from typing import Any, Dict, Optional, Tuple


def validate_summarize_request(
    data: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Validate single-source summarization request.
    Returns (error dict, status code) if invalid, None if valid.
    """
    if not data or "query" not in data:
        return {"error": "Missing query parameter"}, 400

    query = data["query"].strip()
    max_lines = data.get("max_lines", 30)

    if not query:
        return {"error": "Empty query provided"}, 400
    if max_lines < 5 or max_lines > 100:
        return {"error": "max_lines must be between 5 and 100"}, 400

    return None


def validate_multi_source_request(
    data: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Validate multi-source summarization request.
    Returns (error dict, status code) if invalid, None if valid.
    """
    if not data or "query" not in data:
        return {"error": "Missing query parameter"}, 400

    query = data["query"].strip()
    max_lines = data.get("max_lines", 30)
    max_articles = data.get("max_articles", 3)
    cost_mode = data.get("cost_mode", "BALANCED")

    if not query:
        return {"error": "Empty query provided"}, 400
    if max_lines < 5 or max_lines > 100:
        return {"error": "max_lines must be between 5 and 100"}, 400
    if max_articles < 1 or max_articles > 10:
        return {"error": "max_articles must be between 1 and 10"}, 400
    if cost_mode not in ["MINIMAL", "BALANCED", "COMPREHENSIVE"]:
        return {"error": "cost_mode must be MINIMAL, BALANCED, or COMPREHENSIVE"}, 400

    return None
