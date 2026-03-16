from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

GAIA_API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"


def _request_json(url: str, timeout: int = 15) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def get_gaia_questions(limit: int | None = None, timeout: int = 15) -> list[dict[str, str]]:
    """Fetch GAIA questions from the Hugging Face scoring API."""
    raw_questions = _request_json(f"{GAIA_API_BASE_URL}/questions", timeout=timeout)

    if not isinstance(raw_questions, list):
        raise ValueError("Unexpected questions API response format.")

    normalized: list[dict[str, str]] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        normalized.append(
            {
                "task_id": str(item.get("task_id", "")).strip(),
                "question": str(item.get("question", "")).strip(),
                "level": str(item.get("Level", "")).strip(),
                "file_name": str(item.get("file_name", "")).strip(),
            }
        )

    normalized = [row for row in normalized if row["task_id"] and row["question"]]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        normalized = normalized[:limit]

    return normalized


def get_task_file(task_id: str, timeout: int = 15) -> str:
    """Fetch the task-associated file content (if available) by task_id."""
    clean_task_id = task_id.strip()
    if not clean_task_id:
        raise ValueError("task_id is required")

    url = f"{GAIA_API_BASE_URL}/files/{clean_task_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"file fetch failed: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"file fetch failed: {exc}") from exc

    if "application/json" in content_type:
        return raw_bytes.decode("utf-8", errors="replace")

    return raw_bytes.decode("utf-8", errors="replace")
