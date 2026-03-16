from __future__ import annotations

import ast
import argparse
import csv
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics
import sys
from typing import Any
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pypdf import PdfReader
from questions import GAIA_API_BASE_URL, get_gaia_questions, get_task_file
from youtube_transcript_api import YouTubeTranscriptApi


load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


def load_system_prompt() -> str:
    try:
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read system prompt file: {SYSTEM_PROMPT_PATH}: {exc}") from exc

    if not prompt:
        raise RuntimeError(f"System prompt file is empty: {SYSTEM_PROMPT_PATH}")
    return prompt


SYSTEM_PROMPT = load_system_prompt()

search_client = DuckDuckGoSearchRun()
wikipedia_client = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "mean": statistics.mean,
    "median": statistics.median,
    "min": min,
    "round": round,
    "sqrt": math.sqrt,
    "stdev": statistics.stdev,
    "sum": sum,
    "variance": statistics.variance,
}
PYTHON_EXECUTOR_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "len": len,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "enumerate": enumerate,
    "zip": zip,
    "math": math,
    "statistics": statistics,
}


@tool
def web_search(query: str) -> str:
    """Search the web for recent information about a topic."""
    return search_client.invoke(query)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background information about a topic."""
    return wikipedia_client.invoke(query)


def safe_calculate(expression: str) -> float | int:
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
    }
    allowed_unaryops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.List):
            return [evaluate(element) for element in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(element) for element in node.elts)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return allowed_binops[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unaryops:
            return allowed_unaryops[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function_name = node.func.id
            if function_name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Unsupported function: {function_name}")
            args = [evaluate(arg) for arg in node.args]
            return ALLOWED_FUNCTIONS[function_name](*args)
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    return evaluate(parsed)


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic and basic statistical expressions like 12*(4+3), mean([1,2,3]), median([2,4,9]), or stdev([1,5,7])."""
    try:
        result = safe_calculate(expression)
    except Exception as exc:
        return f"Calculator error: {exc}"
    return str(result)


@tool
def fetch_webpage(url: str) -> str:
    """Fetch and extract readable text content from a webpage URL."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return "Webpage fetch error: URL must start with http:// or https://"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as exc:
        return f"Webpage fetch error: {exc}"

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Webpage fetch error: no readable text found on the page."

    return text[:4000]


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    return text or "No readable text found in the PDF."


def read_csv_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        return "CSV file is empty."

    preview_rows = rows[:20]
    return "\n".join(",".join(row) for row in preview_rows)


def read_json_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return json.dumps(payload, indent=2)[:4000]


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        return file.read(4000)


@tool
def read_file_content(file_path: str) -> str:
    """Read the contents of a local pdf, csv, txt, or json file from a file path."""
    path = Path(file_path).expanduser()

    if not path.exists():
        return f"File read error: file not found at {path}"
    if not path.is_file():
        return f"File read error: path is not a file: {path}"

    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            content = read_pdf_file(path)
        elif suffix == ".csv":
            content = read_csv_file(path)
        elif suffix == ".json":
            content = read_json_file(path)
        elif suffix == ".txt":
            content = read_text_file(path)
        else:
            return "File read error: only .pdf, .csv, .txt, and .json files are supported."
    except Exception as exc:
        return f"File read error: {exc}"

    return content[:4000]


def extract_youtube_video_id(url: str) -> str | None:
    parsed_url = urlparse(url)
    host = parsed_url.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed_url.path.strip("/")
        return video_id or None

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        if parsed_url.path.startswith("/shorts/") or parsed_url.path.startswith("/embed/"):
            parts = [part for part in parsed_url.path.split("/") if part]
            return parts[1] if len(parts) > 1 else None

    return None


@tool
def youtube_transcript(url: str) -> str:
    """Fetch the transcript of a YouTube video from its URL."""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return "YouTube transcript error: unsupported or invalid YouTube URL."

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    except Exception as exc:
        return f"YouTube transcript error: {exc}"

    text = " ".join(snippet.text.strip() for snippet in transcript if snippet.text.strip())
    if not text:
        return "YouTube transcript error: no transcript text available for this video."
    return text[:4000]


@tool
def python_executor(code: str) -> str:
    """Execute small Python snippets for complex calculations. Put the final value in a variable named result or print it."""
    stdout_buffer = io.StringIO()
    local_scope: dict[str, object] = {}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, PYTHON_EXECUTOR_GLOBALS, local_scope)
    except Exception as exc:
        return f"Python executor error: {exc}"

    stdout_output = stdout_buffer.getvalue().strip()
    result_value = local_scope.get("result")

    parts = []
    if stdout_output:
        parts.append(f"stdout:\n{stdout_output}")
    if result_value is not None:
        parts.append(f"result:\n{result_value}")

    if not parts:
        return "Python executor completed with no output. Use print(...) or assign the final value to `result`."

    return "\n\n".join(parts)[:4000]


def run_with_tools(prompt: str) -> str:
    tools = [
        web_search,
        wikipedia_search,
        calculator,
        python_executor,
        fetch_webpage,
        read_file_content,
        youtube_transcript,
    ]
    tools_by_name = {tool.name: tool for tool in tools}

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    for tool_call in response.tool_calls:
        selected_tool = tools_by_name.get(tool_call["name"])
        if selected_tool is None:
            continue

        tool_result = selected_tool.invoke(tool_call["args"])
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
            )
        )

    if response.tool_calls:
        response = llm_with_tools.invoke(messages)

    return response.content


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def attach_gaia_task_files(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []

    for item in questions:
        row = dict(item)
        task_id = row.get("task_id", "")
        file_name = row.get("file_name", "")

        row["task_file_content"] = ""
        row["task_file_error"] = ""

        if file_name and task_id:
            try:
                row["task_file_content"] = get_task_file(task_id)[:4000]
            except Exception as exc:
                row["task_file_error"] = str(exc)

        enriched.append(row)

    return enriched


def run_gaia_batch(limit: int = 20) -> None:
    questions_path = Path("questions.json")
    answers_path = Path("answers.json")

    base_questions = get_gaia_questions(limit=limit)
    questions = attach_gaia_task_files(base_questions)
    fetched_at = datetime.now(timezone.utc).isoformat()
    questions_payload = {
        "source": f"{GAIA_API_BASE_URL}/questions",
        "fetched_at": fetched_at,
        "count": len(questions),
        "questions": questions,
    }
    write_json(questions_path, questions_payload)

    run_started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        start_time = time.perf_counter()
        task_id = item.get("task_id", "")
        question = item.get("question", "")
        level = item.get("level", "")
        file_name = item.get("file_name", "")
        task_file_content = item.get("task_file_content", "")
        task_file_error = item.get("task_file_error", "")
        status = "ok"
        error = ""
        model_answer = ""

        gaia_prompt = (
            "You are answering one GAIA benchmark question. "
            "Use tools when needed and return only the final answer.\n"
            f"task_id: {task_id}\n"
            f"level: {level}\n"
            f"file_name: {file_name or 'none'}\n"
            f"task_file_error: {task_file_error or 'none'}\n"
            f"task_file_content: {task_file_content or 'none'}\n"
            f"question: {question}"
        )

        try:
            model_answer = run_with_tools(gaia_prompt).strip()
        except Exception as exc:
            status = "error"
            error = str(exc)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result_item = {
            "index": index,
            "task_id": task_id,
            "level": level,
            "file_name": file_name,
            "task_file_error": task_file_error,
            "question": question,
            "model_answer": model_answer,
            "status": status,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result_item)

        answers_payload = {
            "model": MODEL_NAME,
            "run_started_at": run_started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(questions),
            "processed": len(results),
            "answers": results,
        }
        write_json(answers_path, answers_payload)
        print(f"[{index}/{len(questions)}] task_id={task_id} status={status}")

    failed = sum(1 for row in results if row["status"] == "error")
    print(f"Saved {len(questions)} questions to {questions_path}")
    print(f"Saved {len(results)} answers to {answers_path}")
    print(f"Failed tasks: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-enabled chatbot with GAIA batch mode")
    parser.add_argument(
        "--gaia-limit",
        type=int,
        default=20,
        help="Number of GAIA questions to process when running batch mode (default: 20).",
    )
    parser.add_argument(
        "--quick-test",
        nargs="?",
        const=5,
        type=int,
        help="Run GAIA batch in quick mode with a small question count (default: 5).",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text for normal chat mode")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt = " ".join(args.prompt).strip()

    if not prompt:
        limit = args.quick_test if args.quick_test is not None else args.gaia_limit
        if limit < 1:
            print("GAIA limit must be >= 1")
            return
        run_gaia_batch(limit=limit)
        return

    response = run_with_tools(prompt)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt}")
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
