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
from urllib.parse import parse_qs, quote, urlparse


from bs4 import BeautifulSoup

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pypdf import PdfReader
from questions import GAIA_API_BASE_URL, get_gaia_questions, get_task_file
from tavily import TavilyClient
from youtube_transcript_api import YouTubeTranscriptApi


load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MAX_TOOL_ITERATIONS = 10
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def load_system_prompt() -> str:
    try:
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read system prompt file: {SYSTEM_PROMPT_PATH}: {exc}") from exc

    if not prompt:
        raise RuntimeError(f"System prompt file is empty: {SYSTEM_PROMPT_PATH}")
    return prompt


SYSTEM_PROMPT = load_system_prompt()

tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
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


def _llm_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _invoke_plain_llm(messages: list[SystemMessage | HumanMessage]) -> str:
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
    )
    response = llm.invoke(messages)
    return _llm_to_text(response.content)


def _optimize_search_query(original_query: str) -> str:
    prompt = (
        "Rewrite the user question into one high-quality web search query. "
        "Return only the query text, no explanations, no quotes, one line only."
    )
    optimized = _invoke_plain_llm(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"User question: {original_query}"),
        ]
    )
    optimized = optimized.strip().strip('"').strip("'")
    optimized = optimized.splitlines()[0].strip() if optimized else ""
    return optimized or original_query


def _site_key_from_url(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_tavily_results(
    original_query: str,
    generated_query: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    normalized_results = []
    seen_sites: set[str] = set()
    raw_results = response.get("results", [])

    if isinstance(raw_results, list):
        for row in raw_results:
            if not isinstance(row, dict):
                continue

            title = str(row.get("title", "")).strip()
            url = str(row.get("url", "")).strip()
            snippet = str(row.get("content", "")).strip()
            score = row.get("score")
            site_key = _site_key_from_url(url)

            if site_key and site_key in seen_sites:
                continue
            if site_key:
                seen_sites.add(site_key)

            normalized_results.append(
                {
                    "rank": len(normalized_results) + 1,
                    "title": title,
                    "url": url,
                    "snippet": snippet[:700],
                    "score": score,
                }
            )
            if len(normalized_results) >= 8:
                break

    return {
        "original_query": original_query,
        "generated_search_query": generated_query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "result_count": len(normalized_results),
        "results": normalized_results,
    }



def _answer_from_structured_web_results(question: str, evidence: dict[str, Any]) -> str:
    evidence_json = json.dumps(evidence, indent=2)
    system_prompt = (
          "Rewrite the user question into a precise web search query.\n"
        "Rules:\n"
        "- Preserve all key entities (names, dates, locations, numbers).\n"
        "- Include important constraints (e.g., 'latest', '2025', 'comparison', etc.).\n"
        "- Make the query specific and unambiguous.\n"
        "- Avoid adding new assumptions.\n"
        "- Use concise keyword-style phrasing (like a search engine query).\n"
        "- Return only ONE query, no explanations, no quotes.\n"
    )
    human_prompt = (
        f"User question:\n{question}\n\n"
        "Structured web evidence (JSON):\n"
        f"{evidence_json}"
    )
    return _invoke_plain_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )


def _safe_json_loads(value: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _format_trace_for_cli(trace: dict[str, Any]) -> str:
    lines: list[str] = []

    tools_used = trace.get("tools_used", [])
    lines.append("Tools used:")
    if tools_used:
        lines.extend(f"- {tool_name}" for tool_name in tools_used)
    else:
        lines.append("- none")

    searches = trace.get("searches", [])
    if searches:
        lines.append("")
        lines.append("Search queries:")
        for search in searches:
            tool_name = str(search.get("tool_name", "")).strip() or "unknown"
            original_query = str(search.get("query", "")).strip()
            generated_query = str(search.get("generated_search_query", "")).strip()
            page_title = str(search.get("page_title", "")).strip()
            error_text = str(search.get("error", "")).strip()

            lines.append(f"- {tool_name}: {original_query or 'n/a'}")
            if generated_query and generated_query != original_query:
                lines.append(f"  generated query: {generated_query}")
            if page_title:
                lines.append(f"  page title: {page_title}")
            if error_text:
                lines.append(f"  error: {error_text}")

            tavily_results = search.get("tavily_results", [])
            if isinstance(tavily_results, list) and tavily_results:
                lines.append("  Tavily results:")
                for row in tavily_results:
                    if not isinstance(row, dict):
                        continue
                    rank = row.get("rank", "?")
                    title = str(row.get("title", "")).strip() or "Untitled"
                    url = str(row.get("url", "")).strip() or "n/a"
                    snippet = str(row.get("snippet", "")).strip()
                    score = row.get("score")
                    score_text = f" | score={score}" if score is not None else ""
                    lines.append(f"  - [{rank}] {title}{score_text}")
                    lines.append(f"    {url}")
                    if snippet:
                        lines.append(f"    {snippet}")

    return "\n".join(lines)


def _http_get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def _strip_html_text(value: str) -> str:
    if BeautifulSoup is None:
        text = re.sub(r"<[^>]+>", " ", value)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    soup = BeautifulSoup(value, "html.parser")
    return soup.get_text("\n", strip=True)


def _search_wikipedia_page(query: str) -> tuple[str | None, str | None]:
    encoded_query = quote(query)
    url = (
        f"{WIKIPEDIA_API_URL}?action=query&list=search&srsearch={encoded_query}"
        "&utf8=1&format=json&srlimit=1"
    )
    payload = _http_get_json(url)
    search_results = payload.get("query", {}).get("search", [])
    if not search_results:
        return None, None

    first_result = search_results[0]
    page_title = str(first_result.get("title", "")).strip()
    page_snippet = _strip_html_text(str(first_result.get("snippet", ""))).strip()
    return page_title or None, page_snippet or None


def _get_wikipedia_sections(page_title: str) -> list[dict[str, str]]:
    encoded_title = quote(page_title)
    url = (
        f"{WIKIPEDIA_API_URL}?action=parse&page={encoded_title}"
        "&prop=sections&format=json"
    )
    payload = _http_get_json(url)
    sections = payload.get("parse", {}).get("sections", [])
    normalized_sections: list[dict[str, str]] = []

    if not isinstance(sections, list):
        return normalized_sections

    for section in sections:
        if not isinstance(section, dict):
            continue
        normalized_sections.append(
            {
                "index": str(section.get("index", "")).strip(),
                "line": str(section.get("line", "")).strip(),
                "number": str(section.get("number", "")).strip(),
                "level": str(section.get("level", "")).strip(),
            }
        )

    return normalized_sections


def _get_wikipedia_section_text(page_title: str, section_index: str) -> str:
    encoded_title = quote(page_title)
    encoded_index = quote(section_index)
    url = (
        f"{WIKIPEDIA_API_URL}?action=parse&page={encoded_title}&prop=text"
        f"&section={encoded_index}&format=json"
    )
    payload = _http_get_json(url)
    raw_html = payload.get("parse", {}).get("text", {}).get("*", "")
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""

    text = _strip_html_text(raw_html)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@tool
def web_search(query: str) -> str:
    """Search the web for recent information about a topic."""
    clean_query = query.strip()
    if not clean_query:
        return "Web search error: query cannot be empty."

    if tavily_client is None:
        return "Web search error: missing TAVILY_API_KEY environment variable."

    try:
        generated_query = _optimize_search_query(clean_query)
    except Exception:
        generated_query = clean_query

    try:
        tavily_response = tavily_client.search(
            query=generated_query,
            search_depth="advanced",
            max_results=8,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        return f"Web search error: {exc}"

    # Keep Tavily output in-memory as structured JSON-like evidence for synthesis.
    structured_web_results = _normalize_tavily_results(
        original_query=clean_query,
        generated_query=generated_query,
        response=tavily_response,
    )

    try:
        answer = _answer_from_structured_web_results(clean_query, structured_web_results)
    except Exception as exc:
        return (
            "Web search synthesis error: "
            f"{exc}\n"
            f"Structured evidence:\n{json.dumps(structured_web_results, indent=2)}"
        )

    payload = {
        "query": clean_query,
        "generated_search_query": generated_query,
        "tavily_results": structured_web_results.get("results", []),
        "answer": answer or "",
    }
    return json.dumps(payload, indent=2)


@tool
def wikipedia_search(query: str) -> str:
    """Resolve a Wikipedia page for the query and return the available sections. Call wikipedia_section_content next for the most relevant section."""

    try:
        generated_query = _optimize_search_query(query.strip())
    except Exception:
        generated_query = query.strip()

    try:
        page_title, page_snippet = _search_wikipedia_page(generated_query)
    except Exception as exc:
        return f"Wikipedia search error: {exc}"

    if not page_title:
        return f"Wikipedia search error: no page found for query '{generated_query}'."

    try:
        sections = _get_wikipedia_sections(page_title)
    except Exception as exc:
        return f"Wikipedia search error: unable to list sections for '{page_title}': {exc}"

    payload = {
        "query": query.strip(),
        "generated_search_query": generated_query,
        "page_title": page_title,
        "page_snippet": page_snippet or "",
        "sections": sections,
        "instruction": (
            "Choose the most relevant section index for the user's query, then call "
            "wikipedia_section_content with that page_title and section_index."
        ),
    }
    return json.dumps(payload, indent=2)


@tool
def wikipedia_section_content(page_title: str, section_index: str) -> str:
    """Fetch the text content of one Wikipedia section using the page title and section index returned by wikipedia_search."""
    clean_title = page_title.strip()
    clean_index = section_index.strip()
    if not clean_title:
        return "Wikipedia section error: page_title cannot be empty."
    if not clean_index:
        return "Wikipedia section error: section_index cannot be empty."

    try:
        section_text = _get_wikipedia_section_text(clean_title, clean_index)
    except Exception as exc:
        return f"Wikipedia section error: {exc}"

    if not section_text:
        return (
            f"Wikipedia section error: no content found for page '{clean_title}' "
            f"section '{clean_index}'."
        )

    payload = {
        "page_title": clean_title,
        "section_index": clean_index,
        "content": section_text[:4000],
    }
    return json.dumps(payload, indent=2)


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

    if BeautifulSoup is None:
        html = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<noscript\b[^<]*(?:(?!</noscript>)<[^<]*)*</noscript>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    else:
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


def run_with_tools(prompt: str) -> dict[str, Any]:
    tools = [
        web_search,
        wikipedia_search,
        wikipedia_section_content,
        calculator,
        python_executor,
        fetch_webpage,
        read_file_content,
        youtube_transcript,
    ]
    tools_by_name = {tool.name: tool for tool in tools}

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        base_url=OLLAMA_HOST,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    trace: dict[str, Any] = {
        "tools_used": [],
        "tool_calls": [],
        "searches": [],
    }
    seen_tools: set[str] = set()

    for _ in range(MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                "answer": _llm_to_text(response.content),
                "trace": trace,
            }

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            selected_tool = tools_by_name.get(tool_name)
            tool_args = tool_call.get("args", {})

            if tool_name not in seen_tools:
                trace["tools_used"].append(tool_name)
                seen_tools.add(tool_name)

            if selected_tool is None:
                tool_result = f"Tool error: unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = selected_tool.invoke(tool_args)
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"

            tool_event = {
                "tool_name": tool_name,
                "args": tool_args,
                "result": tool_result,
            }
            trace["tool_calls"].append(tool_event)

            parsed_result = _safe_json_loads(tool_result)
            if tool_name == "web_search" and parsed_result is not None:
                trace["searches"].append(
                    {
                        "tool_name": tool_name,
                        "query": parsed_result.get("query", ""),
                        "generated_search_query": parsed_result.get("generated_search_query", ""),
                        "tavily_results": parsed_result.get("tavily_results", []),
                    }
                )
            elif tool_name == "web_search":
                trace["searches"].append(
                    {
                        "tool_name": tool_name,
                        "query": tool_args.get("query", ""),
                        "generated_search_query": "",
                        "tavily_results": [],
                        "error": tool_result,
                    }
                )
            elif tool_name == "wikipedia_search" and parsed_result is not None:
                trace["searches"].append(
                    {
                        "tool_name": tool_name,
                        "query": parsed_result.get("query", ""),
                        "generated_search_query": parsed_result.get("generated_search_query", ""),
                        "page_title": parsed_result.get("page_title", ""),
                        "page_snippet": parsed_result.get("page_snippet", ""),
                    }
                )
            elif tool_name == "wikipedia_search":
                trace["searches"].append(
                    {
                        "tool_name": tool_name,
                        "query": tool_args.get("query", ""),
                        "generated_search_query": "",
                        "error": tool_result,
                    }
                )

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                )
            )

    messages.append(
        HumanMessage(
            content=(
                "Tool iteration limit reached. Provide your best final answer now "
                "without any additional tool calls."
            )
        )
    )
    final_response = llm.invoke(messages)
    return {
        "answer": _llm_to_text(final_response.content),
        "trace": trace,
    }


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
            run_result = run_with_tools(gaia_prompt)
            model_answer = str(run_result.get("answer", "")).strip()
            execution_trace = run_result.get("trace", {})
        except Exception as exc:
            status = "error"
            error = str(exc)
            execution_trace = {}

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result_item = {
            "index": index,
            "task_id": task_id,
            "level": level,
            "file_name": file_name,
            "task_file_error": task_file_error,
            "question": question,
            "model_answer": model_answer,
            "execution_trace": execution_trace,
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

    run_result = run_with_tools(prompt)
    response = str(run_result.get("answer", ""))
    trace = run_result.get("trace", {})
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt}")
    print(_format_trace_for_cli(trace))
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
