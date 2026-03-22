# ChatBot GAIA Runner

## Behavior

- Running `agent.py` with no arguments starts GAIA batch mode.
- GAIA batch mode fetches 20 questions from the Hugging Face scoring API and stores them in `questions.json`.
- Each question is then answered one-by-one by the tool-enabled agent, and detailed records are written to `answers.json`.
- Running `agent.py` with a prompt argument keeps normal chat behavior.
- Web search now uses Tavily with an internal 2-step LLM flow: query optimization before search, then answer synthesis from structured JSON evidence kept in memory.
- Normal prompt runs now print the tools used, any web/Wikipedia search queries, and the normalized Tavily results shown to the agent.
- GAIA batch output now stores an `execution_trace` per answer in `answers.json`, including tool calls and search metadata.

## Environment

Set these environment variables in your shell or `.env` file:

```bash
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
TAVILY_API_KEY=your_tavily_api_key
```

## Commands

Run GAIA batch mode (default 20 questions):

```bash
python agent.py
```

Run GAIA batch mode with a custom count:

```bash
python agent.py --gaia-limit 20
```

Run quick GAIA test mode (default 5 questions):

```bash
python agent.py --quick-test
```

Run quick GAIA test mode with an explicit count:

```bash
python agent.py --quick-test 3
```

Run normal prompt mode:

```bash
python agent.py "Who is Ada Lovelace?"
```

## Output Files

- `questions.json`: fetched GAIA questions with source metadata.
- `answers.json`: per-question model answers, timing, status, and errors.
