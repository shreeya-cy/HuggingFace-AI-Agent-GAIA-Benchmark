# ChatBot GAIA Runner

## Behavior

- Running `agent.py` with no arguments starts GAIA batch mode.
- GAIA batch mode fetches 20 questions from the Hugging Face scoring API and stores them in `questions.json`.
- Each question is then answered one-by-one by the tool-enabled agent, and detailed records are written to `answers.json`.
- Running `agent.py` with a prompt argument keeps normal chat behavior.

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
