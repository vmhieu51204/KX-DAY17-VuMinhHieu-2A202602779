# Completed lab

The implementation runs offline by default and needs only Python 3.11+.
Install pytest to run the tests:

```powershell
python -m pip install -r requirements.txt
python src/benchmark.py
python -m pytest src/test_agents.py -v --basetemp=state/test-tmp
```

The explicit test directory keeps temporary files inside the repository, including
in environments that restrict access to the system temporary directory.

## Memory design

Baseline keeps the complete user/assistant history per thread. Its facts are
reconstructed exclusively from user messages in that thread. A fresh thread has
no earlier facts. Advanced writes stable, explicitly declared fields to
`state/profiles/user-<encoded-user-id>/User.md`. Field updates replace previous
values, so the latest location or profession correction wins. Profiles survive
agent restarts; thread histories and compact summaries are held in memory.

Advanced compacts older messages once the estimated context exceeds the
threshold, retaining the configured number of recent messages verbatim. Summaries
carry structured facts forward and bounded excerpts from recent older user turns.
They remain at most 1,200 characters. A single large recent message can exceed the
threshold: the threshold is a trigger, not a hard context-size limit.

Questions, recall requests, hypothetical clauses, and jokes do not become profile
facts. This conservative structured extraction is the bonus: it reduces false
writes and resolves explicit corrections. It is a Vietnamese heuristic, not a
general entity extraction model; unfamiliar phrasing may be missed. It should be
extended with validation or confidence-calibrated extraction for real deployment.

## Configuration and optional live mode

`load_config()` optionally reads root `.env` when python-dotenv is installed.
Environment variables take priority. Knobs:

- `COMPACT_THRESHOLD_TOKENS=1200`, `COMPACT_KEEP_MESSAGES=4`.
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_API_KEY`, `LLM_BASE_URL`.
- `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE`, `JUDGE_API_KEY`, `JUDGE_BASE_URL`.
- Provider-specific keys: `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`),
  `ANTHROPIC_API_KEY`, `CUSTOM_API_KEY`, `OPENROUTER_API_KEY`.
- Provider-specific endpoints: `CUSTOM_BASE_URL`, `OLLAMA_BASE_URL`, and the same
  `<PROVIDER>_BASE_URL` pattern for other providers.

Optional live dependencies are listed in `requirements-live.txt`. Set
`LAB_LIVE_MODE=1` and instantiate either agent without `force_offline=True`.
Live operation uses LangChain chat models with the same explicit history and
memory pipeline. OpenRouter uses its OpenAI-compatible endpoint. No LangGraph
middleware or model-controlled filesystem tools are required. Configuration for
a judge is available, but the offline benchmark uses only a documented quality
heuristic. Live API calls have not been tested with credentials.

```python
# Run with src on the Python import path (for example from inside src/).
from agent_advanced import AdvancedAgent
agent = AdvancedAgent()
print(agent.reply('demo', 'session-1', 'Mình tên là An.')['response'])
print(agent.reply('demo', 'session-2', 'Mình tên gì?')['response'])
```

`reply()` returns `response`, estimated output `token_usage`, and estimated
`prompt_tokens` for that turn. Getter methods return cumulative thread totals.
Both modes estimate tokens from characters; they do not report provider billing
usage. Benchmark output tokens include replies to recall questions. Prompts
include the common system instructions, retained history, and (for advanced)
profile and summary. The quality proxy is 80% expected-string recall plus 20%
nonempty/concise response scoring, on a 0–1 scale. It is not independent evidence
of natural-language response quality.

Each benchmark suite uses temporary clean profiles under `state/`, then removes
only its own temporary directory. Repeated runs do not inherit earlier facts or
erase existing user profiles. Recall is checked immediately after each
conversation in a fresh thread for each question, before later corrections.

See [RESULTS.md](RESULTS.md) for measured results and reflection.
