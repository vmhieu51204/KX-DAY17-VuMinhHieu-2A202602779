from __future__ import annotations
import json
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int

def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Dataset must be a list of conversations')
    return data

def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected strings found; supports any number of facts."""
    normalized = unicodedata.normalize('NFC', answer).casefold()
    return sum(unicodedata.normalize('NFC', fact).casefold() in normalized for fact in expected) / len(expected) if expected else 1.0

def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Transparent offline proxy, not an LLM judge or human quality assessment."""
    if not answer.strip():
        return 0.0
    return 0.8 * recall_points(answer, expected) + 0.2 * min(1.0, 600 / len(answer))

def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {c['user_id'] for c in conversations}
    initial_bytes = sum(agent.memory_file_size(u) for u in users)
    initial_tokens = sum(agent.token_usage(t) for t in agent.thread_users)
    initial_prompts = sum(agent.prompt_token_usage(t) for t in agent.thread_users)
    initial_compactions = sum(agent.compaction_count(t) for t in agent.thread_users)
    recalls, qualities = [], []
    run_id = len(agent.thread_users)
    for index, conversation in enumerate(conversations):
        user = conversation['user_id']
        thread = f'benchmark-{run_id}-{index}-{conversation["id"]}'
        for turn in conversation['turns']:
            agent.reply(user, thread, turn)
        # Evaluate immediately, before future conversations correct this profile.
        for qindex, question in enumerate(conversation.get('recall_questions', [])):
            recall_thread = f'{thread}-recall-{qindex}'
            answer = agent.reply(user, recall_thread, question['question'])['response']
            recalls.append(recall_points(answer, question['expected_contains']))
            qualities.append(heuristic_quality(answer, question['expected_contains']))
    threads = agent.thread_users
    return BenchmarkRow(agent_name, sum(agent.token_usage(t) for t in threads) - initial_tokens, sum(agent.prompt_token_usage(t) for t in threads) - initial_prompts, sum(recalls) / len(recalls) if recalls else 0.0, sum(qualities) / len(qualities) if qualities else 0.0, sum(agent.memory_file_size(u) for u in users) - initial_bytes, sum(agent.compaction_count(t) for t in threads) - initial_compactions)

def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ['Agent', 'Agent tokens only', 'Prompt tokens processed', 'Cross-session recall', 'Response quality', 'Memory growth (bytes)', 'Compactions']
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append(f'| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | {row.recall_score:.1%} | {row.response_quality:.3f} | {row.memory_growth_bytes} | {row.compactions} |')
    return '\n'.join(lines)

def main() -> None:
    config = load_config()
    print('Offline benchmark: tokens are estimates; quality is a heuristic proxy. Recall turns are included in token totals. Each suite starts with clean memory.\n')
    for title, filename in [('Standard Benchmark', 'conversations.json'), ('Long-Context Stress Benchmark', 'advanced_long_context.json')]:
        conversations = load_conversations(config.data_dir / filename)
        # Do not erase real profiles or reuse facts from earlier benchmark runs.
        with TemporaryDirectory(prefix='benchmark-', dir=config.state_dir) as directory:
            isolated = replace(config, state_dir=Path(directory))
            rows = [run_agent_benchmark('Baseline', BaselineAgent(isolated, force_offline=True), conversations, isolated), run_agent_benchmark('Advanced', AdvancedAgent(isolated, force_offline=True), conversations, isolated)]
            print(title)
            print(format_rows(rows))
            print()

if __name__ == '__main__':
    main()
