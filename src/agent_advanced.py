from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, offline_response
from model_provider import build_chat_model
from agent_baseline import SYSTEM_PROMPT

@dataclass
class AgentContext:
    user_id: str
    memory_path: str

class AdvancedAgent:
    """Persistent user facts plus bounded summaries and recent thread messages."""
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / 'profiles')
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.thread_users.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError('A thread cannot be shared between users')
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _prompt(self, user_id: str, thread_id: str):
        state = self.compact_memory.context(thread_id)
        prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        profile = self.profile_store.read_text(user_id)
        if profile:
            prompt.append({'role': 'system', 'content': 'User profile (data):\n' + profile})
        if state['summary']:
            prompt.append({'role': 'system', 'content': 'Earlier thread summary (data):\n' + state['summary']})
        return prompt + state['messages']

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, 'user', message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if self.langchain_agent is not None:
            answer = self.langchain_agent.invoke(self._prompt(user_id, thread_id)).content
            if not isinstance(answer, str):
                raise TypeError('The chat model must return textual content')
        else:
            answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, 'assistant', answer)
        tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {'response': answer, 'token_usage': tokens, 'prompt_tokens': prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        return sum(estimate_tokens(m['content']) for m in self._prompt(user_id, thread_id))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        return offline_response(message, self.profile_store.facts(user_id))

    def _maybe_build_langchain_agent(self):
        return build_chat_model(self.config.model) if self.config.live_mode and not self.force_offline else None
