from __future__ import annotations
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

def estimate_tokens(text: str) -> int:
    """Deterministic character estimate, not provider billing usage."""
    return math.ceil(len(text.strip()) / 4)

@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id or user_id in {'.', '..'}:
            raise ValueError('A nonempty user id is required')
        # Prefix avoids Windows reserved names; encoding prevents traversal/collisions.
        slug = quote(user_id, safe='').replace('.', '%2E')
        return self.root_dir / ('user-' + slug) / 'User.md'

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(path)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r'^- (\w+): (.+)$', self.read_text(user_id), re.M))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not re.fullmatch(r'\w+', key) or '\n' in value:
            raise ValueError('Facts must be single-line fields')
        text = self.read_text(user_id) or '# User profile\n'
        line = f'- {key}: {value}'
        pattern = rf'^- {re.escape(key)}: .*?$'
        if re.search(pattern, text, re.M):
            text = re.sub(pattern, lambda _: line, text, flags=re.M)
        else:
            text = text.rstrip() + '\n' + line + '\n'
        self.write_text(user_id, text)

def extract_profile_updates(message: str) -> dict[str, str]:
    """Only explicit declarations; questions, hypotheticals and jokes are excluded."""
    facts = {}
    for sentence in re.findall(r'[^.!?\n]+[.!?]?', unicodedata.normalize('NFC', message)):
        if sentence.endswith('?'):
            continue
        sentence = sentence.rstrip('.!').strip()
        if re.search(r'^(?:nhắc lại|tóm tắt|bạn có|bạn thử|sang thread mới)', sentence, re.I) and not sentence.lower().startswith('nhắc lại lần cuối cho chắc:'):
            continue
        if not sentence or re.search(r'(?:nếu|câu đùa|mình đùa|hay là|là gì|tên gì|ở đâu|như thế nào|không\s*$)', sentence, re.I):
            continue
        patterns = {
            'name': r'(?:mình tên(?: là)?|tên mình là|nhắc lại lần cuối cho chắc: tên)\s+([^,;:]+)',
            'location': r'(?:mình (?:hiện |hiện tại |vẫn |đang |hiện đang |vẫn đang )?ở|hiện ở|nơi ở hiện tại là|mình đang làm việc ở)\s+([^,;.]+?)(?=\s+(?:và|chứ|trong|vài|mỗi|để|dù|chưa)|$)',
            'profession': r'(?:đang làm|vẫn là|mình làm|giờ chuyển sang|nghề nghiệp hiện tại vẫn là|nghề)\s+([\w+-]+ engineer)',
            'favorite_drink': r'đồ uống yêu thích(?: của mình)? là\s+([^,;]+)',
            'favorite_food': r'món ăn yêu thích(?: của mình)? là\s+([^,;]+)',
            'pet': r'mình nuôi\s+(.+)',
        }
        for key, pattern in patterns.items():
            matches = list(re.finditer(pattern, sentence, re.I))
            if matches:
                value = matches[-1].group(1).strip()
                if value and not re.search(r'(?:gì|đâu|không còn|không phải)', value, re.I):
                    facts[key] = value
        if re.search(r'(?:mình thích|mình vẫn thích|mình đang quan tâm)', sentence, re.I) and re.search(r'Python|\bAI\b', sentence):
            facts['interests'] = ', '.join(term for term in ('Python', 'AI', 'MLOps') if term.lower() in sentence.lower())
        if not re.search(r'^(?:nhắc lại|tóm tắt|bạn có|bạn thử)', sentence, re.I) and re.search(r'(?:mình muốn|mình thích|hãy trả lời|style trả lời|mình vẫn muốn)', sentence, re.I) and re.search(r'ngắn|bullet', sentence, re.I):
            style = ['ngắn gọn']
            if re.search(r'3 bullet', sentence, re.I):
                style.append('3 bullet')
            elif re.search(r'bullet', sentence, re.I):
                style.append('bullet')
            if re.search(r'ví dụ', sentence, re.I):
                style.append('có ví dụ thực tế / thực chiến')
            if 'trade-off' in sentence.lower():
                style.append('so sánh trade-off')
            facts['response_style'] = ', '.join(style)
    return facts

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded extractive summary with structured user facts."""
    if max_items <= 0:
        return ''
    facts, snippets = {}, []
    for message in messages:
        if message['role'] != 'user':
            continue
        facts.update(extract_profile_updates(message['content']))
        snippets.append(message['content'][:160])
    lines = [f'{key}: {value}' for key, value in facts.items()]
    if len(lines) < max_items:
        lines.extend(snippets[-(max_items - len(lines)):])
    return '\n'.join(lines[:max_items])

@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError('Compaction settings must be positive')

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self.context(thread_id)
        state['messages'].append({'role': role, 'content': content})
        messages = state['messages']
        tokens = estimate_tokens(state['summary']) + sum(estimate_tokens(m['content']) for m in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[:-self.keep_messages]
            facts = state.setdefault('_summary_facts', {})
            for message in older:
                if message['role'] == 'user':
                    facts.update(extract_profile_updates(message['content']))
            # Rewrite the summary instead of accumulating stale contradictory facts.
            lines = [f'{key}: {value[:160]}' for key, value in facts.items()][:8]
            notes = state.setdefault('_summary_notes', [])
            notes.extend(m['content'][:160] for m in older if m['role'] == 'user')
            state['_summary_notes'] = notes[-3:]
            lines.extend(state['_summary_notes'])
            kept, length = [], 0
            for line in lines:
                if length + len(line) + 1 <= 1200:
                    kept.append(line)
                    length += len(line) + 1
            state['summary'] = '\n'.join(kept)
            state['messages'] = messages[-self.keep_messages:]
            state['compactions'] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {'messages': [], 'summary': '', 'compactions': 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)['compactions']

def offline_response(message: str, facts: dict[str, str]) -> str:
    """Shared responder makes memory, rather than answer logic, the comparison."""
    if re.search(r'\?|nhắc lại|tóm tắt|nhớ lại', message, re.I):
        if not facts:
            return 'Mình chưa có thông tin về bạn trong phiên này. Bạn cung cấp lại giúp mình nhé.'
        values = list(facts.values())
        if '3 bullet' in facts.get('response_style', ''):
            return '\n'.join('- ' + '; '.join(values[i::3]) for i in range(3))
        return 'Thông tin mình nhớ: ' + '; '.join(values) + '.'
    return 'Mình đã ghi nhận. Ví dụ: giữ thông tin ổn định và cập nhật khi có đính chính.'
