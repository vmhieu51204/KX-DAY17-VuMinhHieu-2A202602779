from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from model_provider import ProviderConfig, normalize_provider

@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live_mode: bool = False

    def __post_init__(self):
        if self.compact_threshold_tokens < 1 or self.compact_keep_messages < 1:
            raise ValueError('Compaction settings must be positive')

def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / '.env', override=False)
    def provider_config(prefix, fallback=None):
        provider = normalize_provider(os.getenv(f'{prefix}_PROVIDER', fallback.provider if fallback else 'openai'))
        defaults = {'openai': 'gpt-4o-mini', 'custom': 'local-model', 'gemini': 'gemini-2.0-flash', 'anthropic': 'claude-sonnet-4-20250514', 'ollama': 'llama3.2', 'openrouter': 'openai/gpt-4o-mini'}
        key = os.getenv(f'{prefix}_API_KEY') or os.getenv(f'{provider.upper()}_API_KEY')
        if provider == 'gemini':
            key = key or os.getenv('GOOGLE_API_KEY')
        return ProviderConfig(provider, os.getenv(f'{prefix}_MODEL', fallback.model_name if fallback and fallback.provider == provider else defaults[provider]), float(os.getenv(f'{prefix}_TEMPERATURE', '0')), key, os.getenv(f'{prefix}_BASE_URL') or os.getenv(f'{provider.upper()}_BASE_URL'))
    state = root / 'state'
    state.mkdir(parents=True, exist_ok=True)
    model = provider_config('LLM')
    return LabConfig(root, root / 'data', state, int(os.getenv('COMPACT_THRESHOLD_TOKENS', '1200')), int(os.getenv('COMPACT_KEEP_MESSAGES', '4')), model, provider_config('JUDGE', model), os.getenv('LAB_LIVE_MODE', '0').lower() in {'1', 'true', 'yes'})
