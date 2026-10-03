from dataclasses import replace
from pathlib import Path
import pytest
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import normalize_provider

def make_config(tmp_path: Path):
    return replace(load_config(tmp_path), compact_threshold_tokens=180, compact_keep_messages=4, live_mode=False)

def test_user_markdown_read_write_edit(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text('alice') == ''
    assert store.file_size('alice') == 0
    path = store.write_text('alice', '# User\nTên: An\n')
    assert path.name == 'User.md'
    assert store.edit_text('alice', 'An', 'Bình')
    assert 'Bình' in store.read_text('alice')
    assert not store.edit_text('alice', 'missing', 'value')
    assert store.file_size('alice') == len(store.read_text('alice').encode('utf-8'))
    store.upsert_fact('alice', 'location', 'Huế')
    store.upsert_fact('alice', 'location', 'Đà Nẵng')
    assert store.facts('alice')['location'] == 'Đà Nẵng'
    assert 'Huế' not in store.read_text('alice')
    assert 'Tên: Bình' in store.read_text('alice')

def test_profile_paths_are_isolated(tmp_path):
    store = UserProfileStore(tmp_path)
    for user in ('../../escape', 'a/b', 'a_b', 'CON', 'a\\b'):
        path = store.write_text(user, user)
        assert path.resolve().is_relative_to(tmp_path.resolve())
        assert store.read_text(user) == user
    assert store.path_for('a/b') != store.path_for('a_b')
    with pytest.raises(ValueError):
        store.path_for('..')

def test_compact_trigger(tmp_path):
    manager = CompactMemoryManager(100, 2)
    manager.append('one', 'user', 'Mình tên là An.')
    manager.append('one', 'assistant', 'Đã ghi nhận.')
    manager.append('one', 'user', 'Thông tin dài ' * 80)
    context = manager.context('one')
    assert manager.compaction_count('one') == 1
    assert len(context['messages']) == 2
    assert 'An' in context['summary']
    assert manager.compaction_count('other') == 0
    for i in range(20):
        manager.append('one', 'user', f'Ghi chú {i} ' * 80)
    assert len(manager.context('one')['summary']) <= 1200

def test_cross_session_recall(tmp_path):
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply('alice', 'first', 'Mình tên là An. Mình ở Huế.')
        assert 'An' in agent.reply('alice', 'first', 'Mình tên gì?')['response']
    assert 'An' not in baseline.reply('alice', 'second', 'Mình tên gì?')['response']
    restarted = AdvancedAgent(config, force_offline=True)
    assert 'An' in restarted.reply('alice', 'second', 'Mình tên gì?')['response']
    assert 'An' not in restarted.reply('bob', 'bob-thread', 'Mình tên gì?')['response']
    with pytest.raises(ValueError):
        restarted.reply('bob', 'second', 'Mình tên gì?')

def test_compact_reduces_prompt_load_on_long_thread(tmp_path):
    config = make_config(tmp_path)
    baseline, advanced = BaselineAgent(config, True), AdvancedAgent(config, True)
    for agent in (baseline, advanced):
        agent.reply('alice', 'long', 'Mình tên là An.')
        for _ in range(16):
            agent.reply('alice', 'long', 'Nội dung tạm thời để kiểm tra context. ' * 100)
    assert advanced.compaction_count('long') > 1
    assert advanced.prompt_token_usage('long') < baseline.prompt_token_usage('long') * 0.6
    assert 'An' in advanced.reply('alice', 'fresh', 'Mình tên gì?')['response']

@pytest.mark.parametrize('message', [
    'Mình tên là DũngCT không?',
    'Mình đang ở Hà Nội hay Huế?',
    'Nếu mình ở Hà Nội thì sao.',
    'Mình đùa rằng đang làm product manager.',
    'Nhắc lại style trả lời mình thích, ngắn gọn nhé?',
])
def test_questions_and_hypotheticals_do_not_write_facts(message):
    assert extract_profile_updates(message) == {}

def test_corrections_and_noise(tmp_path):
    agent = AdvancedAgent(make_config(tmp_path), True)
    for message in [
        'Mình ở Huế và đang làm backend engineer.',
        'Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.',
        'Mình đang làm việc ở Đà Nẵng vài tháng để tiện gặp team.',
        'Hà Nội chỉ là nơi mình đi họp, không phải nơi ở hiện tại.',
        'Mình đùa rằng đang làm product manager.',
        'Mình muốn bạn trả lời ngắn gọn theo 3 bullet có ví dụ thực chiến.',
    ]:
        agent.reply('alice', 'one', message)
    before = agent.profile_store.read_text('alice')
    agent.reply('alice', 'new', 'Nhắc lại ngắn về mình và style trả lời mình thích?')
    assert agent.profile_store.read_text('alice') == before
    facts = agent.profile_store.facts('alice')
    assert facts['location'] == 'Đà Nẵng'
    assert facts['profession'] == 'MLOps engineer'
    assert '3 bullet' in facts['response_style']
    assert 'backend engineer' not in before

def test_accounting_and_scoring(tmp_path):
    assert estimate_tokens('') == 0
    assert estimate_tokens('     ') == 0
    assert estimate_tokens('abcde') == 2
    assert recall_points('PYTHON, AI', ['Python', 'AI', 'MLOps', 'Huế']) == 0.5
    for agent in (BaselineAgent(make_config(tmp_path), True), AdvancedAgent(make_config(tmp_path), True)):
        results = [agent.reply('a', 't', message) for message in ('Mình tên là An.', 'Mình tên gì?')]
        assert agent.token_usage('t') == sum(r['token_usage'] for r in results)
        assert agent.prompt_token_usage('t') == sum(r['prompt_tokens'] for r in results)
        assert agent.token_usage('missing') == 0

def test_supplied_benchmarks(tmp_path):
    config = make_config(tmp_path)
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    for filename in ('conversations.json', 'advanced_long_context.json'):
        data = load_conversations(data_dir / filename)
        isolated = replace(config, state_dir=tmp_path / filename)
        baseline = run_agent_benchmark('Baseline', BaselineAgent(isolated, True), data, isolated)
        advanced = run_agent_benchmark('Advanced', AdvancedAgent(isolated, True), data, isolated)
        assert baseline.recall_score == 0
        assert advanced.recall_score == 1
        assert advanced.memory_growth_bytes > 0
        if filename == 'advanced_long_context.json':
            assert advanced.compactions > 1
            assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed

@pytest.mark.parametrize('provider', ['openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter'])
def test_supported_providers(provider):
    assert normalize_provider(provider.upper()) == provider

def test_invalid_provider():
    assert normalize_provider('anthorpic') == 'anthropic'
    with pytest.raises(ValueError):
        normalize_provider('unknown')

@pytest.mark.parametrize('provider', ['openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter'])
def test_provider_factory_arguments(monkeypatch, provider):
    import model_provider
    from types import SimpleNamespace
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return kwargs
    monkeypatch.setattr(model_provider, 'import_module', lambda name: SimpleNamespace(ChatOpenAI=factory, ChatGoogleGenerativeAI=factory, ChatAnthropic=factory, ChatOllama=factory))
    config = model_provider.ProviderConfig(provider, 'test-model', 0.1, None if provider == 'ollama' else 'test-key', 'http://localhost:1234' if provider in {'custom', 'ollama'} else None)
    model_provider.build_chat_model(config)
    assert calls[0]['model'] == 'test-model'
    assert calls[0]['temperature'] == 0.1
    if provider == 'gemini':
        assert calls[0]['google_api_key'] == 'test-key'
    elif provider != 'ollama':
        assert calls[0]['api_key'] == 'test-key'
    if provider == 'openrouter':
        assert calls[0]['base_url'] == 'https://openrouter.ai/api/v1'


def test_live_paths_use_the_same_memory_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import agent_advanced
    import agent_baseline
    prompts = []
    class FakeModel:
        def invoke(self, messages):
            prompts.append(messages)
            return SimpleNamespace(content='Test response')
    monkeypatch.setattr(agent_advanced, 'build_chat_model', lambda config: FakeModel())
    monkeypatch.setattr(agent_baseline, 'build_chat_model', lambda config: FakeModel())
    config = replace(make_config(tmp_path), live_mode=True)
    baseline, advanced = BaselineAgent(config), AdvancedAgent(config)
    for agent in (baseline, advanced):
        agent.reply('alice', 'first', 'Mình tên là An.')
        agent.reply('alice', 'second', 'Mình tên gì?')
    assert not any(' An.' in m['content'] for m in prompts[1])
    assert any('name: An' in m['content'] for m in prompts[3])
    assert AdvancedAgent(config, force_offline=True).langchain_agent is None
