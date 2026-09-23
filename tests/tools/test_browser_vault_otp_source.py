import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.browser_vault_otp_source import resolve_configured_otp, OTPSourceError

META = SimpleNamespace(kind='login', origin='https://brightdata.com', identifier='agent@example.test')
SOURCE={'enabled':True,'origin':META.origin,'identifier':META.identifier,'command':['/usr/bin/python3','/trusted/helper.py']}

def config(source=None):
    return {'vault':{'otp_sources':[dict(SOURCE if source is None else source)]}}

def test_matching_source_returns_secret_only_internally():
    def run(cmd, **kw):
        assert cmd == SOURCE['command'] and kw['capture_output'] and kw['timeout'] == 30
        assert not any(k in kw['env'] for k in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','OPENROUTER_API_KEY'))
        return SimpleNamespace(returncode=0,stdout=json.dumps({'code':'123456','issued_at':time.time()}))
    assert resolve_configured_otp(META,META.origin,config=config(),runner=run)=='123456'

@pytest.mark.parametrize('origin,source',[
    ('https://evil.test',SOURCE),
    (META.origin,{**SOURCE,'identifier':'other@example.test'}),
    (META.origin,{**SOURCE,'enabled':False}),
])
def test_mismatch_never_executes(origin,source):
    def fail(*a,**k): raise AssertionError('must not execute')
    assert resolve_configured_otp(META,origin,config=config(source),runner=fail) is None

@pytest.mark.parametrize('payload',[
    {'code':'123456','issued_at':time.time()-500},
    {'code':'123456','issued_at':time.time()+500},
    {'code':'abcdef','issued_at':time.time()},
    {'code':'１２３４５６','issued_at':time.time()},
    {'code':'123456','issued_at':True},
])
def test_invalid_result_is_sanitized(payload):
    with pytest.raises(OTPSourceError) as e:
        resolve_configured_otp(META,META.origin,config=config(),runner=lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(payload)))
    assert payload['code'] not in str(e.value)

def test_exceptions_do_not_leak_output():
    def fail(*a,**k): raise RuntimeError('sensitive-code-123456')
    with pytest.raises(OTPSourceError) as e:
        resolve_configured_otp(META,META.origin,config=config(),runner=fail)
    assert '123456' not in str(e.value)

def test_native_tool_headless_source_and_origin_binding(tmp_path,monkeypatch):
    from agent.vault_store import VaultStore
    from tools import browser_vault_tool as tool
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    store=VaultStore(tmp_path/'vault')
    meta=store.add_item('login','test',{'identifier_type':'email','identifier':META.identifier,'password':'fixture'},origin=META.origin)
    controls=[{'index':i,'type':'text','name':'code','label':'Verification code','autocomplete':'one-time-code','formIndex':0,'maxLength':1} for i in range(6)]
    code='314159'
    seen=[]
    def evaluate(task,expr):
        return {'success':True,'result':json.dumps(controls) if 'querySelectorAll' in expr else META.origin}
    def secret(task,expr):
        seen.append(expr)
        return {'success':True,'result':json.dumps({'filled':6})}
    with patch('agent.vault_store.get_vault_store',return_value=store), patch.object(tool,'_focus_bound_origin'), patch.object(tool,'_eval_js',side_effect=evaluate), patch.object(tool,'_eval_js_secret',side_effect=secret), patch('agent.vault_backends.unlock.can_prompt_here',return_value=False), patch('tools.browser_vault_otp_source.resolve_configured_otp',return_value=code) as source:
        out=tool.browser_vault_enter_code(meta.id,task_id='test')
        assert json.loads(out)['source']=='configured_command' and json.loads(out)['filled_fields']==6
        assert code not in out and len(seen)==1
        with patch.object(tool,'_current_page_origin',return_value='https://evil.test'):
            source.reset_mock()
            out=tool.browser_vault_enter_code(meta.id,task_id='test')
            assert json.loads(out)['error_type']=='origin_mismatch'
            source.assert_not_called()
