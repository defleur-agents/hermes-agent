def test_split_code_fills_wait_between_controlled_input_events():
    from agent.vault_login_classifier import build_fill_js
    js = build_fill_js([{'index':0,'token':'one-time-code','value':'1'},
                        {'index':1,'token':'one-time-code','value':'2'}],
                       'https://brightdata.com', 'test', settle=True)
    assert js.startswith('(async () => {')
    assert 'await new Promise' in js
    assert js.count('window.location.origin !== expectedOrigin') == 2


def test_password_fill_remains_synchronous():
    from agent.vault_login_classifier import build_fill_js
    js=build_fill_js([{'index':0,'value':'synthetic-password'}],'https://example.com','test')
    assert js.startswith('(() => {')
    assert 'await new Promise' not in js
