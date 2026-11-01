"""Offline test without importing the package or executing any network call."""
import ast
import importlib.util
from pathlib import Path
import os
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_sender():
    spec=importlib.util.spec_from_file_location('analytics_privacy_test',ROOT/'omicverse/utils/_analytics_sender.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('value',['1','true','TRUE','yes','on'])
def test_optout_all_senders_skip_collection_and_network(monkeypatch,value):
    monkeypatch.setenv('OMICVERSE_DISABLE_ANALYTICS',value)
    module=load_sender()
    def forbidden(*args,**kwargs): raise AssertionError('Analytics collection or network must not run')
    monkeypatch.setattr(module.requests,'get',forbidden)
    monkeypatch.setattr(module,'generate_user_hash',forbidden)
    monkeypatch.setattr(module,'get_geolocation',forbidden)
    for name in dir(module):
        if name.startswith('send_analytics'):
            assert getattr(module,name)('unit-test') is False
    assert module.test_analytics_connection() is False


def test_config_init_does_not_import_analytics_under_optout(monkeypatch):
    monkeypatch.setenv('OMICVERSE_DISABLE_ANALYTICS','1')
    # Compile only the actual __init__ source; no package import side effects.
    tree=ast.parse((ROOT/'omicverse/_settings.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='omicverseConfig')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    code=compile(ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[])),str(ROOT/'omicverse/_settings.py'),'exec')
    import builtins
    def forbidden_import(*args,**kwargs): raise AssertionError('Init must not import analytics under opt-out')
    namespace={'os':os,'__builtins__':dict(vars(builtins),__import__=forbidden_import)}
    exec(code,namespace)
    class Config: pass
    obj=Config(); namespace['__init__'](obj)
    assert obj.mode=='cpu' and obj.seed is None and obj.device is None
