"""Backend-free public namespace and actionable dependency errors."""
import importlib
from importlib.metadata import PackageNotFoundError
import pytest
import omicverse as ov
boundary=importlib.import_module('omicverse.pp._scran_pooling')


def test_missing_dependency(monkeypatch):
    def absent(_):raise PackageNotFoundError('scran-pooling-py')
    monkeypatch.setattr(boundary,'version',absent)
    with pytest.raises(ImportError,match='INSTALL_QUALIFIED.md'):ov.pp.scran_pooling(None)


def test_wrong_backend_version(monkeypatch):
    monkeypatch.setattr(boundary,'version',lambda _:'999')
    with pytest.raises(ImportError,match='found 999'):ov.pp.scran_pooling(None)


def test_public_export():
    assert callable(ov.pp.scran_pooling) and 'scran_pooling' in ov.pp.__all__
