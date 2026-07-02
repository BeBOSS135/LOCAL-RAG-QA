# Index Write-Lock Tests
import pytest

import config
import lock


@pytest.fixture
def tmp_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(lock, "LOCK_FILE", tmp_path / ".index.lock")


def test_acquire_release_lifecycle(tmp_lock):
    assert lock.held() is False
    lock.acquire()
    assert lock.held() is True
    assert "pid=" in lock.describe()
    lock.release()
    assert lock.held() is False


def test_second_acquire_raises(tmp_lock):
    lock.acquire()
    with pytest.raises(lock.IndexLocked, match="delete"):
        lock.acquire()
    lock.release()


def test_release_without_lock_is_noop(tmp_lock):
    lock.release()  # Must Not Raise - Crash Recovery May Double-Release
    assert lock.held() is False
