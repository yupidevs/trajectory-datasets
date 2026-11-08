import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from trajectory_datasets import downloads

URL = "https://example.test/data/sample.txt"


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, headers: dict | None = None) -> None:
        self.content, self.status_code = content, status
        self.headers = {"content-length": str(len(content))} if headers is None else headers

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


def _serve(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> list[dict]:
    calls: list[dict] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


@pytest.mark.parametrize(
    ("part", "total", "expected"),
    [(1048576, -1, "Downloaded: 1.00 MB"), (5, 10, "50.00%"), (10, 10, "100.00%\n")],
)
def test_progress_text(part: int, total: int, expected: str) -> None:
    assert expected in downloads._get_progress_log(part, total)


def test_download_writes_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _serve(monkeypatch, FakeResponse(b"hello"))
    path = downloads.download_dataset(URL, "demo", uncompress=False)
    assert path.read_bytes() == b"hello"
    assert path.name == "sample.txt"
    assert calls[0]["timeout"] > 0


def test_unknown_size_still_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"hello", headers={}))
    assert downloads.download_dataset(URL, "demo", uncompress=False).read_bytes() == b"hello"


def test_a_failed_request_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"", status=404))
    with pytest.raises(RuntimeError, match="demo"):
        downloads.download_dataset(URL, "demo", uncompress=False)


def test_a_complete_file_is_not_downloaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"hello"))
    path = downloads.download_dataset(URL, "demo", uncompress=False)
    path.write_bytes(b"HELLO")
    assert downloads.download_dataset(URL, "demo", uncompress=False).read_bytes() == b"HELLO"


def test_a_file_of_another_size_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"hello"))
    path = downloads.download_dataset(URL, "demo", uncompress=False)
    path.write_bytes(b"partial")
    assert downloads.download_dataset(URL, "demo", uncompress=False).read_bytes() == b"hello"


def test_an_existing_file_of_unknown_size_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"hello", headers={}))
    path = downloads.download_dataset(URL, "demo", uncompress=False)
    path.write_bytes(b"old")
    assert downloads.download_dataset(URL, "demo", uncompress=False).read_bytes() == b"hello"


def test_without_a_size_check_an_existing_file_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"hello"))
    path = downloads.download_dataset(URL, "demo", uncompress=False)
    path.write_bytes(b"old")
    kept = downloads.download_dataset(URL, "demo", uncompress=False, check_size=False)
    assert kept.read_bytes() == b"old"


def test_archives_are_extracted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("inner.txt", "content")
    _serve(monkeypatch, FakeResponse(archive.read_bytes()))
    path = downloads.download_dataset("https://example.test/pack.zip", "demo")
    assert (path.parent / "inner.txt").read_text() == "content"
