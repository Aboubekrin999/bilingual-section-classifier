"""Tests for dataset download helpers (no network)."""

import pytest

from src.download import (
    CSABSTRUCT_BASE,
    PUBMED_RCT_BASE,
    DownloadError,
    download_csabstruct,
    download_pubmed_rct,
)


class TestPubMedRCT:
    def test_writes_three_split_files(self, tmp_path):
        seen: list[str] = []

        def fake_fetch(url: str) -> bytes:
            seen.append(url)
            return f"BACKGROUND\tFrom {url}\n".encode()

        paths = download_pubmed_rct(tmp_path, fetch=fake_fetch)
        assert set(paths.keys()) == {"train", "dev", "test"}
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_uses_pinned_base_url(self, tmp_path):
        seen: list[str] = []

        def fake_fetch(url: str) -> bytes:
            seen.append(url)
            return b"BACKGROUND\tx\n"

        download_pubmed_rct(tmp_path, fetch=fake_fetch)
        for url in seen:
            assert url.startswith(PUBMED_RCT_BASE)

    def test_creates_output_dir(self, tmp_path):
        target = tmp_path / "nested" / "deep"
        download_pubmed_rct(target, fetch=lambda _: b"BACKGROUND\tx\n")
        assert target.is_dir()

    def test_empty_response_raises(self, tmp_path):
        with pytest.raises(DownloadError, match="empty response"):
            download_pubmed_rct(tmp_path, fetch=lambda _: b"")


class TestCSAbstruct:
    def test_writes_three_split_files(self, tmp_path):
        paths = download_csabstruct(
            tmp_path,
            fetch=lambda _: b'{"sentences":["x"],"labels":["result"]}\n',
        )
        assert set(paths.keys()) == {"train", "dev", "test"}

    def test_uses_pinned_base_url(self, tmp_path):
        seen: list[str] = []

        def fake_fetch(url: str) -> bytes:
            seen.append(url)
            return b'{"sentences":["x"],"labels":["result"]}\n'

        download_csabstruct(tmp_path, fetch=fake_fetch)
        for url in seen:
            assert url.startswith(CSABSTRUCT_BASE)

    def test_jsonl_filenames_used(self, tmp_path):
        paths = download_csabstruct(
            tmp_path,
            fetch=lambda _: b'{"sentences":["x"],"labels":["result"]}\n',
        )
        for path in paths.values():
            assert path.suffix == ".jsonl"


class TestPinning:
    def test_base_urls_pin_a_commit_sha_not_master(self):
        # Catch accidental drift to a moving ref.
        for base in (PUBMED_RCT_BASE, CSABSTRUCT_BASE):
            assert "/master/" not in base
            assert "/main/" not in base
