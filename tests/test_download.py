"""
Tests for the dataset download layer.

No network: ``download_*`` take an injectable fetcher. The one thing that
cannot be asserted offline — that the pinned shas point at commits which
exist — is what ``scripts/verify_sources.py`` is for, and what
``TestPinnedSources`` guards the *shape* of.
"""

from __future__ import annotations

import re
import urllib.error
from pathlib import Path

import pytest

from src.download import (
    CSABSTRUCT_BASE,
    PUBMED_RCT_BASE,
    DownloadError,
    download_csabstruct,
    download_pubmed_rct,
    http_fetch,
)

_SHA_LENGTH = 40


class TestPinnedSources:
    """Guards against the pins regressing to a branch name or a short sha.

    Both URLs previously carried shas that did not exist in either
    repository, which only surfaced as a 404 at download time.
    """

    @pytest.mark.parametrize("base", [PUBMED_RCT_BASE, CSABSTRUCT_BASE])
    def test_base_is_a_raw_github_url(self, base: str) -> None:
        assert base.startswith("https://raw.githubusercontent.com/")

    @pytest.mark.parametrize("base", [PUBMED_RCT_BASE, CSABSTRUCT_BASE])
    def test_pinned_to_a_full_length_sha(self, base: str) -> None:
        """A branch name here would let the corpus drift between runs.

        raw.githubusercontent.com URLs are ``/<owner>/<repo>/<ref>/<path>``,
        so the ref is the segment after the repository name.
        """
        segments = base.split("/")
        ref = segments[5]
        assert re.fullmatch(rf"[0-9a-f]{{{_SHA_LENGTH}}}", ref), (
            f"{ref!r} is not a full commit sha — a branch name here would let "
            f"the corpus drift between runs"
        )

    def test_csabstruct_path_is_not_doubly_nested(self) -> None:
        """The data lives at data/CSAbstruct, not under a repeated repo name."""
        assert CSABSTRUCT_BASE.endswith("/data/CSAbstruct")
        path_after_ref = "/".join(CSABSTRUCT_BASE.split("/")[6:])
        assert path_after_ref == "data/CSAbstruct"

    def test_pubmed_base_targets_the_20k_variant(self) -> None:
        assert PUBMED_RCT_BASE.endswith("/PubMed_20k_RCT")


class TestDownloadSplits:
    def test_writes_every_pubmed_split(self, tmp_path: Path) -> None:
        paths = download_pubmed_rct(tmp_path, fetch=lambda url: b"LABEL\tsentence\n")
        assert set(paths) == {"train", "dev", "test"}
        for path in paths.values():
            assert path.read_bytes() == b"LABEL\tsentence\n"

    def test_writes_every_csabstruct_split(self, tmp_path: Path) -> None:
        paths = download_csabstruct(tmp_path, fetch=lambda url: b'{"a":1}\n')
        assert set(paths) == {"train", "dev", "test"}
        assert all(p.suffix == ".jsonl" for p in paths.values())

    def test_requests_the_pinned_urls(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def fetch(url: str) -> bytes:
            seen.append(url)
            return b"x"

        download_pubmed_rct(tmp_path, fetch=fetch)
        assert seen == [f"{PUBMED_RCT_BASE}/{n}" for n in ("train.txt", "dev.txt", "test.txt")]

    def test_creates_the_output_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper"
        download_pubmed_rct(target, fetch=lambda url: b"x")
        assert target.is_dir()

    def test_empty_body_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DownloadError, match="empty response"):
            download_pubmed_rct(tmp_path, fetch=lambda url: b"")


class TestHttpFetchErrors:
    """A 404 must name the likely cause, not surface a bare urllib error."""

    def _raising(self, exc: Exception):
        def _fake(url, timeout=None):
            raise exc

        return _fake

    def test_404_explains_the_pin_may_have_moved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = urllib.error.HTTPError("http://x/y", 404, "Not Found", {}, None)
        monkeypatch.setattr("urllib.request.urlopen", self._raising(error))
        with pytest.raises(DownloadError, match="pinned commit sha"):
            http_fetch("http://x/y")

    def test_other_http_errors_carry_the_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = urllib.error.HTTPError("http://x/y", 503, "Unavailable", {}, None)
        monkeypatch.setattr("urllib.request.urlopen", self._raising(error))
        with pytest.raises(DownloadError, match="HTTP 503"):
            http_fetch("http://x/y")

    def test_unreachable_host_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        error = urllib.error.URLError("no route to host")
        monkeypatch.setattr("urllib.request.urlopen", self._raising(error))
        with pytest.raises(DownloadError, match="could not reach"):
            http_fetch("http://x/y")
