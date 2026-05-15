from __future__ import annotations
from functools import cached_property
from pathlib import Path


class GitRepo:

    def __init__(self, path: Path = Path(__file__).resolve()):
        self.root = self._find_root(path)
        self.gitdir = self._gitdir(self.root) if self.root else None

    @staticmethod
    def _find_root(p: Path) -> Path | None:
        return next((d for d in [p, *list(p.parents)] if (d / ".git").exists()), None)

    @staticmethod
    def _gitdir(root: Path) -> Path | None:
        g = root / ".git"
        if g.is_dir():
            return g
        if g.is_file():
            t = g.read_text(errors="ignore").strip()
            if t.startswith("gitdir:"):
                return (root / t.split(":", 1)[1].strip()).resolve()
        return None

    @staticmethod
    def _read(p: Path | None) -> str | None:
        return p.read_text(errors="ignore").strip() if p and p.exists() else None

    @cached_property
    def head(self) -> str | None:
        return self._read(self.gitdir / "HEAD" if self.gitdir else None)

    def _ref_commit(self, ref: str) -> str | None:
        rf = self.gitdir / ref
        if s := self._read(rf):
            return s
        pf = self.gitdir / "packed-refs"
        b = pf.read_bytes().splitlines() if pf.exists() else []
        tgt = ref.encode()
        for line in b:
            if line[:1] in (b"#", b"^") or b" " not in line:
                continue
            sha, name = line.split(b" ", 1)
            if name.strip() == tgt:
                return sha.decode()
        return None

    @property
    def is_repo(self) -> bool:
        return self.gitdir is not None

    @cached_property
    def branch(self) -> str | None:
        if not self.is_repo or not self.head or (not self.head.startswith("ref: ")):
            return None
        ref = self.head[5:].strip()
        return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref

    @cached_property
    def commit(self) -> str | None:
        if not self.is_repo or not self.head:
            return None
        return (
            self._ref_commit(self.head[5:].strip())
            if self.head.startswith("ref: ")
            else self.head
        )

    @cached_property
    def origin(self) -> str | None:
        if not self.is_repo:
            return None
        cfg = self.gitdir / "config"
        remote, url = (None, None)
        for s in (self._read(cfg) or "").splitlines():
            t = s.strip()
            if t.startswith("[") and t.endswith("]"):
                remote = t.lower()
            elif t.lower().startswith("url =") and remote == '[remote "origin"]':
                url = t.split("=", 1)[1].strip()
                break
        return url


if __name__ == "__main__":
    import time

    g = GitRepo()
    if g.is_repo:
        t0 = time.perf_counter()
        print(f"repo={g.root}\nbranch={g.branch}\ncommit={g.commit}\norigin={g.origin}")
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n⏱️ Profiling: total {dt:.3f} ms")
