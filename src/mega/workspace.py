"""Per-run, in-memory file store with optimistic concurrency. --major_change

Replaces the git-backed ./files folder. Each sample gets its own Workspace,
so two samples (or two agents in different samples) can never touch the
same filename and no OS file handles or git subprocesses are involved.
A version string plays the role the commit hash used to play:
overwriting an existing file requires the version you last read,
so concurrent edits are detected instead of lost.
"""

import itertools
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Workspace:
    def __init__(self):
        self._files: Dict[str, list] = {}  # name -> [content, version, author]
        self._lock = threading.Lock()
        self._versions = itertools.count(1)

    @staticmethod
    def norm(name: str) -> str:
        name = str(name).strip().replace("\\", "/")
        while name.startswith("./"):
            name = name[2:]
        if name.startswith("files/"):
            name = name[len("files/") :]
        return name

    def read(self, name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (content, version, author), or (None, None, None) if missing."""
        with self._lock:
            entry = self._files.get(self.norm(name))
            return tuple(entry) if entry else (None, None, None)

    def write(
        self,
        name: str,
        content: str,
        base_version: Optional[str] = None,
        author: str = "",
    ) -> Tuple[bool, str, str]:
        """Create, or overwrite if base_version matches the current version.

        Returns (ok, current_content, current_version).
        """
        name = self.norm(name)
        with self._lock:
            cur = self._files.get(name)
            if cur is not None and base_version != cur[1]:
                return False, cur[0], cur[1]
            version = f"v{next(self._versions)}"
            self._files[name] = [content, version, author]
            return True, content, version

    def put(self, name: str, content: str, author: str = "") -> str:
        """Unconditional write (system files such as review.txt or todos)."""
        with self._lock:
            version = f"v{next(self._versions)}"
            self._files[self.norm(name)] = [content, version, author]
            return version

    def list(self, author: Optional[str] = None) -> List[str]:
        with self._lock:
            return sorted(
                n
                for n, (_, _, a) in self._files.items()
                if author is None or a == author
            )

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {
                n: {"content": c, "version": v, "author": a}
                for n, (c, v, a) in self._files.items()
            }

    def dump(self, directory: Path) -> None:
        """Write every file to `directory` once, after the run has finished."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        snap = self.snapshot()
        for name, entry in snap.items():
            target = directory / name.replace("/", "__")
            target.write_text(entry["content"], encoding="utf-8")
        (directory / "_files_index.json").write_text(
            json.dumps(
                {
                    n: {"version": e["version"], "author": e["author"]}
                    for n, e in snap.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
