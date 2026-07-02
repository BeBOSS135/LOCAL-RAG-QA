# Index Write Lock - One Indexer at a Time, Readers Get a Heads-Up
# A Git-Style Lock FILE Beats OS File Locking Here: Cross-Platform, Inspectable,
# and a Crashed Run Leaves an Explainable Artifact (pid + Start Time) Instead of a
# Silently Half-Written Index. Lives Inside chroma_db/ (Next to What It Protects,
# Already Gitignored)
import os
import time

import config

LOCK_FILE = config.CHROMA_DIR / ".index.lock"


class IndexLocked(RuntimeError):
    pass


# Take the Exclusive Indexing Lock. O_CREAT|O_EXCL Is Atomic at the OS Level, So
# There's No Check-Then-Create Race Between Two Indexers Starting Together
def acquire() -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise IndexLocked(
            f"Another indexing run appears to be in progress ({describe()}).\n"
            f"If that run crashed, delete {LOCK_FILE} and retry."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}")


def release() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


# Is an Indexing Run in Progress? Cheap Check for Readers (UI/CLI) to Warn That
# Results May Be Incomplete Until It Finishes
def held() -> bool:
    return LOCK_FILE.exists()


def describe() -> str:
    try:
        # utf-8-sig: a Hand-Made/Edited Lock File May Carry a BOM, Which the
        # Windows cp1252 Console Can't Print (Would Crash the Refusal Message)
        return LOCK_FILE.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return "unreadable lock"
