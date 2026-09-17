"""Dependency-free progress output that also remains readable in saved logs."""


def show_progress(label: str, completed: int, total: int, details: str = "") -> None:
    ratio = completed / total if total else 1.0
    filled = int(24 * ratio)
    bar = "#" * filled + "-" * (24 - filled)
    print(f"{label}: [{bar}] {ratio:6.1%} {completed}/{total} {details}", flush=True)
