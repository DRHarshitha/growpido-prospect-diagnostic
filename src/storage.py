"""Local JSON persistence for typed diagnostic-run artifacts."""
import json
from pathlib import Path
from .models import DiagnosticRun

class LocalRunStore:
    """Stores runs locally; no external database or AWS service is used."""
    def __init__(self, runs_directory: str | Path) -> None:
        self.runs_directory = Path(runs_directory)
    def save(self, run: DiagnosticRun) -> Path:
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        path = self.runs_directory / f"{run.metadata.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path
    def load(self, run_id: str) -> DiagnosticRun:
        path = self.runs_directory / f"{run_id}.json"
        return DiagnosticRun.model_validate(json.loads(path.read_text(encoding="utf-8")))
