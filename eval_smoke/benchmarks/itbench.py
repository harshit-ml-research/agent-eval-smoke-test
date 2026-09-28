from .base import Task


class ITBenchAdapter:
    """Integration point for future ITBench task loading and scoring."""

    def tasks(self) -> list[Task]:
        raise NotImplementedError("ITBench integration is outside the smoke test")

    def score(self, task: Task, result: dict) -> dict:
        raise NotImplementedError("ITBench integration is outside the smoke test")
