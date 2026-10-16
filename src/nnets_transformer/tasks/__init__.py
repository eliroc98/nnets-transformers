"""Task registry; task-specific scoring labels never live in the runner."""

def get_task(name):
    if name == "induction":
        from .induction import Induction
        return Induction()
    if name in {"ioi", "gendered_pronoun", "greater_than", "sva"}:
        from .paper import PaperTask
        return PaperTask(name)
    raise ValueError(f"Unimplemented fresh task: {name}")
