"""Scientific records shared by adapters, tasks, execution and artifact readers."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Selection:
    hook: str
    head: int | None = None

    @property
    def id(self):
        return self.hook if self.head is None else f"{self.hook}.{self.head}"


@dataclass(frozen=True)
class Component:
    hook: str
    layer: int
    head_count: int | None = None
    head_axis: int | None = None
    feature_axis: int = -1
    sequence_axis: int = 1


@dataclass(frozen=True)
class Scoring:
    metric: str
    position: int
    targets: tuple[int, ...]
    contrasts: tuple[int, ...] = ()


@dataclass(frozen=True)
class Example:
    id: str
    tokens: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    scoring: Scoring
    roles: dict
    split: str = "evaluation"
    corrupted_tokens: tuple[int, ...] | None = None
    corrupted_mask: tuple[bool, ...] | None = None

    def corrupted(self):
        if self.corrupted_tokens is None:
            raise ValueError("Task has no declared corrupted pair")
        return replace(self, tokens=self.corrupted_tokens, attention_mask=self.corrupted_mask,
                       corrupted_tokens=None, corrupted_mask=None)

    def validate(self, vocab_size, context_length):
        if not self.tokens or len(self.tokens) > context_length:
            raise ValueError("Empty example or context length exceeded")
        if len(self.tokens) != len(self.attention_mask) or not any(self.attention_mask):
            raise ValueError("Invalid attention mask")
        if any(t < 0 or t >= vocab_size for t in self.tokens):
            raise ValueError("Input token outside model vocabulary")
        score = self.scoring
        if not 0 <= score.position < len(self.tokens) or not self.attention_mask[score.position]:
            raise ValueError("Prediction position must be a valid prompt token")
        candidates = score.targets + score.contrasts
        if not score.targets or len(set(candidates)) != len(candidates):
            raise ValueError("Candidate groups must be nonempty, unique and disjoint")
        if any(t < 0 or t >= vocab_size for t in candidates):
            raise ValueError("Candidate token outside model vocabulary")
        if score.metric == "target_log_probability":
            if len(score.targets) != 1 or score.contrasts:
                raise ValueError("Target log probability requires exactly one target")
        elif score.metric == "logit_difference":
            if len(score.targets) != 1 or len(score.contrasts) != 1:
                raise ValueError("Logit difference requires two single-token answers")
        elif score.metric == "probability_group_difference":
            if not score.contrasts:
                raise ValueError("Probability difference requires both candidate groups")
        else:
            raise ValueError(f"Unknown behavioral metric: {score.metric}")
