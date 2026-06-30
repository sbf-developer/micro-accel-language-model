"""Curriculum scheduler: easy-to-hard staged training (CLPD / L2M-KD inspired)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    name: str
    max_stage: int
    steps: int
    lr_scale: float = 1.0


DEFAULT_STAGES = [
    CurriculumStage("format", max_stage=0, steps=200, lr_scale=1.0),
    CurriculumStage("atomic", max_stage=1, steps=500, lr_scale=0.9),
    CurriculumStage("compose", max_stage=2, steps=350, lr_scale=0.7),
    CurriculumStage("dialogue", max_stage=3, steps=350, lr_scale=0.5),
]


class CurriculumScheduler:
    def __init__(self, stages: list[CurriculumStage] | None = None):
        self.stages = stages or DEFAULT_STAGES
        self._boundaries = []
        total = 0
        for stage in self.stages:
            total += stage.steps
            self._boundaries.append((total, stage))

    @property
    def total_steps(self) -> int:
        return self._boundaries[-1][0] if self._boundaries else 0

    def stage_at(self, step: int) -> CurriculumStage:
        for boundary, stage in self._boundaries:
            if step < boundary:
                return stage
        return self.stages[-1]

    def max_data_stage(self, step: int) -> int:
        return self.stage_at(step).max_stage

    def lr_scale(self, step: int) -> float:
        return self.stage_at(step).lr_scale

    def stage_name(self, step: int) -> str:
        return self.stage_at(step).name
