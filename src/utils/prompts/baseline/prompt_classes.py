from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

Letter = Literal["A","B","C","D","E","F","G","H","I","J"]

class BaselineBM(BaseModel):
    """
    Structured output for time-series MCQ grading.
    """
    rationale: str = Field(
        ...,
        min_length=3,
        max_length=400,
        description="~2 sentences on why the chosen option is best; cite concrete TS properties."
    )
    option_scores: Optional[dict[Letter, float]] = Field(
        default=None,
        description="Optional per-option plausibility scores in [0,1]. Include only options that were shown."
    )
    final_answer: Letter = Field(
        ...,
        description="One uppercase letter A–J corresponding to the selected option."
    )
    answer_string: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Text of the selected option (e.g., 'WALKING'); mirrors the letter in final_answer."
    )


    # --- Field validators (v2) ---
    @field_validator("rationale")
    @classmethod
    def _trim_rationale(cls, v: str) -> str:
        return " ".join(v.strip().split())
    
    @field_validator("answer_string")
    @classmethod
    def _trim_answer_string(cls, v: str) -> str:
        return " ".join(v.strip().split())


    @field_validator("option_scores")
    @classmethod
    def _validate_scores(cls, v: Optional[dict[Letter, float]]) -> Optional[dict[Letter, float]]:
        if v is None:
            return v
        if not v:
            raise ValueError("option_scores, if provided, must not be empty.")
        for k, score in v.items():
            if not (0.0 <= float(score) <= 1.0):
                raise ValueError(f"option_scores[{k}] must be in [0.0, 1.0], got {score}.")
        return v

    @field_validator("final_answer")
    @classmethod
    def _uppercase(cls, v: str) -> str:
        return v.upper()

    # --- Cross-field validation ---
    @model_validator(mode="after")
    def _answer_present_if_scores(self) -> "BaselineBM":
        if self.option_scores is not None and self.final_answer not in self.option_scores:
            raise ValueError(
                f"final_answer '{self.final_answer}' must appear as a key in option_scores when option_scores is provided."
            )
        return self
