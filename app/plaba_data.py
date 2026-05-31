from __future__ import annotations

import random
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel


class Sample(BaseModel):
    source_text: str
    references: List[str]


def load_sentence_level_samples(csv_path: str | Path) -> list[Sample]:
    df = pd.read_csv(csv_path)
    grouped = (
        df.groupby(["question", "pmid", "input_text"], as_index=False)["target_text"]
        .apply(list)
        .rename(columns={"target_text": "references"})
    )
    samples = [
        Sample(source_text=row["input_text"], references=row["references"])
        for _, row in grouped.iterrows()
    ]
    return samples


def pick_random_samples(samples: list[Sample], k: int = 20, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    if k >= len(samples):
        return samples
    return rng.sample(samples, k)
