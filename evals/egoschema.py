from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import weave

if TYPE_CHECKING:
    from transformers import GenerationConfig
    from models.base import BaseVLM


# `\b` su input upper()-ato: matcha A..E come token isolato — quindi
# pesca la lettera dentro "(A)", "A.", "Answer: B" ma non dentro
# "Atlanta" / "BEFORE".
_LETTER_RE = re.compile(r"\b([A-E])\b")
# Le option in EgoSchema arrivano pre-formattate "A. ...": togliamo il
# prefisso così `format_mcq_prompt` rifa il rendering in modo uniforme.
_OPTION_PREFIX_RE = re.compile(r"^[A-E]\.\s*")


def load_egoschema(root: Path) -> list[dict]:
    """Carica le righe EgoSchema dal layout `prefetch_egoschema`.

    Le chiavi del jsonl sorgente sono quelle di `lmms-lab/egoschema`
    (`question_idx, question, video_idx, option, answer, video`); qui le
    rimappiamo ai nomi che Weave matcha agli argomenti di
    `predict` / `mcq_accuracy` (`video_path, question, options, answer`).
    """
    with (root / "metadata.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return [
        {
            "video_idx": r["video_idx"],
            "video_path": str(root / r["video"]),
            "question": r["question"],
            "options": [_OPTION_PREFIX_RE.sub("", o) for o in r["option"]],
            "answer": r.get("answer"),
        }
        for r in rows
    ]


def format_mcq_prompt(question: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
    return (
        f"Question: {question}\n"
        f"Options:\n{body}\n"
        "Answer with the letter of the correct option only."
    )


def parse_mcq_letter(raw: str, n_options: int) -> int | None:
    m = _LETTER_RE.search(raw.upper())
    if m is None:
        return None
    idx = ord(m.group(1)) - ord("A")
    return idx if idx < n_options else None


def make_predict(vlm: BaseVLM, gen_cfg: GenerationConfig, fps: float):
    """Costruisce la `@weave.op predict` chiusa su `vlm`+`gen_cfg`+`fps`.
    """
    from models import Text, Video

    @weave.op
    def predict(video_path: str, question: str, options: list[str]) -> dict:
        prompt = format_mcq_prompt(question, options)
        media = Video(video_path, fps=fps)
        messages = vlm.build_messages(media, Text(prompt))
        raw = vlm.generate(messages, generation_config=gen_cfg)
        return {"raw": raw, "pred": parse_mcq_letter(raw, len(options))}

    return predict


@weave.op
def mcq_accuracy(answer: int, output: dict) -> dict:
    return {"correct": output["pred"] == answer}
