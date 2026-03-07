"""Summarize comparable Part II language-model runs from train_lm logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TARGET_MODELS = ("rnn", "lstm", "transformer")
TARGET_EMBEDS = ("trainable", "fixed_self", "fixed_pretrained")
EMBED_LABELS = {
    "trainable": "Trainable",
    "fixed_self": "Word2Vec",
    "fixed_pretrained": "GloVe",
}


@dataclass
class RunSummary:
    log_path: Path
    model: str
    embed: str
    epochs_completed: int
    best_val_ppl: float
    final_val_ppl: float
    mean_epoch_time: float
    glove_coverage: float | None


HEADER_RE = re.compile(r"run=\S+\s+model=(\w+)\s+embed=(\w+)\s+params=")
EPOCH_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_ppl=([0-9.]+).*time=([0-9.]+)s")
COVERAGE_RE = re.compile(r"glove_coverage=([0-9.]+)")


def parse_train_lm_log(path: Path) -> RunSummary | None:
    model = embed = None
    val_ppls: list[float] = []
    epoch_times: list[float] = []
    glove_coverage: float | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if model is None:
                match = HEADER_RE.search(line)
                if match:
                    model, embed = match.group(1), match.group(2)
                    continue
            match = EPOCH_RE.search(line)
            if match:
                val_ppls.append(float(match.group(3)))
                epoch_times.append(float(match.group(4)))
                continue
            match = COVERAGE_RE.search(line)
            if match:
                glove_coverage = float(match.group(1))

    if model is None or embed is None or not val_ppls:
        return None
    if model not in TARGET_MODELS or embed not in TARGET_EMBEDS:
        return None

    return RunSummary(
        log_path=path,
        model=model,
        embed=embed,
        epochs_completed=len(val_ppls),
        best_val_ppl=min(val_ppls),
        final_val_ppl=val_ppls[-1],
        mean_epoch_time=sum(epoch_times) / len(epoch_times),
        glove_coverage=glove_coverage,
    )


def choose_best_run(candidates: list[RunSummary]) -> RunSummary:
    # Prefer the most complete run; break ties by latest filename.
    return sorted(
        candidates,
        key=lambda run: (run.epochs_completed, run.log_path.name),
    )[-1]


def render_markdown(
    selected: dict[tuple[str, str], RunSummary],
    missing: list[tuple[str, str]],
    incomplete: list[tuple[str, str, int]],
    min_epochs: int,
) -> str:
    lines = [
        "# Part II Comparable Run Summary",
        "",
        "| Model | Embedding | Epochs | Best Val PPL | Final Val PPL | Mean Epoch Time (s) | Coverage | Log |",
        "|-------|-----------|--------|--------------|---------------|---------------------|----------|-----|",
    ]
    for model in TARGET_MODELS:
        for embed in TARGET_EMBEDS:
            run = selected.get((model, embed))
            if run is None:
                lines.append(
                    f"| {model.upper()} | {EMBED_LABELS[embed]} | — | — | — | — | — | MISSING |"
                )
                continue
            coverage = f"{run.glove_coverage:.4f}" if run.glove_coverage is not None else "—"
            log_note = f"`{run.log_path.name}`"
            if run.epochs_completed < min_epochs:
                log_note = f"{log_note} (INCOMPLETE)"
            lines.append(
                f"| {model.upper()} | {EMBED_LABELS[embed]} | {run.epochs_completed} | "
                f"{run.best_val_ppl:.4f} | {run.final_val_ppl:.4f} | {run.mean_epoch_time:.2f} | "
                f"{coverage} | {log_note} |"
            )

    if missing:
        lines.extend(
            [
                "",
                "## Missing Rows",
                "",
                *[f"- {model} / {embed}" for model, embed in missing],
            ]
        )
    else:
        lines.extend(["", "## Missing Rows", "", "- None"])
    if incomplete:
        lines.extend(
            [
                "",
                f"## Incomplete Rows (< {min_epochs} epochs)",
                "",
                *[f"- {model} / {embed}: {epochs} epochs" for model, embed, epochs in incomplete],
            ]
        )
    else:
        lines.extend(["", f"## Incomplete Rows (< {min_epochs} epochs)", "", "- None"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="outputs/logs", help="Directory containing train_lm logs.")
    parser.add_argument(
        "--output",
        default="outputs/part2_summary.md",
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=30,
        help="Treat runs with fewer epochs as incomplete.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    candidates: dict[tuple[str, str], list[RunSummary]] = {}
    for path in sorted(log_dir.glob("*.log")):
        summary = parse_train_lm_log(path)
        if summary is None:
            continue
        candidates.setdefault((summary.model, summary.embed), []).append(summary)

    selected = {key: choose_best_run(runs) for key, runs in candidates.items()}
    missing = [
        (model, embed)
        for model in TARGET_MODELS
        for embed in TARGET_EMBEDS
        if (model, embed) not in selected
    ]
    incomplete = [
        (model, embed, selected[(model, embed)].epochs_completed)
        for model in TARGET_MODELS
        for embed in TARGET_EMBEDS
        if (model, embed) in selected and selected[(model, embed)].epochs_completed < args.min_epochs
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(selected, missing, incomplete, args.min_epochs), encoding="utf-8")
    print(f"Wrote summary to {output_path}")
    if missing:
        print("Missing rows:")
        for model, embed in missing:
            print(f"  - {model} / {embed}")
    else:
        print("All 3x3 rows are present.")
    if incomplete:
        print(f"Incomplete rows (< {args.min_epochs} epochs):")
        for model, embed, epochs in incomplete:
            print(f"  - {model} / {embed}: {epochs} epochs")


if __name__ == "__main__":
    main()
