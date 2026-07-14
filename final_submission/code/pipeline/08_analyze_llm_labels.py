"""Analyze existing generative-AI labels for BOJ text chunks."""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "chunks.csv"
LABELS_PATH = PROJECT_ROOT / "results" / "llm" / "llm_labels.csv"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "llm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary"
REPORT_PATH = ANALYSIS_DIR / "llm_analysis_summary.md"

POLICY_DIRECTIONS = ("引き締め的", "中立的", "緩和的", "該当なし")
COMMUNICATION_FUNCTIONS = (
    "政策行動",
    "政策の根拠",
    "経済見通し",
    "リスク・不確実性",
    "期待形成",
    "背景・その他",
)
CERTAINTY_LABELS = ("明確", "条件付き", "慎重", "該当なし")
LABEL_FIELDS = {
    "policy_direction": POLICY_DIRECTIONS,
    "communication_function": COMMUNICATION_FUNCTIONS,
    "certainty": CERTAINTY_LABELS,
}

REQUIRED_LABEL_COLUMNS = (
    "chunk_id",
    "doc_id",
    "type",
    "period",
    "policy_direction",
    "communication_function",
    "certainty",
    "evidence",
)
REQUIRED_CHUNK_COLUMNS = (
    "chunk_id",
    "doc_id",
    "date",
    "period",
    "type",
    "speaker",
    "title",
    "section_title",
    "char_count",
    "text",
)
MERGED_COLUMNS = (
    "chunk_id",
    "doc_id",
    "date",
    "period",
    "phase",
    "type",
    "speaker",
    "title",
    "section_title",
    "char_count",
    "text",
    "policy_direction",
    "communication_function",
    "certainty",
    "evidence",
)

PHASE_MAP = {
    "before": "before",
    "transition": "post",
    "after": "post",
    "pre_end_nirp": "before",
    "end_nirp": "post",
    "post_end_nirp": "post",
}
TYPE_ORDER = ("statement", "speech")
PHASE_ORDER = ("before", "post")
CEREMONIAL_PATTERNS = (
    "本日は",
    "お招き",
    "ご紹介",
    "挨拶",
    "講演の機会",
    "ありがとうございました",
    "懇談会",
    "当地",
    "地域経済",
    "県内",
    "支店長",
    "皆様",
    "御礼",
)


class LLMAnalysisError(Exception):
    """Raised when LLM label analysis cannot continue safely."""


@dataclass(frozen=True)
class ValidationSummary:
    duplicate_label_ids: list[str]
    missing_label_ids: list[str]
    extra_label_ids: list[str]


@dataclass(frozen=True)
class OutputPaths:
    merged: Path
    overall_counts: Path
    by_type: Path
    by_phase: Path
    by_document: Path
    document_level: Path
    document_balanced_by_type: Path
    document_balanced_by_phase: Path
    direction_certainty_by_type: Path
    direction_certainty_by_phase: Path
    function_certainty_by_type: Path
    function_certainty_by_phase: Path
    narrative_indicators: Path
    review_candidates: Path
    representative_examples: Path
    figures: list[Path]
    report: Path

    def all_paths(self) -> list[Path]:
        return [
            self.merged,
            self.overall_counts,
            self.by_type,
            self.by_phase,
            self.by_document,
            self.document_level,
            self.document_balanced_by_type,
            self.document_balanced_by_phase,
            self.direction_certainty_by_type,
            self.direction_certainty_by_phase,
            self.function_certainty_by_type,
            self.function_certainty_by_phase,
            self.narrative_indicators,
            self.review_candidates,
            self.representative_examples,
            *self.figures,
            self.report,
        ]


def read_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise LLMAnalysisError(f"missing input file: {path}")

    try:
        data = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as exc:
        raise LLMAnalysisError(f"cannot read CSV {path}: {exc}") from exc

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise LLMAnalysisError(f"{path} is missing column(s): {', '.join(missing)}")
    return data


def strip_analysis_columns(labels: pd.DataFrame, chunks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    chunks = chunks.copy()

    for column in REQUIRED_LABEL_COLUMNS:
        labels[column] = labels[column].astype(str).str.strip()
    for column in REQUIRED_CHUNK_COLUMNS:
        chunks[column] = chunks[column].astype(str).str.strip()

    return labels, chunks


def validate_inputs(labels: pd.DataFrame, chunks: pd.DataFrame) -> ValidationSummary:
    duplicate_ids = sorted(
        labels.loc[labels["chunk_id"].duplicated(keep=False), "chunk_id"].unique().tolist()
    )

    chunk_ids = set(chunks["chunk_id"])
    label_ids = set(labels["chunk_id"])
    missing_ids = sorted(chunk_ids - label_ids)
    extra_ids = sorted(label_ids - chunk_ids)

    errors: list[str] = []
    if duplicate_ids:
        errors.append(f"duplicate chunk_id values in llm_labels.csv: {duplicate_ids}")
    if missing_ids:
        errors.append(f"missing labels for chunk_id values: {missing_ids}")
    if extra_ids:
        errors.append(f"unknown chunk_id values in llm_labels.csv: {extra_ids}")

    for column, allowed in LABEL_FIELDS.items():
        invalid = sorted(set(labels[column]) - set(allowed))
        if invalid:
            errors.append(f"{column} contains invalid value(s): {invalid}")

    substantive = (
        (labels["policy_direction"] != "該当なし")
        | (labels["communication_function"] != "背景・その他")
        | (labels["certainty"] != "該当なし")
    )
    empty_evidence = labels["evidence"].str.strip().eq("")
    if (substantive & empty_evidence).any():
        bad_ids = labels.loc[substantive & empty_evidence, "chunk_id"].tolist()
        errors.append(f"substantive classifications with empty evidence: {bad_ids}")

    if errors:
        raise LLMAnalysisError("; ".join(errors))

    return ValidationSummary(
        duplicate_label_ids=duplicate_ids,
        missing_label_ids=missing_ids,
        extra_label_ids=extra_ids,
    )


def merge_labels(labels: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    label_columns = ["chunk_id", "policy_direction", "communication_function", "certainty", "evidence"]
    merged = chunks[list(REQUIRED_CHUNK_COLUMNS)].merge(
        labels[label_columns], on="chunk_id", how="left", validate="one_to_one"
    )
    merged["phase"] = merged["period"].map(PHASE_MAP)
    if merged["phase"].isna().any():
        unknown = sorted(merged.loc[merged["phase"].isna(), "period"].unique().tolist())
        raise LLMAnalysisError(f"cannot map period value(s) to phase: {unknown}")

    merged["char_count"] = pd.to_numeric(merged["char_count"], errors="coerce").fillna(0).astype(int)
    return merged[list(MERGED_COLUMNS)]


def group_order(values: pd.Series, dimension: str) -> list[str]:
    observed = values.astype(str).drop_duplicates().tolist()
    if dimension == "type":
        return [value for value in TYPE_ORDER if value in set(observed)] + [
            value for value in observed if value not in TYPE_ORDER
        ]
    if dimension == "phase":
        return [value for value in PHASE_ORDER if value in set(observed)] + [
            value for value in observed if value not in PHASE_ORDER
        ]
    return observed


def make_label_summary(merged: pd.DataFrame, dimension: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if dimension is None:
        groups = [("overall", merged)]
        dimension_name = "overall"
    else:
        dimension_name = dimension
        groups = [
            (group, merged[merged[dimension] == group])
            for group in group_order(merged[dimension], dimension)
        ]

    for group, subset in groups:
        denominator = len(subset)
        for label_field, labels in LABEL_FIELDS.items():
            counts = subset[label_field].value_counts().reindex(labels, fill_value=0)
            for label in labels:
                count = int(counts[label])
                rows.append(
                    {
                        "dimension": dimension_name,
                        "group": group,
                        "label_field": label_field,
                        "label": label,
                        "count": count,
                        "proportion": safe_divide(count, denominator),
                        "weighting_method": "chunk_weighted_supplementary",
                    }
                )

    return pd.DataFrame(rows)


def make_document_level_proportions(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for doc_id, doc in merged.groupby("doc_id", sort=False):
        document_size = len(doc)
        first = doc.iloc[0]
        for label_field, labels in LABEL_FIELDS.items():
            counts = doc[label_field].value_counts().reindex(labels, fill_value=0)
            for label in labels:
                count = int(counts[label])
                rows.append(
                    {
                        "doc_id": doc_id,
                        "type": first["type"],
                        "period": first["period"],
                        "phase": first["phase"],
                        "speaker": first["speaker"],
                        "title": first["title"],
                        "label_field": label_field,
                        "label": label,
                        "count": count,
                        "document_chunk_count": document_size,
                        "proportion": safe_divide(count, document_size),
                        "weighting_method": "within_document",
                    }
                )
    return pd.DataFrame(rows)


def make_document_balanced(document_level: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group in group_order(document_level[dimension], dimension):
        subset = document_level[document_level[dimension] == group]
        document_count = subset["doc_id"].nunique()
        for label_field, labels in LABEL_FIELDS.items():
            field_subset = subset[subset["label_field"] == label_field]
            for label in labels:
                values = field_subset.loc[field_subset["label"] == label, "proportion"]
                rows.append(
                    {
                        "dimension": dimension,
                        "group": group,
                        "label_field": label_field,
                        "label": label,
                        "document_count": document_count,
                        "proportion": float(values.mean()) if not values.empty else 0.0,
                        "weighting_method": "document_balanced",
                    }
                )

    return pd.DataFrame(rows)


def make_joint_table(
    merged: pd.DataFrame,
    first_field: str,
    first_labels: tuple[str, ...],
    dimension: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    combinations = list(product(first_labels, CERTAINTY_LABELS))

    for group in group_order(merged[dimension], dimension):
        subset = merged[merged[dimension] == group]
        denominator = len(subset)
        counts = subset.groupby([first_field, "certainty"]).size()
        doc_ids = subset["doc_id"].drop_duplicates().tolist()

        for first_label, certainty in combinations:
            count = int(counts.get((first_label, certainty), 0))
            document_values = []
            for doc_id, doc in subset.groupby("doc_id", sort=False):
                doc_match = (doc[first_field] == first_label) & (doc["certainty"] == certainty)
                document_values.append(float(doc_match.mean()))

            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "first_label_field": first_field,
                    "first_label": first_label,
                    "certainty": certainty,
                    "label_pair": f"{first_label} × {certainty}",
                    "count": count,
                    "chunk_weighted_proportion": safe_divide(count, denominator),
                    "document_balanced_proportion": float(np.mean(document_values)) if document_values else 0.0,
                    "document_count": len(doc_ids),
                }
            )

    return pd.DataFrame(rows)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def add_indicator(
    rows: list[dict[str, object]],
    merged: pd.DataFrame,
    narrative: str,
    indicator: str,
    group: str,
    subset_mask: pd.Series,
    condition_mask: pd.Series,
    interpretation_note: str,
) -> None:
    subset = merged[subset_mask].copy()
    condition = condition_mask.loc[subset.index]

    chunk_weighted = float(condition.mean()) if len(subset) else 0.0
    doc_values = condition.groupby(subset["doc_id"]).mean() if len(subset) else pd.Series(dtype=float)
    document_balanced = float(doc_values.mean()) if not doc_values.empty else 0.0

    for method, value in (
        ("document_balanced", document_balanced),
        ("chunk_weighted_supplementary", chunk_weighted),
    ):
        rows.append(
            {
                "narrative": narrative,
                "indicator": indicator,
                "group": group,
                "value": value,
                "weighting_method": method,
                "interpretation_note": interpretation_note,
            }
        )


def make_narrative_indicators(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_rows = pd.Series(True, index=merged.index)
    note_a = "Higher values may be relevant to Narrative A, but do not prove it."
    note_b = "Higher values may be relevant to Narrative B, but do not prove it."

    for doc_type in group_order(merged["type"], "type"):
        for phase in group_order(merged["phase"], "phase"):
            subset = (merged["type"] == doc_type) & (merged["phase"] == phase)
            add_indicator(
                rows,
                merged,
                "Narrative A",
                "Proportion of 引き締め的 chunks by type and phase",
                f"type={doc_type}; phase={phase}",
                subset,
                merged["policy_direction"] == "引き締め的",
                note_a,
            )

    add_indicator(
        rows,
        merged,
        "Narrative A",
        "Proportion of 政策の根拠 chunks in speeches",
        "type=speech",
        merged["type"] == "speech",
        merged["communication_function"] == "政策の根拠",
        note_a,
    )
    for doc_type in group_order(merged["type"], "type"):
        add_indicator(
            rows,
            merged,
            "Narrative A",
            "Proportion of 明確 chunks by document type",
            f"type={doc_type}",
            merged["type"] == doc_type,
            merged["certainty"] == "明確",
            note_a,
        )

    post_speech = (merged["type"] == "speech") & (merged["phase"] == "post")
    for label_field, label in (
        ("policy_direction", "引き締め的"),
        ("communication_function", "政策の根拠"),
        ("certainty", "明確"),
    ):
        add_indicator(
            rows,
            merged,
            "Narrative A",
            f"Post-period speech proportion: {label_field}={label}",
            "type=speech; phase=post",
            post_speech,
            merged[label_field] == label,
            note_a,
        )
    add_indicator(
        rows,
        merged,
        "Narrative A",
        "Post-period speech joint proportion: 引き締め的 × 政策の根拠 × 明確",
        "type=speech; phase=post",
        post_speech,
        (merged["policy_direction"] == "引き締め的")
        & (merged["communication_function"] == "政策の根拠")
        & (merged["certainty"] == "明確"),
        note_a,
    )

    for label_field, label in (
        ("communication_function", "政策行動"),
        ("certainty", "明確"),
    ):
        add_indicator(
            rows,
            merged,
            "Narrative B",
            f"Statement proportion: {label_field}={label}",
            "type=statement",
            merged["type"] == "statement",
            merged[label_field] == label,
            note_b,
        )
    add_indicator(
        rows,
        merged,
        "Narrative B",
        "Statement joint proportion: 政策行動 × 明確",
        "type=statement",
        merged["type"] == "statement",
        (merged["communication_function"] == "政策行動") & (merged["certainty"] == "明確"),
        note_b,
    )

    for label_field, label in (
        ("certainty", "条件付き"),
        ("certainty", "慎重"),
        ("communication_function", "リスク・不確実性"),
        ("communication_function", "期待形成"),
    ):
        add_indicator(
            rows,
            merged,
            "Narrative B",
            f"Speech proportion: {label_field}={label}",
            "type=speech",
            merged["type"] == "speech",
            merged[label_field] == label,
            note_b,
        )

    for certainty in ("条件付き", "慎重"):
        add_indicator(
            rows,
            merged,
            "Narrative B",
            f"Speech joint proportion: 引き締め的 × {certainty}",
            "type=speech",
            merged["type"] == "speech",
            (merged["policy_direction"] == "引き締め的") & (merged["certainty"] == certainty),
            note_b,
        )
        add_indicator(
            rows,
            merged,
            "Narrative B",
            f"Post-period speech proportion: certainty={certainty}",
            "type=speech; phase=post",
            post_speech,
            merged["certainty"] == certainty,
            note_b,
        )

    return pd.DataFrame(rows)


def evidence_is_found(evidence: str, text: str) -> bool:
    evidence = evidence.strip()
    if not evidence:
        return False
    if evidence in text:
        return True

    # Treat simple whitespace differences as still findable.
    compact_evidence = "".join(evidence.split())
    compact_text = "".join(text.split())
    return bool(compact_evidence and compact_evidence in compact_text)


def evidence_visible_length(evidence: str) -> int:
    return len("".join(evidence.split()))


def is_ceremonial_or_regional(text: str) -> bool:
    return any(pattern in text for pattern in CEREMONIAL_PATTERNS)


def make_review_candidates(merged: pd.DataFrame) -> pd.DataFrame:
    combo_counts = (
        merged.groupby(["policy_direction", "communication_function", "certainty"])
        .size()
        .to_dict()
    )
    rows: list[dict[str, object]] = []

    for row in merged.itertuples(index=False):
        reasons: list[str] = []
        combo = (row.policy_direction, row.communication_function, row.certainty)

        if row.policy_direction == "該当なし" and row.certainty != "該当なし":
            reasons.append("policy_direction is 該当なし but certainty is substantive")
        if row.communication_function == "背景・その他" and row.policy_direction in {"引き締め的", "緩和的"}:
            reasons.append("background/other function paired with directional policy label")
        if evidence_visible_length(row.evidence) < 5:
            reasons.append("evidence has fewer than 5 visible characters")
        if not evidence_is_found(row.evidence, row.text):
            reasons.append("evidence text not found inside chunk text")
        if combo_counts.get(combo, 0) < 2:
            reasons.append("three-label combination appears fewer than 2 times")
        if is_ceremonial_or_regional(row.text):
            reasons.append("text appears ceremonial or regional-background oriented")
        if row.policy_direction == "引き締め的" and row.certainty == "慎重":
            reasons.append("引き締め的 paired with 慎重")
        if row.policy_direction == "緩和的" and row.certainty == "明確":
            reasons.append("緩和的 paired with 明確")

        if reasons:
            rows.append(
                {
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "type": row.type,
                    "phase": row.phase,
                    "text": row.text,
                    "policy_direction": row.policy_direction,
                    "communication_function": row.communication_function,
                    "certainty": row.certainty,
                    "evidence": row.evidence,
                    "review_reason": " | ".join(reasons),
                }
            )

    return pd.DataFrame(rows)


def select_examples(
    merged: pd.DataFrame,
    example_set: str,
    mask: pd.Series,
    limit: int = 3,
) -> pd.DataFrame:
    subset = merged[mask].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["evidence_found"] = [
        evidence_is_found(evidence, text)
        for evidence, text in zip(subset["evidence"], subset["text"], strict=False)
    ]
    subset["evidence_length"] = subset["evidence"].map(evidence_visible_length)
    subset["is_ceremonial"] = subset["text"].map(is_ceremonial_or_regional)
    subset["moderate_length_distance"] = (subset["char_count"] - 250).abs()
    subset = subset.sort_values(
        by=["is_ceremonial", "evidence_found", "evidence_length", "moderate_length_distance", "chunk_id"],
        ascending=[True, False, False, True, True],
    ).drop_duplicates(subset=["text"])
    subset = subset.head(limit).copy()
    subset.insert(0, "rank", range(1, len(subset) + 1))
    subset.insert(0, "example_set", example_set)
    return subset[
        [
            "example_set",
            "rank",
            "chunk_id",
            "doc_id",
            "type",
            "phase",
            "period",
            "policy_direction",
            "communication_function",
            "certainty",
            "evidence",
            "char_count",
            "text",
        ]
    ]


def make_representative_examples(merged: pd.DataFrame) -> pd.DataFrame:
    criteria: list[tuple[str, pd.Series]] = []
    for label in ("引き締め的", "中立的", "緩和的"):
        criteria.append((f"policy_direction={label}", merged["policy_direction"] == label))
    for label in ("政策行動", "政策の根拠", "経済見通し", "リスク・不確実性", "期待形成"):
        criteria.append((f"communication_function={label}", merged["communication_function"] == label))
    for label in ("明確", "条件付き", "慎重"):
        criteria.append((f"certainty={label}", merged["certainty"] == label))

    criteria.extend(
        [
            (
                "statement: 政策行動 × 明確",
                (merged["type"] == "statement")
                & (merged["communication_function"] == "政策行動")
                & (merged["certainty"] == "明確"),
            ),
            (
                "speech: 政策の根拠",
                (merged["type"] == "speech") & (merged["communication_function"] == "政策の根拠"),
            ),
            (
                "speech: リスク・不確実性 × 慎重",
                (merged["type"] == "speech")
                & (merged["communication_function"] == "リスク・不確実性")
                & (merged["certainty"] == "慎重"),
            ),
            (
                "speech: 引き締め的 × 条件付き",
                (merged["type"] == "speech")
                & (merged["policy_direction"] == "引き締め的")
                & (merged["certainty"] == "条件付き"),
            ),
            ("phase=before", merged["phase"] == "before"),
            ("phase=post", merged["phase"] == "post"),
        ]
    )

    examples = [select_examples(merged, name, mask) for name, mask in criteria]
    examples = [example for example in examples if not example.empty]
    return pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


def configure_japanese_font() -> None:
    candidates = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            return
    print("WARNING: Japanese-compatible matplotlib font not found; figures may show missing glyphs.")


def wrap_japanese_label(label: str, width: int = 14) -> str:
    wrapped = textwrap.wrap(
        str(label),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return "\n".join(wrapped) if wrapped else str(label)


def proportion_axis_limit(max_value: float) -> float:
    return max(1.0, max_value + 0.12)


def add_horizontal_value_labels(ax: plt.Axes, bars, x_limit: float) -> None:
    offset = x_limit * 0.01
    for bar in bars:
        value = float(bar.get_width())
        text_x = min(value + offset, x_limit * 0.98)
        align = "left" if text_x < x_limit * 0.96 else "right"
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=align,
            fontsize=8,
        )


def dynamic_left_margin(labels: list[str]) -> float:
    longest_line = max(
        (len(line) for label in labels for line in str(label).splitlines()),
        default=0,
    )
    return min(0.50, max(0.30, 0.16 + longest_line * 0.012))


def plot_label_distribution(
    balanced: pd.DataFrame,
    label_field: str,
    dimension: str,
    path: Path,
    title: str,
) -> None:
    subset = balanced[(balanced["label_field"] == label_field) & (balanced["dimension"] == dimension)]
    labels = list(LABEL_FIELDS[label_field])
    pivot = (
        subset.pivot(index="label", columns="group", values="proportion")
        .reindex(labels)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Average within-document proportion")
    ax.set_ylim(0, 1)
    ax.legend(title=dimension)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_joint_distribution(
    joint: pd.DataFrame,
    important_pairs: tuple[tuple[str, str], ...],
    path: Path,
    title: str,
) -> None:
    labels = [f"{first} × {certainty}" for first, certainty in important_pairs]
    subset = joint[joint["label_pair"].isin(labels)].copy()
    pivot = (
        subset.pivot(index="label_pair", columns="group", values="document_balanced_proportion")
        .reindex(labels)
        .fillna(0.0)
    )

    categories = pivot.index.tolist()
    wrapped_categories = [wrap_japanese_label(label) for label in categories]
    groups = pivot.columns.tolist()
    y_positions = np.arange(len(categories), dtype=float)
    bar_height = min(0.35, 0.8 / max(len(groups), 1))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * bar_height
    max_value = float(pivot.to_numpy().max()) if not pivot.empty else 0.0
    x_limit = proportion_axis_limit(max_value)

    fig_height = max(5, len(categories) * 0.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    for group, offset in zip(groups, offsets, strict=False):
        values = pivot[group].to_numpy(dtype=float)
        bars = ax.barh(y_positions + offset, values, height=bar_height, label=group)
        add_horizontal_value_labels(ax, bars, x_limit)

    ax.set_title(title)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_categories)
    ax.set_xlim(0, x_limit)
    ax.invert_yaxis()
    ax.legend(title="type")
    fig.tight_layout()
    fig.subplots_adjust(left=dynamic_left_margin(wrapped_categories))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_narrative_indicators(indicators: pd.DataFrame, path: Path) -> None:
    subset = indicators[indicators["weighting_method"] == "document_balanced"].copy()
    subset["plot_label"] = (
        subset["narrative"].str.replace("Narrative ", "", regex=False)
        + " | "
        + subset["group"]
        + " | "
        + subset["indicator"].str.replace("Proportion of ", "", regex=False)
        .str.replace(" proportion", "", regex=False)
    )
    subset = subset.sort_values(["narrative", "indicator", "group"]).tail(35)
    subset["wrapped_label"] = subset["plot_label"].map(lambda label: wrap_japanese_label(label, width=34))

    max_label_lines = max((label.count("\n") + 1 for label in subset["wrapped_label"]), default=1)
    per_category_height = 0.42 + min(max_label_lines, 4) * 0.16
    fig_height = max(6, len(subset) * per_category_height)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    bars = ax.barh(subset["wrapped_label"], subset["value"])
    max_value = float(subset["value"].max()) if not subset.empty else 0.0
    x_limit = proportion_axis_limit(max_value)
    add_horizontal_value_labels(ax, bars, x_limit)
    ax.set_title("Narrative indicators (document-balanced)")
    ax.set_xlabel("Proportion")
    ax.set_xlim(0, x_limit)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.subplots_adjust(left=dynamic_left_margin(subset["wrapped_label"].tolist()))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_figures(
    by_type_balanced: pd.DataFrame,
    by_phase_balanced: pd.DataFrame,
    direction_certainty_by_type: pd.DataFrame,
    function_certainty_by_type: pd.DataFrame,
    narrative_indicators: pd.DataFrame,
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_japanese_font()

    figures = [
        FIGURES_DIR / "policy_direction_by_type.png",
        FIGURES_DIR / "policy_direction_by_phase.png",
        FIGURES_DIR / "communication_function_by_type.png",
        FIGURES_DIR / "communication_function_by_phase.png",
        FIGURES_DIR / "certainty_by_type.png",
        FIGURES_DIR / "certainty_by_phase.png",
        FIGURES_DIR / "direction_certainty_by_type.png",
        FIGURES_DIR / "function_certainty_by_type.png",
        FIGURES_DIR / "narrative_indicators.png",
    ]

    plot_label_distribution(
        by_type_balanced,
        "policy_direction",
        "type",
        figures[0],
        "Policy direction by type (document-balanced)",
    )
    plot_label_distribution(
        by_phase_balanced,
        "policy_direction",
        "phase",
        figures[1],
        "Policy direction by phase (document-balanced)",
    )
    plot_label_distribution(
        by_type_balanced,
        "communication_function",
        "type",
        figures[2],
        "Communication function by type (document-balanced)",
    )
    plot_label_distribution(
        by_phase_balanced,
        "communication_function",
        "phase",
        figures[3],
        "Communication function by phase (document-balanced)",
    )
    plot_label_distribution(
        by_type_balanced,
        "certainty",
        "type",
        figures[4],
        "Certainty by type (document-balanced)",
    )
    plot_label_distribution(
        by_phase_balanced,
        "certainty",
        "phase",
        figures[5],
        "Certainty by phase (document-balanced)",
    )
    plot_joint_distribution(
        direction_certainty_by_type,
        (
            ("引き締め的", "明確"),
            ("引き締め的", "条件付き"),
            ("引き締め的", "慎重"),
            ("緩和的", "明確"),
            ("緩和的", "条件付き"),
        ),
        figures[6],
        "Policy direction × certainty by type (document-balanced)",
    )
    plot_joint_distribution(
        function_certainty_by_type,
        (
            ("政策行動", "明確"),
            ("政策の根拠", "条件付き"),
            ("リスク・不確実性", "慎重"),
            ("期待形成", "条件付き"),
        ),
        figures[7],
        "Communication function × certainty by type (document-balanced)",
    )
    plot_narrative_indicators(narrative_indicators, figures[8])
    return figures


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    subset = data.loc[:, columns].copy()
    if max_rows is not None:
        subset = subset.head(max_rows)
    if subset.empty:
        return "_No rows._"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in subset.itertuples(index=False):
        values = [format_markdown_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_markdown_summary(
    path: Path,
    merged: pd.DataFrame,
    validation: ValidationSummary,
    overall: pd.DataFrame,
    by_type_balanced: pd.DataFrame,
    by_phase_balanced: pd.DataFrame,
    narrative_indicators: pd.DataFrame,
    review_candidates: pd.DataFrame,
) -> None:
    overall_table = overall[["label_field", "label", "count", "proportion"]]
    type_table = by_type_balanced[["group", "label_field", "label", "proportion"]]
    phase_table = by_phase_balanced[["group", "label_field", "label", "proportion"]]
    narrative_a = narrative_indicators[
        (narrative_indicators["narrative"] == "Narrative A")
        & (narrative_indicators["weighting_method"] == "document_balanced")
    ][["indicator", "group", "value", "interpretation_note"]]
    narrative_b = narrative_indicators[
        (narrative_indicators["narrative"] == "Narrative B")
        & (narrative_indicators["weighting_method"] == "document_balanced")
    ][["indicator", "group", "value", "interpretation_note"]]

    content = f"""# LLM Classification Analysis Summary

## Data and validation

- Merged chunks: {len(merged)}
- Duplicate IDs in llm_labels.csv: {len(validation.duplicate_label_ids)}
- Missing expected IDs: {len(validation.missing_label_ids)}
- Extra unknown IDs: {len(validation.extra_label_ids)}
- Phase mapping used only for analysis: `before` and `pre_end_nirp` map to `before`; `transition`, `after`, `end_nirp`, and `post_end_nirp` map to `post`.

## Overall label distribution

{markdown_table(overall_table, ["label_field", "label", "count", "proportion"])}

## Statement vs speech

Document-balanced label proportions are the primary comparison because speeches contain more chunks than statements.

{markdown_table(type_table, ["group", "label_field", "label", "proportion"])}

## Before vs post

Document-balanced phase proportions use the analysis-only `phase` column.

{markdown_table(phase_table, ["group", "label_field", "label", "proportion"])}

## Evidence relevant to Narrative A

Narrative A: "Policy moves forward, and language reinforces it."

{markdown_table(narrative_a, ["indicator", "group", "value", "interpretation_note"])}

## Evidence relevant to Narrative B

Narrative B: "Policy moves forward, but language does not rush."

{markdown_table(narrative_b, ["indicator", "group", "value", "interpretation_note"])}

## Ambiguous and boundary cases

- Review candidates flagged: {len(review_candidates)}

{markdown_table(review_candidates, ["chunk_id", "doc_id", "type", "phase", "review_reason"], max_rows=10)}

## Limitations of the LLM classification

- The labels depend on the fixed prompt.
- The model may interpret conditional language inconsistently.
- The chunks are not statistically independent.
- Speech and statement genres differ.
- Classification does not prove speaker intention.
- Classification does not prove market effects.
- Evidence quotations support reproducibility but do not remove model subjectivity.

No final conclusion choosing Narrative A or Narrative B is provided here.
"""
    path.write_text(content, encoding="utf-8")


def write_outputs(
    merged: pd.DataFrame,
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    by_phase: pd.DataFrame,
    by_document: pd.DataFrame,
    document_level: pd.DataFrame,
    document_balanced_by_type: pd.DataFrame,
    document_balanced_by_phase: pd.DataFrame,
    direction_certainty_by_type: pd.DataFrame,
    direction_certainty_by_phase: pd.DataFrame,
    function_certainty_by_type: pd.DataFrame,
    function_certainty_by_phase: pd.DataFrame,
    narrative_indicators: pd.DataFrame,
    review_candidates: pd.DataFrame,
    representative_examples: pd.DataFrame,
    validation: ValidationSummary,
) -> OutputPaths:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    paths = OutputPaths(
        merged=ANALYSIS_DIR / "llm_labels_merged.csv",
        overall_counts=ANALYSIS_DIR / "label_counts_overall.csv",
        by_type=ANALYSIS_DIR / "label_proportions_by_type.csv",
        by_phase=ANALYSIS_DIR / "label_proportions_by_phase.csv",
        by_document=ANALYSIS_DIR / "label_proportions_by_document.csv",
        document_level=ANALYSIS_DIR / "document_level_label_proportions.csv",
        document_balanced_by_type=ANALYSIS_DIR / "document_balanced_by_type.csv",
        document_balanced_by_phase=ANALYSIS_DIR / "document_balanced_by_phase.csv",
        direction_certainty_by_type=ANALYSIS_DIR / "direction_certainty_by_type.csv",
        direction_certainty_by_phase=ANALYSIS_DIR / "direction_certainty_by_phase.csv",
        function_certainty_by_type=ANALYSIS_DIR / "function_certainty_by_type.csv",
        function_certainty_by_phase=ANALYSIS_DIR / "function_certainty_by_phase.csv",
        narrative_indicators=ANALYSIS_DIR / "narrative_indicators.csv",
        review_candidates=ANALYSIS_DIR / "llm_review_candidates.csv",
        representative_examples=ANALYSIS_DIR / "representative_llm_examples.csv",
        figures=[],
        report=REPORT_PATH,
    )

    merged.to_csv(paths.merged, index=False, encoding="utf-8-sig")
    overall.to_csv(paths.overall_counts, index=False, encoding="utf-8-sig")
    by_type.to_csv(paths.by_type, index=False, encoding="utf-8-sig")
    by_phase.to_csv(paths.by_phase, index=False, encoding="utf-8-sig")
    by_document.to_csv(paths.by_document, index=False, encoding="utf-8-sig")
    document_level.to_csv(paths.document_level, index=False, encoding="utf-8-sig")
    document_balanced_by_type.to_csv(paths.document_balanced_by_type, index=False, encoding="utf-8-sig")
    document_balanced_by_phase.to_csv(paths.document_balanced_by_phase, index=False, encoding="utf-8-sig")
    direction_certainty_by_type.to_csv(paths.direction_certainty_by_type, index=False, encoding="utf-8-sig")
    direction_certainty_by_phase.to_csv(paths.direction_certainty_by_phase, index=False, encoding="utf-8-sig")
    function_certainty_by_type.to_csv(paths.function_certainty_by_type, index=False, encoding="utf-8-sig")
    function_certainty_by_phase.to_csv(paths.function_certainty_by_phase, index=False, encoding="utf-8-sig")
    narrative_indicators.to_csv(paths.narrative_indicators, index=False, encoding="utf-8-sig")
    review_candidates.to_csv(paths.review_candidates, index=False, encoding="utf-8-sig")
    representative_examples.to_csv(paths.representative_examples, index=False, encoding="utf-8-sig")

    figures = write_figures(
        document_balanced_by_type,
        document_balanced_by_phase,
        direction_certainty_by_type,
        function_certainty_by_type,
        narrative_indicators,
    )
    write_markdown_summary(
        paths.report,
        merged,
        validation,
        overall,
        document_balanced_by_type,
        document_balanced_by_phase,
        narrative_indicators,
        review_candidates,
    )

    return OutputPaths(**{**paths.__dict__, "figures": figures})


def print_dataframe(title: str, data: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 40) -> None:
    print(title)
    if data.empty:
        print("(no rows)")
    else:
        view = data.loc[:, columns] if columns else data
        print(view.head(max_rows).to_string(index=False))
    print()


def print_console_summary(
    merged: pd.DataFrame,
    validation: ValidationSummary,
    overall: pd.DataFrame,
    document_balanced_by_type: pd.DataFrame,
    document_balanced_by_phase: pd.DataFrame,
    narrative_indicators: pd.DataFrame,
    review_candidates: pd.DataFrame,
    outputs: OutputPaths,
) -> None:
    print("LLM label analysis summary:")
    print(f"total merged chunks: {len(merged)}")
    print(
        "missing or duplicate IDs: "
        f"missing={len(validation.missing_label_ids)}, "
        f"duplicates={len(validation.duplicate_label_ids)}, "
        f"extra={len(validation.extra_label_ids)}"
    )
    print("count of chunks per document:")
    print(merged.groupby("doc_id").size().to_string())
    print()

    print_dataframe(
        "Label distributions:",
        overall,
        ["label_field", "label", "count", "proportion"],
    )
    print_dataframe(
        "Document-balanced comparison by type:",
        document_balanced_by_type,
        ["group", "label_field", "label", "proportion"],
    )
    print_dataframe(
        "Document-balanced comparison by phase:",
        document_balanced_by_phase,
        ["group", "label_field", "label", "proportion"],
    )
    print_dataframe(
        "Narrative indicator values:",
        narrative_indicators[narrative_indicators["weighting_method"] == "document_balanced"],
        ["narrative", "indicator", "group", "value", "weighting_method"],
        max_rows=80,
    )

    print(f"number of review candidates: {len(review_candidates)}")
    print_dataframe(
        "Top 10 review candidates for manual inspection:",
        review_candidates,
        ["chunk_id", "doc_id", "type", "phase", "policy_direction", "communication_function", "certainty", "review_reason"],
        max_rows=10,
    )

    print("Generated file paths:")
    for path in outputs.all_paths():
        print(path)


def run_analysis() -> tuple[pd.DataFrame, OutputPaths]:
    labels = read_csv(LABELS_PATH, REQUIRED_LABEL_COLUMNS)
    chunks = read_csv(CHUNKS_PATH, REQUIRED_CHUNK_COLUMNS)
    labels, chunks = strip_analysis_columns(labels, chunks)
    validation = validate_inputs(labels, chunks)
    merged = merge_labels(labels, chunks)

    overall = make_label_summary(merged)
    by_type = make_label_summary(merged, "type")
    by_phase = make_label_summary(merged, "phase")
    by_document = make_label_summary(merged, "doc_id")
    document_level = make_document_level_proportions(merged)
    document_balanced_by_type = make_document_balanced(document_level, "type")
    document_balanced_by_phase = make_document_balanced(document_level, "phase")
    direction_certainty_by_type = make_joint_table(
        merged, "policy_direction", POLICY_DIRECTIONS, "type"
    )
    direction_certainty_by_phase = make_joint_table(
        merged, "policy_direction", POLICY_DIRECTIONS, "phase"
    )
    function_certainty_by_type = make_joint_table(
        merged, "communication_function", COMMUNICATION_FUNCTIONS, "type"
    )
    function_certainty_by_phase = make_joint_table(
        merged, "communication_function", COMMUNICATION_FUNCTIONS, "phase"
    )
    narrative_indicators = make_narrative_indicators(merged)
    review_candidates = make_review_candidates(merged)
    representative_examples = make_representative_examples(merged)

    outputs = write_outputs(
        merged,
        overall,
        by_type,
        by_phase,
        by_document,
        document_level,
        document_balanced_by_type,
        document_balanced_by_phase,
        direction_certainty_by_type,
        direction_certainty_by_phase,
        function_certainty_by_type,
        function_certainty_by_phase,
        narrative_indicators,
        review_candidates,
        representative_examples,
        validation,
    )

    print_console_summary(
        merged,
        validation,
        overall,
        document_balanced_by_type,
        document_balanced_by_phase,
        narrative_indicators,
        review_candidates,
        outputs,
    )
    return merged, outputs


def main() -> int:
    try:
        run_analysis()
    except LLMAnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: file operation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
