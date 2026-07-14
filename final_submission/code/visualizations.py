"""Recreate final figures from packaged analytical result files only."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
PRESENTATION_DIR = PROJECT_ROOT / "figures" / "presentation"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "figures" / "supplementary"

TYPE_LABELS = {"statement": "政策声明", "speech": "講演"}
PHASE_LABELS = {"before": "終了前", "post": "終了期以降"}
TYPE_ORDER = ("statement", "speech")
PHASE_ORDER = ("before", "post")
CLUSTERS = tuple(range(5))
LABEL_ORDERS = {
    "policy_direction": ("引き締め的", "中立的", "緩和的", "該当なし"),
    "communication_function": (
        "政策行動",
        "政策の根拠",
        "経済見通し",
        "リスク・不確実性",
        "期待形成",
        "背景・その他",
    ),
    "certainty": ("明確", "条件付き", "慎重", "該当なし"),
}


class VisualizationError(Exception):
    """Raised when a packaged result cannot be plotted safely."""


def configure_japanese_font() -> None:
    """Use the first available Japanese-capable matplotlib font."""
    candidates = (
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "Hiragino Sans",
        "MS Gothic",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            return
    print("WARNING: Japanese-compatible font was not found; glyphs may be missing.")


def read_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Read a packaged CSV and validate its required columns."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    except Exception as exc:
        raise VisualizationError(f"cannot read {path}: {exc}") from exc
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise VisualizationError(f"{path} is missing columns: {', '.join(missing)}")
    return data


def wrap_japanese_label(label: str, width: int = 14) -> str:
    """Wrap a Japanese label without changing its characters."""
    parts = textwrap.wrap(
        str(label),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return "\n".join(parts) if parts else str(label)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _add_horizontal_labels(ax: plt.Axes, bars, limit: float, decimals: int = 2) -> None:
    offset = max(limit * 0.012, 0.002)
    lower, upper = ax.get_xlim()
    for bar in bars:
        value = float(bar.get_width())
        if value >= 0:
            x = min(value + offset, upper - offset)
            align = "left" if x < upper - 2 * offset else "right"
        else:
            x = max(value - offset, lower + offset)
            align = "right" if x > lower + 2 * offset else "left"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.{decimals}f}",
            va="center",
            ha=align,
            fontsize=9,
        )


def _add_vertical_labels(ax: plt.Axes, bars, limit: float) -> None:
    offset = max(limit * 0.012, 0.002)
    for bar in bars:
        value = float(bar.get_height())
        y = min(value + offset, limit - offset)
        align = "bottom" if y < limit - 2 * offset else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.2f}",
            ha="center",
            va=align,
            fontsize=9,
        )


def _select_contrast_terms(data: pd.DataFrame, count: int) -> pd.DataFrame:
    numeric = data.copy()
    numeric["difference"] = pd.to_numeric(numeric["difference"], errors="coerce")
    if numeric["difference"].isna().any():
        raise VisualizationError("TF-IDF contrast contains non-numeric differences")
    statement = numeric[numeric["difference"] > 0].nlargest(count, "difference")
    speech = numeric[numeric["difference"] < 0].nsmallest(count, "difference")
    if len(statement) < count or len(speech) < count:
        raise VisualizationError("TF-IDF contrast lacks enough terms on both sides")
    return (
        pd.concat([speech, statement], ignore_index=True)
        .sort_values("difference")
        .reset_index(drop=True)
    )


def _plot_tfidf_contrast(source: Path, output: Path, title: str, count: int) -> Path:
    data = read_csv(source, ("term", "difference"))
    selected = _select_contrast_terms(data, count)
    values = selected["difference"].to_numpy(dtype=float)
    positions = np.arange(len(selected))
    limit = float(np.abs(values).max()) * 1.30

    fig, ax = plt.subplots(figsize=(11, max(6, len(selected) * 0.38)))
    bars = ax.barh(positions, values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(selected["term"].astype(str))
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("政策声明スコア − 講演スコア")
    ax.set_title(title)
    ax.text(-limit * 0.97, len(selected) - 0.5, "講演に特徴的", ha="left", va="bottom")
    ax.text(limit * 0.97, len(selected) - 0.5, "政策声明に特徴的", ha="right", va="bottom")
    _add_horizontal_labels(ax, bars, limit, decimals=3)
    return _save_figure(fig, output)


def plot_tfidf_noun_contrast() -> Path:
    return _plot_tfidf_contrast(
        RESULTS_DIR / "tfidf_nouns" / "tfidf_noun_type_contrast.csv",
        PRESENTATION_DIR / "tfidf_statement_speech_contrast_nouns.png",
        "名詞のみTF-IDF：政策声明と講演の特徴語",
        count=8,
    )


def plot_tfidf_original_contrast() -> Path:
    return _plot_tfidf_contrast(
        RESULTS_DIR / "tfidf" / "tfidf_type_contrast.csv",
        SUPPLEMENTARY_DIR / "tfidf_statement_speech_contrast_original.png",
        "TF-IDF：政策声明と講演の特徴語",
        count=10,
    )


def _load_cluster_labels() -> dict[int, str]:
    labels = read_csv(METADATA_DIR / "cluster_labels_k5.csv", ("cluster", "label"))
    labels["cluster"] = pd.to_numeric(labels["cluster"], errors="coerce")
    if labels["cluster"].isna().any():
        raise VisualizationError("cluster label mapping contains a non-numeric cluster")
    mapping = {
        int(row.cluster): str(row.label).strip()
        for row in labels.itertuples(index=False)
        if str(row.label).strip()
    }
    missing = sorted(set(CLUSTERS) - set(mapping))
    if missing:
        raise VisualizationError(f"cluster label mapping is missing: {missing}")
    return mapping


def plot_embedding_cluster_distribution() -> Path:
    source = RESULTS_DIR / "embedding" / "presentation_embedding_cluster_distribution.csv"
    data = read_csv(
        source,
        ("cluster", "type", "document_balanced_proportion"),
    )
    labels = _load_cluster_labels()
    data["cluster"] = pd.to_numeric(data["cluster"], errors="coerce")
    data["document_balanced_proportion"] = pd.to_numeric(
        data["document_balanced_proportion"], errors="coerce"
    )
    if data[["cluster", "document_balanced_proportion"]].isna().any().any():
        raise VisualizationError("embedding cluster distribution contains invalid numbers")
    data["cluster"] = data["cluster"].astype(int)
    if not data["document_balanced_proportion"].between(0, 1).all():
        raise VisualizationError("embedding cluster proportions must be between 0 and 1")

    pivot = (
        data.pivot(index="cluster", columns="type", values="document_balanced_proportion")
        .reindex(CLUSTERS)
        .fillna(0.0)
    )
    groups = [group for group in TYPE_ORDER if group in pivot.columns]
    if not groups:
        raise VisualizationError("embedding cluster distribution has no document types")
    positions = np.arange(len(pivot), dtype=float)
    bar_height = min(0.34, 0.8 / len(groups))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * bar_height
    max_value = float(pivot[groups].to_numpy().max())
    limit = max(1.0, max_value + 0.12)

    fig, ax = plt.subplots(figsize=(11, max(5, len(pivot) * 0.8)))
    for group, offset in zip(groups, offsets, strict=False):
        bars = ax.barh(
            positions + offset,
            pivot[group].to_numpy(dtype=float),
            height=bar_height,
            label=TYPE_LABELS.get(group, group),
        )
        _add_horizontal_labels(ax, bars, limit)
    ax.set_yticks(positions)
    ax.set_yticklabels([wrap_japanese_label(labels[cluster]) for cluster in CLUSTERS])
    ax.set_xlim(0, limit)
    ax.set_xlabel("文書均等化比率")
    ax.set_title("埋め込みクラスタ分布（文書種別）")
    ax.legend(title="文書種別")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    return _save_figure(fig, PRESENTATION_DIR / "embedding_cluster_distribution_by_type.png")


def plot_embedding_pca() -> Path:
    embeddings = read_csv(RESULTS_DIR / "embedding" / "chunk_embeddings.csv", ("chunk_id", "type"))
    assignments = read_csv(
        RESULTS_DIR / "embedding" / "clustering_assignments.csv",
        ("chunk_id", "cluster_k5"),
    )
    if embeddings["chunk_id"].duplicated().any() or assignments["chunk_id"].duplicated().any():
        raise VisualizationError("embedding or cluster assignment chunk_id is not unique")
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise VisualizationError("chunk_embeddings.csv has no embedding dimensions")
    embeddings[embedding_columns] = embeddings[embedding_columns].apply(pd.to_numeric, errors="coerce")
    assignments["cluster_k5"] = pd.to_numeric(assignments["cluster_k5"], errors="coerce")
    if embeddings[embedding_columns].isna().any().any() or assignments["cluster_k5"].isna().any():
        raise VisualizationError("embedding inputs contain missing or non-numeric values")

    merged = embeddings.merge(
        assignments[["chunk_id", "cluster_k5"]],
        on="chunk_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(embeddings) or len(merged) != len(assignments):
        raise VisualizationError("chunk_id values do not match between embeddings and assignments")
    merged["cluster_k5"] = merged["cluster_k5"].astype(int)
    if not set(merged["cluster_k5"]).issubset(CLUSTERS):
        raise VisualizationError("cluster_k5 contains an unexpected cluster")

    coordinates = PCA(n_components=2).fit_transform(merged[embedding_columns].to_numpy(dtype=float))
    merged["pc1"] = coordinates[:, 0]
    merged["pc2"] = coordinates[:, 1]
    labels = _load_cluster_labels()
    markers = {"statement": "o", "speech": "^"}

    fig, ax = plt.subplots(figsize=(10, 7))
    last_scatter = None
    for doc_type in TYPE_ORDER:
        subset = merged[merged["type"] == doc_type]
        if subset.empty:
            continue
        last_scatter = ax.scatter(
            subset["pc1"],
            subset["pc2"],
            c=subset["cluster_k5"],
            vmin=min(CLUSTERS),
            vmax=max(CLUSTERS),
            marker=markers[doc_type],
            alpha=0.78,
            label=TYPE_LABELS[doc_type],
        )
    if last_scatter is None:
        raise VisualizationError("embedding PCA has no plottable document types")
    for cluster in CLUSTERS:
        subset = merged[merged["cluster_k5"] == cluster]
        if not subset.empty:
            ax.text(
                float(subset["pc1"].mean()),
                float(subset["pc2"].mean()),
                wrap_japanese_label(labels[cluster], width=10),
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("埋め込み空間のPCA投影")
    ax.legend(title="文書種別")
    fig.colorbar(last_scatter, ax=ax, ticks=CLUSTERS, label="クラスタ")
    return _save_figure(fig, SUPPLEMENTARY_DIR / "embedding_pca_scatter.png")


def _load_balanced_labels(scope: str, label_field: str) -> tuple[pd.DataFrame, list[str]]:
    path = RESULTS_DIR / "llm" / f"document_balanced_by_{scope}.csv"
    data = read_csv(
        path,
        ("dimension", "group", "label_field", "label", "proportion", "weighting_method"),
    )
    data = data[data["label_field"] == label_field].copy()
    if data.empty:
        raise VisualizationError(f"{path} has no rows for {label_field}")
    if not (data["weighting_method"] == "document_balanced").all():
        raise VisualizationError(f"{path} is not document-balanced")
    data["proportion"] = pd.to_numeric(data["proportion"], errors="coerce")
    if data["proportion"].isna().any() or not data["proportion"].between(0, 1).all():
        raise VisualizationError(f"{path} contains invalid proportions")
    order = [label for label in LABEL_ORDERS[label_field] if label in set(data["label"])]
    return data, order


def _plot_balanced_horizontal(
    scope: str,
    label_field: str,
    output: Path,
    title: str,
) -> Path:
    data, labels = _load_balanced_labels(scope, label_field)
    group_order = TYPE_ORDER if scope == "type" else PHASE_ORDER
    group_labels = TYPE_LABELS if scope == "type" else PHASE_LABELS
    pivot = data.pivot(index="label", columns="group", values="proportion").reindex(labels).fillna(0.0)
    groups = [group for group in group_order if group in pivot.columns]
    positions = np.arange(len(labels), dtype=float)
    bar_height = min(0.34, 0.8 / max(len(groups), 1))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * bar_height
    limit = max(1.0, float(pivot[groups].to_numpy().max()) + 0.12)

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.72)))
    for group, offset in zip(groups, offsets, strict=False):
        bars = ax.barh(
            positions + offset,
            pivot[group].to_numpy(dtype=float),
            height=bar_height,
            label=group_labels.get(group, group),
        )
        _add_horizontal_labels(ax, bars, limit)
    ax.set_yticks(positions)
    ax.set_yticklabels([wrap_japanese_label(label) for label in labels])
    ax.set_xlim(0, limit)
    ax.set_xlabel("文書均等化比率")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    longest = max((len(label) for label in labels), default=0)
    fig.subplots_adjust(left=min(0.36, max(0.22, 0.18 + longest * 0.008)))
    return _save_figure(fig, output)


def _plot_balanced_vertical(
    scope: str,
    label_field: str,
    output: Path,
    title: str,
) -> Path:
    data, labels = _load_balanced_labels(scope, label_field)
    group_order = TYPE_ORDER if scope == "type" else PHASE_ORDER
    group_labels = TYPE_LABELS if scope == "type" else PHASE_LABELS
    pivot = data.pivot(index="label", columns="group", values="proportion").reindex(labels).fillna(0.0)
    groups = [group for group in group_order if group in pivot.columns]
    positions = np.arange(len(labels), dtype=float)
    width = min(0.36, 0.8 / max(len(groups), 1))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width
    limit = max(1.0, float(pivot[groups].to_numpy().max()) + 0.12)

    fig, ax = plt.subplots(figsize=(10, 6))
    for group, offset in zip(groups, offsets, strict=False):
        bars = ax.bar(
            positions + offset,
            pivot[group].to_numpy(dtype=float),
            width=width,
            label=group_labels.get(group, group),
        )
        _add_vertical_labels(ax, bars, limit)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylim(0, limit)
    ax.set_ylabel("文書均等化比率")
    ax.set_title(title)
    ax.legend()
    return _save_figure(fig, output)


def plot_llm_communication_function() -> Path:
    return _plot_balanced_horizontal(
        "type",
        "communication_function",
        PRESENTATION_DIR / "communication_function_by_type.png",
        "生成AI分類：コミュニケーション機能",
    )


def plot_llm_certainty() -> Path:
    return _plot_balanced_vertical(
        "type",
        "certainty",
        PRESENTATION_DIR / "certainty_by_type.png",
        "生成AI分類：確実性",
    )


def _run_optional(name: str, function: Callable[[], Path], outputs: list[Path]) -> None:
    try:
        output = function()
    except FileNotFoundError as exc:
        print(f"WARNING: skipped {name}; missing optional input: {exc.filename or exc}")
    except VisualizationError as exc:
        print(f"WARNING: skipped {name}; {exc}")
    else:
        outputs.append(output)
        print(f"Created: {output.relative_to(PROJECT_ROOT)}")


def main() -> int:
    configure_japanese_font()
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    tasks: list[tuple[str, Callable[[], Path]]] = [
        ("noun-only TF-IDF contrast", plot_tfidf_noun_contrast),
        ("original TF-IDF contrast", plot_tfidf_original_contrast),
        ("embedding cluster distribution", plot_embedding_cluster_distribution),
        ("embedding PCA", plot_embedding_pca),
        ("LLM communication function by type", plot_llm_communication_function),
        ("LLM certainty by type", plot_llm_certainty),
        (
            "LLM policy direction by type",
            lambda: _plot_balanced_vertical(
                "type",
                "policy_direction",
                SUPPLEMENTARY_DIR / "policy_direction_by_type.png",
                "生成AI分類：政策方向（文書種別）",
            ),
        ),
        (
            "LLM policy direction by phase",
            lambda: _plot_balanced_vertical(
                "phase",
                "policy_direction",
                SUPPLEMENTARY_DIR / "policy_direction_by_phase.png",
                "生成AI分類：政策方向（時期）",
            ),
        ),
        (
            "LLM communication function by phase",
            lambda: _plot_balanced_horizontal(
                "phase",
                "communication_function",
                SUPPLEMENTARY_DIR / "communication_function_by_phase.png",
                "生成AI分類：コミュニケーション機能（時期）",
            ),
        ),
        (
            "LLM certainty by phase",
            lambda: _plot_balanced_vertical(
                "phase",
                "certainty",
                SUPPLEMENTARY_DIR / "certainty_by_phase.png",
                "生成AI分類：確実性（時期）",
            ),
        ),
    ]
    for name, function in tasks:
        _run_optional(name, function, outputs)

    if not outputs:
        print("ERROR: no figures could be generated", file=sys.stderr)
        return 1
    print(f"Generated {len(outputs)} figure(s) from packaged result CSV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
