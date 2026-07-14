"""Run sentence embedding and KMeans clustering analysis for BOJ chunks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "chunks.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "embedding"
FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K_VALUES = (3, 4, 5)
RANDOM_STATE = 42
N_INIT = 20
BATCH_SIZE = 16

REQUIRED_COLUMNS = [
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
]
EMBEDDING_METADATA_COLUMNS = [
    "chunk_id",
    "doc_id",
    "date",
    "period",
    "type",
    "speaker",
    "title",
    "section_title",
    "char_count",
]
ASSIGNMENT_COLUMNS = [
    "chunk_id",
    "doc_id",
    "type",
    "period",
    "speaker",
    "title",
    "cluster_k3",
    "cluster_k4",
    "cluster_k5",
]
REPRESENTATIVE_COLUMNS = [
    "k",
    "cluster",
    "rank",
    "chunk_id",
    "doc_id",
    "type",
    "period",
    "speaker",
    "title",
    "distance_to_centroid",
    "text",
]


class EmbeddingStageError(Exception):
    """Raised when embedding or clustering analysis cannot continue safely."""


@dataclass(frozen=True)
class ClusteringResult:
    k: int
    labels: np.ndarray
    centers: np.ndarray
    silhouette: float


@dataclass(frozen=True)
class OutputPaths:
    embeddings: Path
    assignments: Path
    metrics: Path
    representatives: Path
    type_distribution: Path
    period_distribution: Path
    figures: list[Path]


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise EmbeddingStageError(f"missing chunks CSV: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise EmbeddingStageError(f"cannot read chunks CSV {path}: {exc}") from exc


def validate_chunks(chunks: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in chunks.columns]
    if missing:
        raise EmbeddingStageError("chunks CSV is missing column(s): " + ", ".join(missing))

    if chunks["chunk_id"].duplicated().any():
        duplicates = chunks.loc[chunks["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise EmbeddingStageError(f"duplicate chunk_id values found: {duplicates}")

    if chunks["text"].fillna("").astype(str).str.strip().eq("").any():
        raise EmbeddingStageError("one or more chunks have empty text")

    if chunks.groupby("doc_id").size().empty:
        raise EmbeddingStageError("no document has chunks")


def print_validation_summary(chunks: pd.DataFrame) -> None:
    print("Validation summary:")
    print(f"total number of chunks: {len(chunks)}")
    print("number of chunks per document:")
    print(chunks.groupby("doc_id").size().to_string())
    print("number of chunks by type:")
    print(chunks.groupby("type").size().to_string())
    print("number of chunks by period:")
    print(chunks.groupby("period").size().to_string())
    print()


def make_embeddings(chunks: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = chunks["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return normalize_embeddings(np.asarray(embeddings, dtype=np.float32))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingStageError("one or more embedding vectors have zero norm")
    return embeddings / norms


def run_clustering(embeddings: np.ndarray) -> list[ClusteringResult]:
    results: list[ClusteringResult] = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        results.append(
            ClusteringResult(
                k=k,
                labels=labels,
                centers=model.cluster_centers_,
                silhouette=float(score),
            )
        )
    return results


def make_embeddings_table(chunks: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    embedding_columns = [f"embedding_{index:03d}" for index in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
    return pd.concat(
        [chunks[EMBEDDING_METADATA_COLUMNS].reset_index(drop=True), embedding_df],
        axis=1,
    )


def make_assignments_table(chunks: pd.DataFrame, results: list[ClusteringResult]) -> pd.DataFrame:
    assignments = chunks[["chunk_id", "doc_id", "type", "period", "speaker", "title"]].copy()
    for result in results:
        assignments[f"cluster_k{result.k}"] = result.labels
    return assignments[ASSIGNMENT_COLUMNS]


def make_metrics_table(results: list[ClusteringResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        sizes = pd.Series(result.labels).value_counts().sort_index()
        rows.append(
            {
                "k": result.k,
                "silhouette_score": result.silhouette,
                "minimum_cluster_size": int(sizes.min()),
                "maximum_cluster_size": int(sizes.max()),
            }
        )
    return pd.DataFrame(rows)


def make_representatives_table(
    chunks: pd.DataFrame, embeddings: np.ndarray, results: list[ClusteringResult]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        for cluster in range(result.k):
            indices = np.where(result.labels == cluster)[0]
            center = result.centers[cluster]
            distances = np.linalg.norm(embeddings[indices] - center, axis=1)
            order = np.argsort(distances)[:5]
            for rank, local_index in enumerate(order, start=1):
                row_index = int(indices[local_index])
                chunk = chunks.iloc[row_index]
                rows.append(
                    {
                        "k": result.k,
                        "cluster": cluster,
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "type": chunk["type"],
                        "period": chunk["period"],
                        "speaker": chunk["speaker"],
                        "title": chunk["title"],
                        "distance_to_centroid": float(distances[local_index]),
                        "text": chunk["text"],
                    }
                )
    return pd.DataFrame(rows, columns=REPRESENTATIVE_COLUMNS)


def make_distribution_tables(
    chunks: pd.DataFrame, results: list[ClusteringResult]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    type_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    for result in results:
        labeled = chunks.copy()
        labeled["cluster"] = result.labels
        for cluster, cluster_df in labeled.groupby("cluster"):
            cluster_size = len(cluster_df)
            for doc_type, count in cluster_df["type"].value_counts().sort_index().items():
                type_rows.append(
                    {
                        "k": result.k,
                        "cluster": int(cluster),
                        "type": doc_type,
                        "chunk_count": int(count),
                        "proportion_within_cluster": count / cluster_size,
                    }
                )
            for period, count in cluster_df["period"].value_counts().sort_index().items():
                period_rows.append(
                    {
                        "k": result.k,
                        "cluster": int(cluster),
                        "period": period,
                        "chunk_count": int(count),
                        "proportion_within_cluster": count / cluster_size,
                    }
                )

    return pd.DataFrame(type_rows), pd.DataFrame(period_rows)


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


def make_projection(embeddings: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)


def plot_clusters(projection: np.ndarray, labels: np.ndarray, k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        ax.scatter(projection[mask, 0], projection[mask, 1], label=f"cluster {cluster}")
    ax.set_title(f"Embedding clusters k={k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_group_projection(
    projection: np.ndarray, chunks: pd.DataFrame, column: str, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for value in sorted(chunks[column].astype(str).unique()):
        mask = chunks[column].astype(str).to_numpy() == value
        ax.scatter(projection[mask, 0], projection[mask, 1], label=value)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(
    chunks: pd.DataFrame, embeddings: np.ndarray, results: list[ClusteringResult]
) -> OutputPaths:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_japanese_font()

    embeddings_path = RESULTS_DIR / "chunk_embeddings.csv"
    assignments_path = RESULTS_DIR / "clustering_assignments.csv"
    metrics_path = RESULTS_DIR / "clustering_metrics.csv"
    representatives_path = RESULTS_DIR / "cluster_representative_chunks.csv"
    type_distribution_path = RESULTS_DIR / "cluster_distribution_by_type.csv"
    period_distribution_path = RESULTS_DIR / "cluster_distribution_by_period.csv"

    embeddings_table = make_embeddings_table(chunks, embeddings)
    assignments_table = make_assignments_table(chunks, results)
    metrics_table = make_metrics_table(results)
    representatives_table = make_representatives_table(chunks, embeddings, results)
    type_distribution, period_distribution = make_distribution_tables(chunks, results)

    embeddings_table.to_csv(embeddings_path, index=False, encoding="utf-8-sig")
    assignments_table.to_csv(assignments_path, index=False, encoding="utf-8-sig")
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    representatives_table.to_csv(representatives_path, index=False, encoding="utf-8-sig")
    type_distribution.to_csv(type_distribution_path, index=False, encoding="utf-8-sig")
    period_distribution.to_csv(period_distribution_path, index=False, encoding="utf-8-sig")

    projection = make_projection(embeddings)
    figures = [
        FIGURES_DIR / "embedding_clusters_k3.png",
        FIGURES_DIR / "embedding_clusters_k4.png",
        FIGURES_DIR / "embedding_clusters_k5.png",
        FIGURES_DIR / "embedding_by_type.png",
        FIGURES_DIR / "embedding_by_period.png",
    ]
    for result, path in zip(results, figures[:3], strict=True):
        plot_clusters(projection, result.labels, result.k, path)
    plot_group_projection(projection, chunks, "type", figures[3], "Embedding projection by document type")
    plot_group_projection(projection, chunks, "period", figures[4], "Embedding projection by period")

    return OutputPaths(
        embeddings=embeddings_path,
        assignments=assignments_path,
        metrics=metrics_path,
        representatives=representatives_path,
        type_distribution=type_distribution_path,
        period_distribution=period_distribution_path,
        figures=figures,
    )


def print_clustering_summary(
    results: list[ClusteringResult], embeddings: np.ndarray, chunks: pd.DataFrame
) -> None:
    print(f"embedding dimension: {embeddings.shape[1]}")
    for result in results:
        labeled = chunks.copy()
        labeled["cluster"] = result.labels
        sizes = pd.Series(result.labels).value_counts().sort_index()
        print(f"k = {result.k}")
        print(f"silhouette score: {result.silhouette:.6f}")
        print("cluster sizes:")
        print(sizes.to_string())
        print("document-type distribution:")
        print(pd.crosstab(labeled["cluster"], labeled["type"]).to_string())
        print("period distribution:")
        print(pd.crosstab(labeled["cluster"], labeled["period"]).to_string())
        print()


def print_representatives(representatives_path: Path) -> None:
    representatives = pd.read_csv(representatives_path, encoding="utf-8-sig")
    print("Representative chunks for manual interpretation:")
    for k in K_VALUES:
        print(f"k = {k}")
        k_rows = representatives[representatives["k"] == k]
        for cluster in sorted(k_rows["cluster"].unique()):
            print(f"cluster {cluster}")
            cluster_rows = k_rows[k_rows["cluster"] == cluster].sort_values("rank")
            for _, row in cluster_rows.iterrows():
                print(
                    f"rank {int(row['rank'])} | {row['chunk_id']} | "
                    f"{row['type']} | {row['period']} | "
                    f"distance={row['distance_to_centroid']:.6f}"
                )
                print(row["text"])
                print()


def print_generated_paths(paths: OutputPaths) -> None:
    print("Generated files:")
    for path in [
        paths.embeddings,
        paths.assignments,
        paths.metrics,
        paths.representatives,
        paths.type_distribution,
        paths.period_distribution,
        *paths.figures,
    ]:
        print(path)


def main() -> int:
    try:
        chunks = load_chunks(CHUNKS_PATH)
        validate_chunks(chunks)
        print_validation_summary(chunks)

        embeddings = make_embeddings(chunks)
        results = run_clustering(embeddings)
        print_clustering_summary(results, embeddings, chunks)

        paths = write_outputs(chunks, embeddings, results)
        print_representatives(paths.representatives)
        print_generated_paths(paths)
    except EmbeddingStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
