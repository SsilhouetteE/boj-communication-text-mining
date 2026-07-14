"""Run shared-vocabulary TF-IDF analysis on BOJ text chunks.

Tokenization:
    SudachiPy with sudachidict_core, SplitMode.C. Tokens use normalized forms
    where available and retain only nouns, verbs, and adjectives. Punctuation,
    symbols, numbers, URLs, isolated Latin letters, and one-character Japanese
    tokens are removed unless explicitly preserved.

Stop words:
    Loaded from metadata/stopwords_ja.txt. The list is intentionally small and
    excludes substantive monetary-policy vocabulary.

TF-IDF:
    A single TfidfVectorizer is fit over all chunks with unigrams only,
    min_df=2, max_df=0.85, sublinear_tf=True, and L2 normalization.

Aggregation:
    Chunk vectors are averaged within each doc_id, then document vectors are
    averaged within comparison groups so each document receives equal weight.
    A chunk-weighted group average is also reported as a supplementary score.

Period mapping:
    The original period column is preserved. For phase comparison only,
    before/pre_end_nirp map to before; transition/after/post/end_nirp/
    post_end_nirp map to post.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
from sudachipy import dictionary, tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "chunks.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "documents.csv"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "metadata" / "stopwords_ja.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "tfidf"
FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary"

REQUIRED_CHUNK_COLUMNS = [
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
REQUIRED_METADATA_COLUMNS = ["doc_id", "date", "period", "type", "speaker", "title"]

STOPWORDS = [
    "こと",
    "もの",
    "ため",
    "よう",
    "これ",
    "それ",
    "ところ",
    "場合",
    "日本銀行",
    "本日",
    "以上",
]
PRESERVE_TERMS = {
    "当面",
    "慎重",
    "不確実性",
    "緩和的",
    "見極める",
    "金利",
    "賃金",
    "物価",
    "政策",
    "正常化",
}
EXPECTED_TYPES = {"statement", "speech"}
EXPECTED_PERIODS = {
    "before",
    "transition",
    "after",
    "pre_end_nirp",
    "end_nirp",
    "post_end_nirp",
}
PHASE_MAP = {
    "before": "before",
    "pre_end_nirp": "before",
    "transition": "post",
    "after": "post",
    "post": "post",
    "end_nirp": "post",
    "post_end_nirp": "post",
}
POS_TO_KEEP = {"名詞", "動詞", "形容詞"}
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
ASCII_LETTER_RE = re.compile(r"^[A-Za-z]$")
NUMBER_RE = re.compile(r"^[0-9０-９.,，．%％+-]+$")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
ONE_JAPANESE_RE = re.compile(r"^[\u3040-\u30ff\u3400-\u9fff]$")


class TfidfStageError(Exception):
    """Raised when TF-IDF analysis cannot continue safely."""


@dataclass(frozen=True)
class AnalysisOutputs:
    chunk_matrix: Path
    document_scores: Path
    top_terms_by_type: Path
    top_terms_by_phase: Path
    type_contrast: Path
    phase_contrast: Path
    figures: list[Path]


def ensure_stopwords_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text("\n".join(STOPWORDS) + "\n", encoding="utf-8")


def load_stopwords(path: Path) -> set[str]:
    ensure_stopwords_file(path)
    try:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as exc:
        raise TfidfStageError(f"cannot read stop-word file {path}: {exc}") from exc


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise TfidfStageError(f"missing chunks CSV: {path}")
    try:
        chunks = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise TfidfStageError(f"cannot read chunks CSV {path}: {exc}") from exc
    return chunks


def validate_metadata_file(path: Path) -> None:
    if not path.exists():
        raise TfidfStageError(f"missing metadata file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
    except OSError as exc:
        raise TfidfStageError(f"cannot read metadata file {path}: {exc}") from exc

    missing = [field for field in REQUIRED_METADATA_COLUMNS if field not in fields]
    if missing:
        raise TfidfStageError(
            "metadata file is missing required field(s): " + ", ".join(missing)
        )


def validate_chunks(chunks: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CHUNK_COLUMNS if column not in chunks.columns]
    if missing:
        raise TfidfStageError("chunks CSV is missing column(s): " + ", ".join(missing))

    if chunks["chunk_id"].duplicated().any():
        duplicates = chunks.loc[chunks["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise TfidfStageError(f"duplicate chunk_id values found: {duplicates}")

    if chunks["text"].fillna("").astype(str).str.strip().eq("").any():
        raise TfidfStageError("one or more chunks have empty text")

    types = set(chunks["type"].dropna().astype(str))
    unexpected_types = sorted(types - EXPECTED_TYPES)
    if unexpected_types:
        raise TfidfStageError(f"unexpected type value(s): {unexpected_types}")

    periods = set(chunks["period"].dropna().astype(str))
    unexpected_periods = sorted(periods - EXPECTED_PERIODS)
    if unexpected_periods:
        raise TfidfStageError(f"unexpected period value(s): {unexpected_periods}")

    if chunks.groupby("doc_id").size().empty:
        raise TfidfStageError("no document has chunks")


def print_preanalysis_validation(chunks: pd.DataFrame) -> None:
    print("Validation summary:")
    print(f"total number of chunks: {len(chunks)}")
    print("number of chunks per document:")
    print(chunks.groupby("doc_id").size().to_string())
    print("number of chunks by type:")
    print(chunks.groupby("type").size().to_string())
    print("number of chunks by period:")
    print(chunks.groupby("period").size().to_string())
    print("median character count by document:")
    print(chunks.groupby("doc_id")["char_count"].median().to_string())
    print()


def map_period_to_phase(period: str) -> str:
    phase = PHASE_MAP.get(period)
    if not phase:
        raise TfidfStageError(f"cannot map period to phase: {period}")
    return phase


def make_tokenizer(stopwords: set[str]) -> Callable[[str], list[str]]:
    sudachi = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.C

    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        for morpheme in sudachi.tokenize(text, mode):
            if morpheme.part_of_speech()[0] not in POS_TO_KEEP:
                continue

            token = normalized_token(morpheme)
            if should_keep_token(token, stopwords):
                tokens.append(token)
        return tokens

    return tokenize


def normalized_token(morpheme: object) -> str:
    normalized = getattr(morpheme, "normalized_form", lambda: "")()
    dictionary_form = getattr(morpheme, "dictionary_form", lambda: "")()
    surface = getattr(morpheme, "surface", lambda: "")()
    return str(normalized or dictionary_form or surface).strip()


def should_keep_token(token: str, stopwords: set[str]) -> bool:
    if not token or token in stopwords:
        return False
    if token in PRESERVE_TERMS:
        return True
    if URL_RE.search(token):
        return False
    if NUMBER_RE.fullmatch(token):
        return False
    if ASCII_LETTER_RE.fullmatch(token):
        return False
    if ONE_JAPANESE_RE.fullmatch(token):
        return False
    if not JAPANESE_RE.search(token) and len(token) <= 1:
        return False
    return True


def fit_tfidf(chunks: pd.DataFrame, stopwords: set[str]) -> tuple[pd.DataFrame, list[str]]:
    vectorizer = TfidfVectorizer(
        tokenizer=make_tokenizer(stopwords),
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
        ngram_range=(1, 1),
    )
    matrix = vectorizer.fit_transform(chunks["text"].astype(str).tolist())
    terms = list(vectorizer.get_feature_names_out())
    scores = pd.DataFrame(matrix.toarray(), columns=terms)
    return scores, terms


def make_score_tables(
    chunks: pd.DataFrame, scores: pd.DataFrame, terms: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_meta = chunks[REQUIRED_CHUNK_COLUMNS[:-1]].reset_index(drop=True)
    chunk_matrix = pd.concat([chunk_meta, scores.reset_index(drop=True)], axis=1)

    doc_meta = (
        chunks[["doc_id", "date", "period", "type", "speaker", "title"]]
        .drop_duplicates("doc_id")
        .sort_values("doc_id")
        .reset_index(drop=True)
    )
    doc_scores_only = chunk_matrix.groupby("doc_id", sort=True)[terms].mean().reset_index()
    doc_scores = doc_meta.merge(doc_scores_only, on="doc_id", how="left")
    doc_scores.insert(6, "chunk_count", chunks.groupby("doc_id").size().reindex(doc_scores["doc_id"]).to_numpy())
    return chunk_matrix, doc_scores


def top_terms_by_group(
    chunk_matrix: pd.DataFrame,
    doc_scores: pd.DataFrame,
    terms: list[str],
    group_column: str,
    groups: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    doc_group_scores = doc_scores.groupby(group_column)[terms].mean()
    chunk_group_scores = chunk_matrix.groupby(group_column)[terms].mean()

    for group in groups:
        doc_scores_for_group = doc_group_scores.loc[group].sort_values(ascending=False)
        chunk_scores_for_group = chunk_group_scores.loc[group]
        for rank, (term, score) in enumerate(doc_scores_for_group.head(30).items(), start=1):
            rows.append(
                {
                    "group": group,
                    "rank": rank,
                    "term": term,
                    "document_balanced_score": score,
                    "chunk_weighted_score": chunk_scores_for_group[term],
                }
            )

    return pd.DataFrame(rows)


def contrast_table(
    group_scores: pd.DataFrame,
    left_group: str,
    right_group: str,
    left_column: str,
    right_column: str,
    difference_direction: str,
) -> pd.DataFrame:
    left_scores = group_scores.loc[left_group]
    right_scores = group_scores.loc[right_group]
    if difference_direction == "left_minus_right":
        difference = left_scores - right_scores
    elif difference_direction == "right_minus_left":
        difference = right_scores - left_scores
    else:
        raise TfidfStageError(f"unknown contrast direction: {difference_direction}")

    rows = []
    for term, diff in difference.items():
        favored = right_group if diff > 0 and difference_direction == "right_minus_left" else left_group
        if diff < 0:
            favored = left_group if difference_direction == "right_minus_left" else right_group
        rows.append(
            {
                "term": term,
                left_column: left_scores[term],
                right_column: right_scores[term],
                "difference": diff,
                "favored_group": favored,
            }
        )
    table = pd.DataFrame(rows)
    return table.reindex(table["difference"].abs().sort_values(ascending=False).index).reset_index(drop=True)


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


def plot_top_terms(top_terms: pd.DataFrame, group: str, path: Path, title: str) -> None:
    subset = top_terms[top_terms["group"] == group].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["term"], subset["document_balanced_score"])
    ax.set_title(title)
    ax.set_xlabel("Document-balanced TF-IDF score")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_contrast(
    contrast: pd.DataFrame,
    negative_group: str,
    positive_group: str,
    path: Path,
    title: str,
) -> None:
    negative = contrast[contrast["favored_group"] == negative_group].tail(0)
    negative = contrast[contrast["favored_group"] == negative_group].copy()
    positive = contrast[contrast["favored_group"] == positive_group].copy()
    negative = negative.nsmallest(10, "difference")
    positive = positive.nlargest(10, "difference")
    subset = pd.concat([negative, positive]).sort_values("difference")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(subset["term"], subset["difference"])
    ax.set_title(title)
    ax.set_xlabel("TF-IDF score difference")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(
    chunks: pd.DataFrame,
    scores: pd.DataFrame,
    terms: list[str],
    chunk_matrix: pd.DataFrame,
    doc_scores: pd.DataFrame,
) -> AnalysisOutputs:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_japanese_font()

    chunk_matrix_path = RESULTS_DIR / "tfidf_chunk_matrix.csv"
    document_scores_path = RESULTS_DIR / "tfidf_document_scores.csv"
    top_terms_by_type_path = RESULTS_DIR / "tfidf_top_terms_by_type.csv"
    top_terms_by_phase_path = RESULTS_DIR / "tfidf_top_terms_by_phase.csv"
    type_contrast_path = RESULTS_DIR / "tfidf_type_contrast.csv"
    phase_contrast_path = RESULTS_DIR / "tfidf_phase_contrast.csv"

    chunk_matrix = chunk_matrix.copy()
    doc_scores = doc_scores.copy()
    chunk_matrix["phase_group"] = chunk_matrix["period"].map(map_period_to_phase)
    doc_scores["phase_group"] = doc_scores["period"].map(map_period_to_phase)

    top_type = top_terms_by_group(chunk_matrix, doc_scores, terms, "type", ["statement", "speech"])
    top_phase = top_terms_by_group(chunk_matrix, doc_scores, terms, "phase_group", ["before", "post"])

    type_group_scores = doc_scores.groupby("type")[terms].mean()
    phase_group_scores = doc_scores.groupby("phase_group")[terms].mean()
    type_contrast = contrast_table(
        type_group_scores,
        "statement",
        "speech",
        "statement_score",
        "speech_score",
        "left_minus_right",
    )
    phase_contrast = contrast_table(
        phase_group_scores,
        "before",
        "post",
        "before_score",
        "post_score",
        "right_minus_left",
    )

    chunk_matrix.drop(columns=["phase_group"]).to_csv(chunk_matrix_path, index=False, encoding="utf-8-sig")
    doc_scores.drop(columns=["phase_group"]).to_csv(document_scores_path, index=False, encoding="utf-8-sig")
    top_type.to_csv(top_terms_by_type_path, index=False, encoding="utf-8-sig")
    top_phase.to_csv(top_terms_by_phase_path, index=False, encoding="utf-8-sig")
    type_contrast.to_csv(type_contrast_path, index=False, encoding="utf-8-sig")
    phase_contrast.to_csv(phase_contrast_path, index=False, encoding="utf-8-sig")

    figures = [
        FIGURES_DIR / "tfidf_statement_top_terms.png",
        FIGURES_DIR / "tfidf_speech_top_terms.png",
        FIGURES_DIR / "tfidf_before_top_terms.png",
        FIGURES_DIR / "tfidf_post_top_terms.png",
        FIGURES_DIR / "tfidf_type_contrast.png",
        FIGURES_DIR / "tfidf_phase_contrast.png",
    ]
    plot_top_terms(top_type, "statement", figures[0], "Statement top TF-IDF terms")
    plot_top_terms(top_type, "speech", figures[1], "Speech top TF-IDF terms")
    plot_top_terms(top_phase, "before", figures[2], "Before top TF-IDF terms")
    plot_top_terms(top_phase, "post", figures[3], "Post top TF-IDF terms")
    plot_contrast(type_contrast, "speech", "statement", figures[4], "Statement vs. speech TF-IDF contrast")
    plot_contrast(phase_contrast, "before", "post", figures[5], "Post vs. before TF-IDF contrast")

    return AnalysisOutputs(
        chunk_matrix=chunk_matrix_path,
        document_scores=document_scores_path,
        top_terms_by_type=top_terms_by_type_path,
        top_terms_by_phase=top_terms_by_phase_path,
        type_contrast=type_contrast_path,
        phase_contrast=phase_contrast_path,
        figures=figures,
    )


def print_top_terms(label: str, top_terms: pd.DataFrame, group: str) -> None:
    print(f"Top 20 terms for {label}:")
    subset = top_terms[top_terms["group"] == group].head(20)
    for _, row in subset.iterrows():
        print(f"{int(row['rank']):2d}. {row['term']} ({row['document_balanced_score']:.6f})")
    print()


def print_contrasts(label: str, contrast: pd.DataFrame) -> None:
    print(f"Strongest {label} contrasts:")
    for _, row in contrast.head(20).iterrows():
        print(
            f"{row['term']}: difference={row['difference']:.6f}, "
            f"favored={row['favored_group']}"
        )
    print()


def print_results_summary(outputs: AnalysisOutputs, terms: list[str]) -> None:
    top_type = pd.read_csv(outputs.top_terms_by_type, encoding="utf-8-sig")
    top_phase = pd.read_csv(outputs.top_terms_by_phase, encoding="utf-8-sig")
    type_contrast = pd.read_csv(outputs.type_contrast, encoding="utf-8-sig")
    phase_contrast = pd.read_csv(outputs.phase_contrast, encoding="utf-8-sig")

    print(f"final vocabulary size: {len(terms)}")
    print()
    print_top_terms("statement", top_type, "statement")
    print_top_terms("speech", top_type, "speech")
    print_top_terms("before", top_phase, "before")
    print_top_terms("post", top_phase, "post")
    print_contrasts("type", type_contrast)
    print_contrasts("phase", phase_contrast)

    print("Generated files:")
    for path in [
        STOPWORDS_PATH,
        outputs.chunk_matrix,
        outputs.document_scores,
        outputs.top_terms_by_type,
        outputs.top_terms_by_phase,
        outputs.type_contrast,
        outputs.phase_contrast,
        *outputs.figures,
    ]:
        print(path)


def main() -> int:
    try:
        validate_metadata_file(METADATA_PATH)
        chunks = load_chunks(CHUNKS_PATH)
        validate_chunks(chunks)
        print_preanalysis_validation(chunks)

        stopwords = load_stopwords(STOPWORDS_PATH)
        scores, terms = fit_tfidf(chunks, stopwords)
        if not terms:
            raise TfidfStageError("TF-IDF vocabulary is empty")

        chunk_matrix, doc_scores = make_score_tables(chunks, scores, terms)
        outputs = write_outputs(chunks, scores, terms, chunk_matrix, doc_scores)
        print_results_summary(outputs, terms)
    except TfidfStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
