"""Run noun-only supplementary TF-IDF analysis for BOJ text chunks.

This is a presentation-oriented auxiliary analysis: 名詞に限定した補助分析.
It keeps only Sudachi major POS 名詞 using SplitMode.C and excludes transparent
non-substantive noun placeholders such as 形式名詞 and 数詞. The rule is
deliberately conservative: it does not manually join adjacent nouns, so compound
nouns are preserved only when Sudachi returns them as one token.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
from sudachipy import dictionary, tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "chunks.csv"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "metadata" / "stopwords_ja_nouns.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "tfidf_nouns"
PRESENTATION_FIGURES_DIR = PROJECT_ROOT / "figures" / "presentation"
NOTE_PATH = RESULTS_DIR / "tfidf_noun_analysis_note.md"

CHUNK_MATRIX_PATH = RESULTS_DIR / "tfidf_noun_chunk_matrix.csv"
DOCUMENT_SCORES_PATH = RESULTS_DIR / "tfidf_noun_document_scores.csv"
TOP_TERMS_BY_TYPE_PATH = RESULTS_DIR / "tfidf_noun_top_terms_by_type.csv"
TYPE_CONTRAST_PATH = RESULTS_DIR / "tfidf_noun_type_contrast.csv"
PRESENTATION_TERMS_PATH = RESULTS_DIR / "presentation_tfidf_noun_terms.csv"
PRESENTATION_FIGURE_PATH = PRESENTATION_FIGURES_DIR / "tfidf_statement_speech_contrast_nouns.png"

REQUIRED_COLUMNS = (
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
EXPECTED_TYPES = {"statement", "speech"}
INITIAL_NOUN_STOPWORDS = (
    "こと",
    "もの",
    "ため",
    "ところ",
    "場合",
    "本日",
    "以上",
    "以下",
    "図表",
    "日本銀行",
)
PROTECTED_POLICY_NOUNS = {
    "金利",
    "政策",
    "金融",
    "国債",
    "買い入れ",
    "オペ",
    "物価",
    "賃金",
    "企業",
    "経済",
    "展望",
    "基調",
    "上昇率",
    "緩和",
    "正常化",
    "指し値",
    "方針",
    "決定",
    "調節",
}
NON_SUBSTANTIVE_NOUN_MARKERS = {"形式名詞", "数詞"}
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
ISOLATED_LATIN_RE = re.compile(r"^[A-Za-zＡ-Ｚａ-ｚ]$")
NUMBER_RE = re.compile(r"^[0-9０-９.,，．%％+-]+$")
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
NON_EMPTY_RE = re.compile(r"\S")


class NounTfidfError(Exception):
    """Raised when noun-only TF-IDF analysis cannot continue safely."""


@dataclass(frozen=True)
class AnalysisOutputs:
    chunk_matrix: Path
    document_scores: Path
    top_terms_by_type: Path
    type_contrast: Path
    presentation_terms: Path
    presentation_figure: Path
    stopwords: Path
    note: Path

    def generated_paths(self) -> list[Path]:
        return [
            self.chunk_matrix,
            self.document_scores,
            self.top_terms_by_type,
            self.type_contrast,
            self.presentation_terms,
            self.presentation_figure,
            self.stopwords,
            self.note,
        ]


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise NounTfidfError(f"missing chunks CSV: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as exc:
        raise NounTfidfError(f"cannot read chunks CSV {path}: {exc}") from exc


def validate_chunks(chunks: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in chunks.columns]
    if missing:
        raise NounTfidfError("chunks CSV is missing column(s): " + ", ".join(missing))

    if chunks["chunk_id"].duplicated().any():
        duplicates = chunks.loc[chunks["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise NounTfidfError(f"duplicate chunk_id values found: {duplicates}")

    if chunks["text"].astype(str).str.strip().eq("").any():
        raise NounTfidfError("one or more chunks have empty text")

    types = set(chunks["type"].astype(str))
    if not EXPECTED_TYPES.issubset(types):
        raise NounTfidfError("type column must contain both statement and speech")

    if chunks.groupby("doc_id").size().empty:
        raise NounTfidfError("no document has chunks")


def ensure_noun_stopwords_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(INITIAL_NOUN_STOPWORDS) + "\n", encoding="utf-8")


def load_noun_stopwords(path: Path) -> set[str]:
    ensure_noun_stopwords_file(path)
    try:
        stopwords = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as exc:
        raise NounTfidfError(f"cannot read noun stop-word file {path}: {exc}") from exc

    protected_overlap = sorted(stopwords & PROTECTED_POLICY_NOUNS)
    if protected_overlap:
        raise NounTfidfError(
            "noun stop-word file contains protected substantive policy noun(s): "
            + ", ".join(protected_overlap)
        )
    return stopwords


def normalized_token(morpheme: object) -> str:
    normalized = getattr(morpheme, "normalized_form", lambda: "")()
    dictionary_form = getattr(morpheme, "dictionary_form", lambda: "")()
    surface = getattr(morpheme, "surface", lambda: "")()
    return str(normalized or dictionary_form or surface).strip()


def is_substantive_noun(pos: tuple[str, ...]) -> bool:
    if not pos or pos[0] != "名詞":
        return False
    # Sudachi stores noun subtype information across later POS slots. Removing
    # these markers keeps ordinary, proper, verbal, and adjectival nouns while
    # excluding placeholder and numeric nouns.
    return not any(marker in pos for marker in NON_SUBSTANTIVE_NOUN_MARKERS)


def should_keep_token(token: str, stopwords: set[str]) -> bool:
    if not token or not NON_EMPTY_RE.search(token):
        return False
    if token in stopwords:
        return False
    if URL_RE.search(token):
        return False
    if NUMBER_RE.fullmatch(token):
        return False
    if ISOLATED_LATIN_RE.fullmatch(token):
        return False
    if SYMBOL_ONLY_RE.fullmatch(token):
        return False
    return True


def make_noun_tokenizer(stopwords: set[str]) -> Callable[[str], list[str]]:
    sudachi = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.C

    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        for morpheme in sudachi.tokenize(str(text), mode):
            pos = tuple(morpheme.part_of_speech())
            if not is_substantive_noun(pos):
                continue
            token = normalized_token(morpheme)
            if should_keep_token(token, stopwords):
                tokens.append(token)
        return tokens

    return tokenize


def count_noun_tokens(chunks: pd.DataFrame, noun_tokenizer: Callable[[str], list[str]]) -> int:
    return sum(len(noun_tokenizer(text)) for text in chunks["text"].astype(str))


def fit_tfidf(
    chunks: pd.DataFrame, noun_tokenizer: Callable[[str], list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    vectorizer = TfidfVectorizer(
        tokenizer=noun_tokenizer,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        norm="l2",
        ngram_range=(1, 1),
    )
    matrix = vectorizer.fit_transform(chunks["text"].astype(str).tolist())
    terms = list(vectorizer.get_feature_names_out())
    if not terms:
        raise NounTfidfError("noun-only TF-IDF vocabulary is empty after filtering")
    scores = pd.DataFrame(matrix.toarray(), columns=terms)
    return scores, terms


def make_score_tables(
    chunks: pd.DataFrame, scores: pd.DataFrame, terms: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_meta = chunks[list(REQUIRED_COLUMNS[:-1])].reset_index(drop=True)
    chunk_meta["char_count"] = pd.to_numeric(chunk_meta["char_count"], errors="coerce").fillna(0).astype(int)
    chunk_matrix = pd.concat([chunk_meta, scores.reset_index(drop=True)], axis=1)

    doc_meta = (
        chunks[["doc_id", "date", "period", "type", "speaker", "title"]]
        .drop_duplicates("doc_id")
        .sort_values("doc_id")
        .reset_index(drop=True)
    )
    doc_scores_only = chunk_matrix.groupby("doc_id", sort=True)[terms].mean().reset_index()
    doc_scores = doc_meta.merge(doc_scores_only, on="doc_id", how="left")
    chunk_counts = chunks.groupby("doc_id").size().reindex(doc_scores["doc_id"]).to_numpy()
    doc_scores.insert(6, "chunk_count", chunk_counts)
    return chunk_matrix, doc_scores


def top_terms_by_type(
    chunk_matrix: pd.DataFrame, doc_scores: pd.DataFrame, terms: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    document_balanced = doc_scores.groupby("type")[terms].mean()
    chunk_weighted = chunk_matrix.groupby("type")[terms].mean()

    for group in ["statement", "speech"]:
        scores = document_balanced.loc[group].sort_values(ascending=False)
        for rank, (term, score) in enumerate(scores.head(30).items(), start=1):
            rows.append(
                {
                    "group": group,
                    "rank": rank,
                    "term": term,
                    "document_balanced_score": float(score),
                    "chunk_weighted_score": float(chunk_weighted.loc[group, term]),
                }
            )
    return pd.DataFrame(rows)


def make_type_contrast(doc_scores: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    type_scores = doc_scores.groupby("type")[terms].mean()
    statement_scores = type_scores.loc["statement"]
    speech_scores = type_scores.loc["speech"]
    difference = statement_scores - speech_scores

    rows = []
    for term in terms:
        diff = float(difference[term])
        rows.append(
            {
                "term": term,
                "statement_score": float(statement_scores[term]),
                "speech_score": float(speech_scores[term]),
                "difference": diff,
                "favored_group": "statement" if diff >= 0 else "speech",
            }
        )
    contrast = pd.DataFrame(rows)
    return contrast.reindex(contrast["difference"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def select_presentation_terms(contrast: pd.DataFrame) -> pd.DataFrame:
    statement_terms = contrast[contrast["difference"] > 0].nlargest(8, "difference")
    speech_terms = contrast[contrast["difference"] < 0].nsmallest(8, "difference")
    if len(statement_terms) < 8 or len(speech_terms) < 8:
        raise NounTfidfError("not enough statement- or speech-favored nouns for the presentation figure")
    return (
        pd.concat([speech_terms, statement_terms], ignore_index=True)
        .sort_values("difference")
        .reset_index(drop=True)
    )


def validate_display_terms(
    presentation_terms: pd.DataFrame,
    stopwords: set[str],
    noun_tokenizer: Callable[[str], list[str]],
) -> None:
    invalid: list[str] = []
    for term in presentation_terms["term"].astype(str):
        if not should_keep_token(term, stopwords):
            invalid.append(term)
            continue
        tokenized = noun_tokenizer(term)
        if term not in tokenized and not tokenized:
            invalid.append(term)

    if invalid:
        raise NounTfidfError(
            "displayed figure terms include invalid noun-only tokens: " + ", ".join(sorted(set(invalid)))
        )


def configure_japanese_font() -> None:
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "Hiragino Sans",
        "MS Gothic",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            return
    print("WARNING: Japanese-compatible matplotlib font not found; figures may show missing glyphs.")


def add_value_labels(ax: plt.Axes, bars, axis_limit: float) -> None:
    offset = max(axis_limit * 0.012, 0.002)
    x_min, x_max = ax.get_xlim()
    for bar in bars:
        value = float(bar.get_width())
        if value >= 0:
            x = min(value + offset, x_max - offset)
            ha = "left" if x < x_max - offset * 1.1 else "right"
        else:
            x = max(value - offset, x_min + offset)
            ha = "right" if x > x_min + offset * 1.1 else "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha=ha, fontsize=9)


def plot_presentation_contrast(presentation_terms: pd.DataFrame, path: Path) -> None:
    labels = presentation_terms["term"].astype(str).tolist()
    values = presentation_terms["difference"].to_numpy(dtype=float)
    y_positions = np.arange(len(presentation_terms))
    axis_limit = float(np.max(np.abs(values))) * 1.35

    fig_height = max(5.5, len(presentation_terms) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(y_positions, values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("政策声明スコア − 講演スコア")
    ax.set_title("政策声明と講演を特徴づける名詞\n名詞に限定した補助分析")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.text(-axis_limit * 0.96, len(presentation_terms) - 0.4, "左側：講演に特徴的", ha="left", va="bottom")
    ax.text(axis_limit * 0.96, len(presentation_terms) - 0.4, "右側：政策声明に特徴的", ha="right", va="bottom")
    add_value_labels(ax, bars, axis_limit)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_analysis_note(
    path: Path, presentation_terms: pd.DataFrame, statement_terms: pd.DataFrame, speech_terms: pd.DataFrame
) -> None:
    statement_list = "\n".join(f"- {term}" for term in statement_terms["term"].astype(str))
    speech_list = "\n".join(f"- {term}" for term in speech_terms["term"].astype(str))
    displayed_list = "\n".join(
        f"- {row.term}: {row.favored_group} ({row.difference:.4f})"
        for row in presentation_terms.itertuples(index=False)
    )

    content = f"""# Noun-Only TF-IDF Supplementary Analysis

名詞に限定した補助分析

### Position of this analysis

- The original TF-IDF analysis keeps nouns, verbs, and adjectives.
- The noun-only version is a supplementary presentation analysis.
- The original result is preserved.
- Noun restriction improves conceptual readability but may discard information contained in verbs and evaluative expressions.

### Main comparison

Displayed statement-favored nouns:

{statement_list}

Displayed speech-favored nouns:

{speech_list}

Displayed terms with differences:

{displayed_list}

### Limitation

- Selecting nouns changes the feature space.
- The result depends on morphological analysis.
- Compound-noun segmentation may affect the ranking.
- This auxiliary result should be interpreted together with the original TF-IDF result.

No final conclusion choosing Narrative A or Narrative B is provided here.
"""
    path.write_text(content, encoding="utf-8")


def write_outputs(
    chunk_matrix: pd.DataFrame,
    doc_scores: pd.DataFrame,
    top_terms: pd.DataFrame,
    contrast: pd.DataFrame,
    presentation_terms: pd.DataFrame,
) -> AnalysisOutputs:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    configure_japanese_font()

    chunk_matrix.to_csv(CHUNK_MATRIX_PATH, index=False, encoding="utf-8-sig")
    doc_scores.to_csv(DOCUMENT_SCORES_PATH, index=False, encoding="utf-8-sig")
    top_terms.to_csv(TOP_TERMS_BY_TYPE_PATH, index=False, encoding="utf-8-sig")
    contrast.to_csv(TYPE_CONTRAST_PATH, index=False, encoding="utf-8-sig")
    presentation_terms.to_csv(PRESENTATION_TERMS_PATH, index=False, encoding="utf-8-sig")

    plot_presentation_contrast(presentation_terms, PRESENTATION_FIGURE_PATH)
    statement_display = presentation_terms[presentation_terms["favored_group"] == "statement"].sort_values(
        "difference", ascending=False
    )
    speech_display = presentation_terms[presentation_terms["favored_group"] == "speech"].sort_values(
        "difference"
    )
    write_analysis_note(NOTE_PATH, presentation_terms, statement_display, speech_display)

    return AnalysisOutputs(
        chunk_matrix=CHUNK_MATRIX_PATH,
        document_scores=DOCUMENT_SCORES_PATH,
        top_terms_by_type=TOP_TERMS_BY_TYPE_PATH,
        type_contrast=TYPE_CONTRAST_PATH,
        presentation_terms=PRESENTATION_TERMS_PATH,
        presentation_figure=PRESENTATION_FIGURE_PATH,
        stopwords=STOPWORDS_PATH,
        note=NOTE_PATH,
    )


def print_top_terms(label: str, top_terms: pd.DataFrame, group: str, count: int = 20) -> None:
    print(f"Top {count} nouns for {label}:")
    subset = top_terms[top_terms["group"] == group].head(count)
    for row in subset.itertuples(index=False):
        print(f"{int(row.rank):2d}. {row.term} ({row.document_balanced_score:.6f})")
    print()


def print_contrast_terms(title: str, contrast: pd.DataFrame) -> None:
    print(title)
    for row in contrast.itertuples(index=False):
        print(f"{row.term}: difference={row.difference:.6f}, favored={row.favored_group}")
    print()


def print_summary(
    chunks: pd.DataFrame,
    total_noun_tokens: int,
    terms: list[str],
    top_terms: pd.DataFrame,
    contrast: pd.DataFrame,
    presentation_terms: pd.DataFrame,
    outputs: AnalysisOutputs,
) -> None:
    print("Noun-only TF-IDF supplementary analysis: 名詞に限定した補助分析")
    print(f"total number of chunks: {len(chunks)}")
    print("number of chunks per document:")
    print(chunks.groupby("doc_id").size().to_string())
    print(f"total extracted noun tokens: {total_noun_tokens}")
    print(f"noun vocabulary size: {len(terms)}")
    print()

    print_top_terms("statement", top_terms, "statement")
    print_top_terms("speech", top_terms, "speech")
    print_contrast_terms(
        "Strongest statement-favored nouns:",
        contrast[contrast["favored_group"] == "statement"].nlargest(20, "difference"),
    )
    print_contrast_terms(
        "Strongest speech-favored nouns:",
        contrast[contrast["favored_group"] == "speech"].nsmallest(20, "difference"),
    )
    print("Displayed figure terms verified as noun-only tokens without punctuation, numeric-only tokens, or placeholder stop words.")
    print()
    print("Presentation figure terms:")
    print(presentation_terms.to_string(index=False))
    print()
    print("Generated files:")
    for path in outputs.generated_paths():
        print(path)


def run_analysis() -> tuple[pd.DataFrame, AnalysisOutputs]:
    chunks = load_chunks(CHUNKS_PATH)
    validate_chunks(chunks)
    stopwords = load_noun_stopwords(STOPWORDS_PATH)
    noun_tokenizer = make_noun_tokenizer(stopwords)
    total_noun_tokens = count_noun_tokens(chunks, noun_tokenizer)
    scores, terms = fit_tfidf(chunks, noun_tokenizer)
    chunk_matrix, doc_scores = make_score_tables(chunks, scores, terms)
    top_terms = top_terms_by_type(chunk_matrix, doc_scores, terms)
    contrast = make_type_contrast(doc_scores, terms)
    presentation_terms = select_presentation_terms(contrast)
    validate_display_terms(presentation_terms, stopwords, noun_tokenizer)
    outputs = write_outputs(chunk_matrix, doc_scores, top_terms, contrast, presentation_terms)
    print_summary(chunks, total_noun_tokens, terms, top_terms, contrast, presentation_terms, outputs)
    return presentation_terms, outputs


def main() -> int:
    try:
        run_analysis()
    except NounTfidfError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: file operation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
