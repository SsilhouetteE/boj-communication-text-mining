"""Prepare chunk batches and templates for manual generative-AI classification."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "chunks.csv"
BATCH_DIR = PROJECT_ROOT / "data" / "llm_batches"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "llm_classification_prompt.md"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
TEMPLATE_PATH = RESULTS_DIR / "llm_labels_template.csv"

BATCH_SIZE = 15
REQUIRED_COLUMNS = (
    "chunk_id",
    "doc_id",
    "type",
    "period",
    "speaker",
    "title",
    "text",
)
JSONL_COLUMNS = ("chunk_id", "doc_id", "type", "period", "text")
TEMPLATE_COLUMNS = (
    "chunk_id",
    "doc_id",
    "type",
    "period",
    "policy_direction",
    "communication_function",
    "certainty",
    "evidence",
)

PROMPT_TEXT = """# 日本銀行テキスト分類プロンプト

あなたは日本銀行の政策声明・講演テキストを分類する補助者です。
以下の各 chunk について、本文に含まれる情報だけを使って分類してください。

## 厳守事項

- テキストに明示されている内容だけを使用する。
- 話し手の意図を推測しない。
- 市場への影響を推測しない。
- 判断の根拠として、本文から短い引用を evidence に入れる。
- 出力は JSON のみとし、説明文や Markdown を追加しない。

## 分類ラベル

### policy_direction

次のいずれか 1 つを選ぶ。

- 引き締め的
- 中立的
- 緩和的
- 該当なし

### communication_function

次のいずれか 1 つを選ぶ。

- 政策行動
- 政策の根拠
- 経済見通し
- リスク・不確実性
- 期待形成
- 背景・その他

### certainty

次のいずれか 1 つを選ぶ。

- 明確
- 条件付き
- 慎重
- 該当なし

## 入力形式

各入力レコードには、chunk_id, doc_id, type, period, text が含まれる。

## 出力形式

JSON 配列のみを返す。各要素は次の形式にする。

[
  {
    "chunk_id": "D01_C001",
    "policy_direction": "中立的",
    "communication_function": "背景・その他",
    "certainty": "慎重",
    "evidence": "本文からの短い引用"
  }
]
"""


class LLMBatchPreparationError(Exception):
    """Raised when LLM batch preparation cannot continue safely."""


@dataclass(frozen=True)
class BatchSummary:
    path: Path
    chunk_count: int


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise LLMBatchPreparationError(f"missing chunks CSV: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise LLMBatchPreparationError(f"cannot read chunks CSV {path}: {exc}") from exc


def validate_chunks(chunks: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in chunks.columns]
    if missing:
        raise LLMBatchPreparationError("chunks CSV is missing column(s): " + ", ".join(missing))

    if chunks["chunk_id"].duplicated().any():
        duplicates = chunks.loc[chunks["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise LLMBatchPreparationError(f"duplicate chunk_id values found: {duplicates}")

    if chunks["text"].fillna("").astype(str).str.strip().eq("").any():
        raise LLMBatchPreparationError("one or more chunks have empty text")


def prepare_output_directories() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def remove_stale_batches() -> None:
    for path in BATCH_DIR.glob("batch_*.jsonl"):
        path.unlink()


def write_batches(chunks: pd.DataFrame) -> list[BatchSummary]:
    summaries: list[BatchSummary] = []
    records = chunks.loc[:, JSONL_COLUMNS].astype(str).to_dict(orient="records")

    for batch_index, start in enumerate(range(0, len(records), BATCH_SIZE), start=1):
        batch_records = records[start : start + BATCH_SIZE]
        path = BATCH_DIR / f"batch_{batch_index:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in batch_records:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
        summaries.append(BatchSummary(path=path, chunk_count=len(batch_records)))

    return summaries


def write_prompt() -> Path:
    PROMPT_PATH.write_text(PROMPT_TEXT, encoding="utf-8")
    return PROMPT_PATH


def write_template(chunks: pd.DataFrame) -> Path:
    template = chunks.loc[:, ["chunk_id", "doc_id", "type", "period"]].copy()
    for column in ("policy_direction", "communication_function", "certainty", "evidence"):
        template[column] = ""

    template.to_csv(
        TEMPLATE_PATH,
        columns=TEMPLATE_COLUMNS,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    return TEMPLATE_PATH


def validate_outputs(chunks: pd.DataFrame, batches: list[BatchSummary]) -> None:
    seen_chunk_ids: list[str] = []

    for batch in batches:
        if batch.chunk_count > BATCH_SIZE:
            raise LLMBatchPreparationError(f"batch exceeds size limit: {batch.path}")

        with batch.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if set(record) != set(JSONL_COLUMNS):
                    raise LLMBatchPreparationError(f"unexpected JSONL columns in {batch.path}")
                if not str(record["text"]).strip():
                    raise LLMBatchPreparationError(f"empty text found in {batch.path}")
                seen_chunk_ids.append(str(record["chunk_id"]))

    expected_chunk_ids = chunks["chunk_id"].astype(str).tolist()
    if seen_chunk_ids != expected_chunk_ids:
        raise LLMBatchPreparationError("batch files do not preserve every chunk exactly once")

    if len(seen_chunk_ids) != len(set(seen_chunk_ids)):
        raise LLMBatchPreparationError("duplicate chunk_id found across batch files")


def print_summary(chunks: pd.DataFrame, batches: list[BatchSummary], generated_files: list[Path]) -> None:
    print("LLM batch preparation summary:")
    print(f"total number of chunks: {len(chunks)}")
    print(f"number of batches: {len(batches)}")
    print("number of chunks in each batch:")
    for batch in batches:
        print(f"- {batch.path.name}: {batch.chunk_count}")

    print("\nGenerated files:")
    for path in generated_files:
        print(f"- {path}")


def main() -> int:
    try:
        chunks = load_chunks(CHUNKS_PATH)
        validate_chunks(chunks)
        prepare_output_directories()
        remove_stale_batches()
        batches = write_batches(chunks)
        prompt_path = write_prompt()
        template_path = write_template(chunks)
        validate_outputs(chunks, batches)

        generated_files = [batch.path for batch in batches] + [prompt_path, template_path]
        print_summary(chunks, batches, generated_files)
    except LLMBatchPreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: file operation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
