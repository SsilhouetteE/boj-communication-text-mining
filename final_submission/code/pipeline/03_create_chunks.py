"""Create paragraph-based text chunks from cleaned Bank of Japan documents."""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "documents.csv"
OUTPUT_PATH = CLEANED_DIR / "chunks.csv"

REQUIRED_METADATA_FIELDS = ("doc_id", "date", "period", "type", "speaker", "title")
OUTPUT_COLUMNS = (
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
BOILERPLATE_PATTERNS = (
    "ホーム",
    "サイト内検索",
    "ページの先頭へ",
    "日本銀行 Bank of Japan",
    "Copyright",
    "利用条件",
    "プライバシーポリシー",
    "サイトマップ",
)
STATEMENT_ADMIN_PATTERNS = (
    "開催時間",
    "出席委員",
    "政府出席者",
    "（参考）",
    "(参考)",
)

DOC_ID_RE = re.compile(r"(D\d+)", re.IGNORECASE)
DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
DATE_LINE_RE = re.compile(r"^20\d{2}年\d{1,2}月\d{1,2}日$")
NUMBERED_HEADING_RE = re.compile(
    r"^((\d+|[０-９]+)[.．、]|[（(]\d+[）)]|\[\d+\]|第[一二三四五六七八九十\d]+)"
)
SENTENCE_RE = re.compile(r".+?[。！？]|.+$", re.DOTALL)


class ChunkingError(Exception):
    """Raised when chunk creation cannot continue safely."""


@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str
    date: str
    period: str
    type: str
    speaker: str
    title: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    date: str
    period: str
    type: str
    speaker: str
    title: str
    section_title: str
    char_count: int
    text: str


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    doc_type: str
    original_character_count: int
    chunk_count: int
    min_chunk_length: int
    median_chunk_length: float
    max_chunk_length: int
    below_100_count: int
    above_500_count: int


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ChunkingError(f"cannot read cleaned text file {path}: {exc}") from exc


def load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ChunkingError(f"missing metadata file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = [field for field in REQUIRED_METADATA_FIELDS if field not in fieldnames]
            if missing:
                raise ChunkingError(
                    "metadata file is missing required field(s): " + ", ".join(missing)
                )
            return [{key: value or "" for key, value in row.items()} for row in reader]
    except OSError as exc:
        raise ChunkingError(f"cannot read metadata file {path}: {exc}") from exc


def iter_cleaned_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise ChunkingError(f"missing cleaned input directory: {input_dir}")
    if not input_dir.is_dir():
        raise ChunkingError(f"cleaned input path is not a directory: {input_dir}")

    files = sorted(input_dir.glob("*_clean.txt"))
    if not files:
        raise ChunkingError(f"no *_clean.txt files found in: {input_dir}")
    return files


def metadata_for_file(
    path: Path, text: str, metadata_rows: list[dict[str, str]]
) -> DocumentMetadata:
    fallback = infer_metadata(path, text)
    matched_row = find_metadata_row(path, fallback.doc_id, metadata_rows)

    if not matched_row:
        return fallback

    values = {
        field: matched_row.get(field, "").strip() or getattr(fallback, field)
        for field in REQUIRED_METADATA_FIELDS
    }
    return DocumentMetadata(**values)


def find_metadata_row(
    path: Path, doc_id: str, metadata_rows: list[dict[str, str]]
) -> dict[str, str] | None:
    clean_name = path.name
    source_name = clean_name.removesuffix("_clean.txt")

    for row in metadata_rows:
        if row.get("doc_id", "").strip().upper() == doc_id.upper():
            return row

    for row in metadata_rows:
        searchable = " ".join(
            row.get(field, "") for field in ("local_path", "file_url", "page_url", "title")
        )
        if clean_name in searchable or source_name in searchable:
            return row

    return None


def infer_metadata(path: Path, text: str) -> DocumentMetadata:
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc_id = infer_doc_id(path)
    date = infer_date(path)
    doc_type = "statement" if "statement" in path.name.lower() else "speech"
    period = infer_period(date)
    speaker = infer_speaker(doc_type, non_empty_lines)
    title = infer_title(doc_type, non_empty_lines)

    return DocumentMetadata(
        doc_id=doc_id,
        date=date,
        period=period,
        type=doc_type,
        speaker=speaker,
        title=title,
    )


def infer_doc_id(path: Path) -> str:
    match = DOC_ID_RE.search(path.name)
    if not match:
        raise ChunkingError(f"cannot infer doc_id from filename: {path.name}")
    return match.group(1).upper()


def infer_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def infer_period(date: str) -> str:
    if not date:
        return ""
    if date < "2024-03-19":
        return "pre_end_nirp"
    if date == "2024-03-19":
        return "end_nirp"
    return "post_end_nirp"


def infer_speaker(doc_type: str, lines: list[str]) -> str:
    if doc_type == "statement":
        return "日本銀行"

    for line in lines[:8]:
        if "日本銀行" in line and not DATE_LINE_RE.fullmatch(line):
            return line
    return ""


def infer_title(doc_type: str, lines: list[str]) -> str:
    if not lines:
        return ""
    if doc_type == "statement":
        return lines[0]

    title_lines: list[str] = []
    for line in lines:
        if DATE_LINE_RE.fullmatch(line) or "日本銀行" in line:
            break
        title_lines.append(line)
    return " ".join(title_lines).strip() or lines[0]


def split_paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text.strip())
        if paragraph.strip()
    ]


def make_chunks_for_document(
    path: Path, metadata: DocumentMetadata, text: str
) -> tuple[list[Chunk], DocumentSummary]:
    chunks: list[Chunk] = []
    current_section = metadata.title
    buffer_text = ""
    buffer_section = current_section

    def flush_buffer() -> None:
        nonlocal buffer_text
        if not buffer_text.strip():
            return
        chunk_number = len(chunks) + 1
        chunk_text = buffer_text.strip()
        chunks.append(
            Chunk(
                chunk_id=f"{metadata.doc_id}_C{chunk_number:03d}",
                doc_id=metadata.doc_id,
                date=metadata.date,
                period=metadata.period,
                type=metadata.type,
                speaker=metadata.speaker,
                title=metadata.title,
                section_title=buffer_section,
                char_count=len(chunk_text),
                text=chunk_text,
            )
        )
        buffer_text = ""

    for paragraph in split_paragraphs(text):
        if is_front_matter(paragraph, metadata):
            continue

        if is_probable_heading(paragraph):
            if paragraph.strip() == "以上":
                continue
            flush_buffer()
            current_section = heading_text(paragraph)
            buffer_section = current_section
            continue

        for part in split_long_paragraph(paragraph):
            if not buffer_text:
                buffer_text = part
                buffer_section = current_section
                continue

            merged_text = f"{buffer_text}\n\n{part}"
            if buffer_section == current_section and len(buffer_text) < 150 and len(merged_text) <= 500:
                buffer_text = merged_text
                continue

            flush_buffer()
            buffer_text = part
            buffer_section = current_section

    flush_buffer()

    if not chunks:
        raise ChunkingError(f"document produced no chunks: {path.name}")

    lengths = [chunk.char_count for chunk in chunks]
    summary = DocumentSummary(
        doc_id=metadata.doc_id,
        doc_type=metadata.type,
        original_character_count=len(text),
        chunk_count=len(chunks),
        min_chunk_length=min(lengths),
        median_chunk_length=median(lengths),
        max_chunk_length=max(lengths),
        below_100_count=sum(1 for length in lengths if length < 100),
        above_500_count=sum(1 for length in lengths if length > 500),
    )
    return chunks, summary


def is_front_matter(paragraph: str, metadata: DocumentMetadata) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return True

    if len(lines) <= 2 and all(
        DATE_LINE_RE.fullmatch(line)
        or line == "日本銀行"
        or line == metadata.speaker
        for line in lines
    ):
        return True

    return False


def is_probable_heading(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False

    text = heading_text(paragraph)
    has_sentence_end = "。" in text or "！" in text or "？" in text
    if len(text) > 100:
        return False
    if NUMBERED_HEADING_RE.match(text) and not has_sentence_end:
        return True
    if not has_sentence_end:
        return True
    return False


def heading_text(paragraph: str) -> str:
    return " ".join(line.strip() for line in paragraph.splitlines() if line.strip())


def split_long_paragraph(paragraph: str) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= 500:
        return [paragraph]

    sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(paragraph) if sentence.strip()]
    if len(sentences) <= 1:
        return [paragraph]

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if current else sentence
        if current and len(candidate) > 500:
            parts.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        if parts and len(current) < 100 and len(parts[-1]) + len(current) <= 500:
            parts[-1] = f"{parts[-1]}{current}"
        else:
            parts.append(current)

    return parts


def validate_chunks(chunks: list[Chunk], expected_doc_ids: set[str]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    doc_ids_with_chunks = {chunk.doc_id for chunk in chunks}

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            errors.append(f"duplicate chunk_id: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)

        if not chunk.text.strip():
            errors.append(f"empty text for chunk_id: {chunk.chunk_id}")
        if chunk.char_count != len(chunk.text):
            errors.append(f"char_count mismatch for chunk_id: {chunk.chunk_id}")

        for field in REQUIRED_METADATA_FIELDS:
            if not getattr(chunk, field):
                errors.append(f"missing metadata field {field} for chunk_id: {chunk.chunk_id}")

        for pattern in BOILERPLATE_PATTERNS:
            if pattern in chunk.text:
                errors.append(f"boilerplate pattern {pattern!r} in chunk_id: {chunk.chunk_id}")

        if chunk.type == "statement":
            for pattern in STATEMENT_ADMIN_PATTERNS:
                if pattern in chunk.text:
                    errors.append(
                        f"statement admin pattern {pattern!r} in chunk_id: {chunk.chunk_id}"
                    )

    for doc_id in sorted(expected_doc_ids - doc_ids_with_chunks):
        errors.append(f"document produced no chunks: {doc_id}")

    return errors


def write_chunks_csv(chunks: list[Chunk], output_path: Path) -> None:
    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for chunk in chunks:
                writer.writerow(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "date": chunk.date,
                        "period": chunk.period,
                        "type": chunk.type,
                        "speaker": chunk.speaker,
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "char_count": chunk.char_count,
                        "text": chunk.text,
                    }
                )
    except OSError as exc:
        raise ChunkingError(f"cannot write chunks CSV {output_path}: {exc}") from exc


def print_summary(summaries: list[DocumentSummary]) -> None:
    print("Document-level summary:")
    for summary in summaries:
        print(f"doc_id: {summary.doc_id}")
        print(f"document type: {summary.doc_type}")
        print(f"original cleaned character count: {summary.original_character_count}")
        print(f"number of chunks: {summary.chunk_count}")
        print(f"minimum chunk length: {summary.min_chunk_length}")
        print(f"median chunk length: {summary.median_chunk_length}")
        print(f"maximum chunk length: {summary.max_chunk_length}")
        print(f"chunks below 100 characters: {summary.below_100_count}")
        print(f"chunks above 500 characters: {summary.above_500_count}")
        print()


def print_sample_chunks(chunks: list[Chunk]) -> None:
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for doc_id in sorted(chunks_by_doc):
        doc_chunks = chunks_by_doc[doc_id]
        print(f"Samples for {doc_id}:")
        for chunk in doc_chunks[:3]:
            print_chunk(chunk)
        print("Last chunk:")
        print_chunk(doc_chunks[-1])

    statement_count = sum(1 for chunk in chunks if chunk.type == "statement")
    speech_count = sum(1 for chunk in chunks if chunk.type == "speech")
    print(f"Total chunks: {len(chunks)}")
    print(f"Statement chunks: {statement_count}")
    print(f"Speech chunks: {speech_count}")


def print_chunk(chunk: Chunk) -> None:
    print(
        f"{chunk.chunk_id} | {chunk.type} | section: {chunk.section_title} | "
        f"chars: {chunk.char_count}"
    )
    print(chunk.text)
    print()


def main() -> int:
    try:
        metadata_rows = load_metadata(METADATA_PATH)
        cleaned_files = iter_cleaned_files(CLEANED_DIR)

        all_chunks: list[Chunk] = []
        summaries: list[DocumentSummary] = []
        expected_doc_ids: set[str] = set()

        for cleaned_file in cleaned_files:
            text = read_text(cleaned_file)
            metadata = metadata_for_file(cleaned_file, text, metadata_rows)
            expected_doc_ids.add(metadata.doc_id)
            chunks, summary = make_chunks_for_document(cleaned_file, metadata, text)
            all_chunks.extend(chunks)
            summaries.append(summary)

        validation_errors = validate_chunks(all_chunks, expected_doc_ids)
        if validation_errors:
            for error in validation_errors:
                print(f"VALIDATION ERROR: {error}", file=sys.stderr)
            return 1

        write_chunks_csv(all_chunks, OUTPUT_PATH)
    except ChunkingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Validation passed: {len(all_chunks)} chunk(s) across {len(summaries)} document(s).")
    print(f"Wrote: {OUTPUT_PATH}")
    print()
    print_summary(summaries)
    print_sample_chunks(all_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
