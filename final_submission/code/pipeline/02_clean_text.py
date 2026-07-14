"""Clean extracted Bank of Japan text files without linguistic processing."""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

BOILERPLATE_LINES = {
    "ホーム",
    "サイト内検索",
    "ページの先頭へ",
    "日本銀行 Bank of Japan",
    "利用条件",
    "プライバシーポリシー",
    "サイトマップ",
}

PAGE_NUMBER_RE = re.compile(r"^[-–—―_\s]*\d+[-–—―_\s]*$")
DECORATIVE_RE = re.compile(r"^[-–—―_=*・･●○■□◆◇▼▽▲△━─\s]+$")
SYMBOL_ONLY_RE = re.compile(r"^[^\w\s]+$")
SPACE_TAB_RE = re.compile(r"[ \t]+")
REFERENCE_MARKERS = {"（参考）", "(参考)"}


class CleaningError(Exception):
    """Raised when a text file cannot be cleaned."""


@dataclass(frozen=True)
class CleaningResult:
    """Summary of one cleaned document."""

    input_name: str
    output_name: str
    document_type: str
    original_character_count: int
    cleaned_character_count: int
    cleaned_paragraph_count: int
    reference_truncation_applied: bool
    truncation_marker_found: str | None
    removed_character_count: int
    removed_line_count: int


def read_text(path: Path) -> str:
    """Read one UTF-8 text file with a clear error on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CleaningError(f"unreadable text file: {exc}") from exc


def clean_text(
    text: str, *, is_statement: bool
) -> tuple[str, int, bool, str | None, int]:
    """Return cleaned text, paragraph count, truncation details, and removed lines."""
    normalized = unicodedata.normalize("NFKC", text)
    original_lines = normalized.splitlines()

    kept_lines: list[str] = []
    for line in original_lines:
        cleaned_line = SPACE_TAB_RE.sub(" ", line.strip())
        if _should_remove_line(cleaned_line):
            continue
        kept_lines.append(cleaned_line)

    truncation_applied = False
    truncation_marker: str | None = None
    if is_statement:
        kept_lines, truncation_applied, truncation_marker = _truncate_reference_section(
            kept_lines
        )

    paragraphs = _dedupe_consecutive_paragraphs(_split_paragraphs(kept_lines))
    cleaned_text = "\n\n".join(paragraphs).strip()
    if cleaned_text:
        cleaned_text += "\n"

    removed_lines = len(original_lines) - len(cleaned_text.splitlines())
    return (
        cleaned_text,
        len(paragraphs),
        truncation_applied,
        truncation_marker,
        removed_lines,
    )


def document_type_for(path: Path) -> str:
    lower_name = path.name.lower()
    if "statement" in lower_name:
        return "statement"
    if "speech" in lower_name:
        return "speech"
    return "other"


def _should_remove_line(line: str) -> bool:
    if not line:
        return False
    if line in BOILERPLATE_LINES:
        return True
    if "Copyright" in line or "©" in line:
        return True
    if PAGE_NUMBER_RE.fullmatch(line):
        return True
    if DECORATIVE_RE.fullmatch(line):
        return True
    return bool(SYMBOL_ONLY_RE.fullmatch(line))


def _truncate_reference_section(lines: list[str]) -> tuple[list[str], bool, str | None]:
    """Remove a statement reference section that starts at a standalone marker."""
    marker_index = next(
        (index for index, line in enumerate(lines) if line in REFERENCE_MARKERS),
        None,
    )
    if marker_index is None:
        return lines, False, None

    cut_index = marker_index
    previous_index = marker_index - 1
    while previous_index >= 0 and not lines[previous_index]:
        previous_index -= 1

    if previous_index >= 0 and lines[previous_index] == "以上":
        cut_index = previous_index

    return lines[:cut_index], True, lines[marker_index]


def _split_paragraphs(lines: list[str]) -> list[str]:
    """Group non-empty lines into paragraphs separated by blank lines."""
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if line:
            current.append(line)
            continue

        if current:
            paragraphs.append("\n".join(current))
            current = []

    if current:
        paragraphs.append("\n".join(current))

    return paragraphs


def _dedupe_consecutive_paragraphs(paragraphs: list[str]) -> list[str]:
    deduped: list[str] = []
    previous: str | None = None

    for paragraph in paragraphs:
        if paragraph == previous:
            continue
        deduped.append(paragraph)
        previous = paragraph

    return deduped


def clean_file(input_path: Path, output_dir: Path) -> CleaningResult:
    original_text = read_text(input_path)
    document_type = document_type_for(input_path)
    cleaned_text, paragraph_count, truncation_applied, marker_found, removed_line_count = (
        clean_text(original_text, is_statement=document_type == "statement")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_clean.txt"
    output_path.write_text(cleaned_text, encoding="utf-8")

    return CleaningResult(
        input_name=input_path.name,
        output_name=output_path.name,
        document_type=document_type,
        original_character_count=len(original_text),
        cleaned_character_count=len(cleaned_text),
        cleaned_paragraph_count=paragraph_count,
        reference_truncation_applied=truncation_applied,
        truncation_marker_found=marker_found,
        removed_character_count=max(len(original_text) - len(cleaned_text), 0),
        removed_line_count=removed_line_count,
    )


def iter_text_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise CleaningError(f"missing input directory: {input_dir}")
    if not input_dir.is_dir():
        raise CleaningError(f"input path is not a directory: {input_dir}")

    text_files = sorted(input_dir.glob("*.txt"))
    if not text_files:
        raise CleaningError(f"no .txt files found in: {input_dir}")

    return text_files


def main() -> int:
    try:
        text_files = iter_text_files(EXTRACTED_DIR)
    except CleaningError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    successes = 0
    failures = 0

    for text_file in text_files:
        try:
            result = clean_file(text_file, CLEANED_DIR)
        except CleaningError as exc:
            failures += 1
            print(f"ERROR: {text_file.name}: {exc}", file=sys.stderr)
            continue

        successes += 1
        print(f"Input: {result.input_name}")
        print(f"Document type: {result.document_type}")
        print(f"Original character count: {result.original_character_count}")
        print(f"Cleaned character count: {result.cleaned_character_count}")
        print(f"Cleaned paragraph count: {result.cleaned_paragraph_count}")
        print(
            "Reference-section truncation applied: "
            f"{result.reference_truncation_applied}"
        )
        print(f"Truncation marker found: {result.truncation_marker_found or 'None'}")
        print(f"Removed characters: {result.removed_character_count}")
        print()

    print(
        f"Summary: cleaned {successes} file(s), failed {failures} file(s), "
        f"output directory: {CLEANED_DIR}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
