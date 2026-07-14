"""Extract main text from locally downloaded Bank of Japan HTML files."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
    from bs4.element import Comment, NavigableString, Tag
except ImportError as import_error:  # pragma: no cover - exercised by runtime setup.
    BeautifulSoup = None  # type: ignore[assignment]
    Comment = NavigableString = Tag = None  # type: ignore[assignment]
    BS4_IMPORT_ERROR: ImportError | None = import_error
else:
    BS4_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

BLOCK_TAGS = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd"}
)
DROP_TAGS = ("script", "style", "noscript", "header", "footer", "nav", "aside", "form")
DROP_SELECTORS = (
    "#header_area",
    "#footer",
    "#topic_path",
    "#sns",
    "#overlay",
    ".block_skip",
    ".search_form",
    ".link-list01",
    ".lang",
    ".visually-hidden",
    ".txt-hide",
    "[role='navigation']",
    "[role='search']",
    "button",
    "input",
    "select",
    "option",
    "datalist",
    "img",
    "svg",
)
MAIN_CONTENT_SELECTORS = (
    "main#contents",
    "article",
    "main",
    "#contents",
    ".contents",
    ".article",
)


class ExtractionError(Exception):
    """Raised when a local HTML file cannot be extracted."""


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of one extracted document."""

    input_name: str
    output_name: str
    character_count: int
    paragraph_count: int


def read_html(path: Path) -> str:
    """Read one UTF-8 HTML file with a clear extraction error on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExtractionError(f"unreadable HTML: {exc}") from exc


def prepare_soup(html: str) -> Any:
    """Parse HTML and remove site UI before text extraction."""
    if BeautifulSoup is None:
        raise ExtractionError("BeautifulSoup is not available")

    soup = BeautifulSoup(html, "html.parser")

    # Remove BOJ page chrome and hidden accessibility duplicates, not article text.
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    for selector in DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    for anchor in soup.find_all("a"):
        text = anchor.get_text(strip=True)
        title = anchor.get("title", "")
        if text == "本文に戻る" or title.startswith("Return"):
            anchor.decompose()

    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")

    return soup


def find_main_content(soup: Any) -> Any:
    """Find the most likely main article container."""
    for selector in MAIN_CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and _visible_text_length(candidate) > 0:
            return candidate

    raise ExtractionError("no main text found")


def extract_blocks(root: Any) -> list[str]:
    """Extract block text in source order while avoiding nested duplicates."""
    blocks: list[str] = []

    for element in root.descendants:
        if not _is_tag(element) or element.name not in BLOCK_TAGS:
            continue

        text = _own_block_text(element)
        if text:
            blocks.append(text)

    if not blocks:
        raise ExtractionError("no main text found")

    return blocks


def _own_block_text(block: Any) -> str:
    if block.name == "h1":
        return _clean_block_text(block.get_text("\n"))

    pieces: list[str] = []

    for node in block.descendants:
        if not _is_navigable_string(node) or _is_comment(node):
            continue
        if _inside_nested_block(node, block):
            continue
        pieces.append(str(node))

    return _clean_block_text("".join(pieces))


def _inside_nested_block(node: Any, block: Any) -> bool:
    parent = node.parent
    while parent is not None and parent is not block:
        if _is_tag(parent) and parent.name in BLOCK_TAGS:
            return True
        parent = parent.parent
    return False


def _clean_block_text(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _visible_text_length(tag: Any) -> int:
    return len(_clean_block_text(tag.get_text()))


def _is_tag(value: Any) -> bool:
    return Tag is not None and isinstance(value, Tag)


def _is_navigable_string(value: Any) -> bool:
    return NavigableString is not None and isinstance(value, NavigableString)


def _is_comment(value: Any) -> bool:
    return Comment is not None and isinstance(value, Comment)


def extract_file(input_path: Path, output_dir: Path) -> ExtractionResult:
    html = read_html(input_path)
    soup = prepare_soup(html)
    main_content = find_main_content(soup)
    blocks = extract_blocks(main_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.txt"

    if output_path.resolve() == input_path.resolve():
        raise ExtractionError("output path would overwrite the source HTML")

    extracted_text = "\n\n".join(blocks).strip() + "\n"
    output_path.write_text(extracted_text, encoding="utf-8")

    return ExtractionResult(
        input_name=input_path.name,
        output_name=output_path.name,
        character_count=len(extracted_text),
        paragraph_count=len(blocks),
    )


def iter_html_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise ExtractionError(f"missing input directory: {input_dir}")
    if not input_dir.is_dir():
        raise ExtractionError(f"input path is not a directory: {input_dir}")
    return sorted(input_dir.glob("*.html"))


def main() -> int:
    if BS4_IMPORT_ERROR is not None:
        print(
            "ERROR: BeautifulSoup is required. Install beautifulsoup4 before running "
            "this extraction stage.",
            file=sys.stderr,
        )
        return 1

    try:
        html_files = iter_html_files(RAW_DIR)
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    successes = 0
    failures = 0

    for html_file in html_files:
        try:
            result = extract_file(html_file, EXTRACTED_DIR)
        except ExtractionError as exc:
            failures += 1
            print(f"ERROR: {html_file.name}: {exc}", file=sys.stderr)
            continue

        successes += 1
        print(f"Input: {result.input_name}")
        print(f"Output: {result.output_name}")
        print(f"Extracted character count: {result.character_count}")
        print(f"Extracted paragraph count: {result.paragraph_count}")
        print()

    print(
        f"Summary: extracted {successes} file(s), failed {failures} file(s), "
        f"output directory: {EXTRACTED_DIR}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
