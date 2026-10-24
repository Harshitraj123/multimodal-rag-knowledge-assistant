"""
Document loaders for all supported file types.
Returns raw extracted content with source metadata.
Clearly reports unsupported or failed extractions.
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv", ".pptx",
    ".png", ".jpg", ".jpeg",
}


@dataclass
class RawDocument:
    """Container for raw content extracted from a file."""
    source_path: str
    file_name: str
    file_type: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_errors: list[str] = field(default_factory=list)


def load_document(file_path: str | Path) -> RawDocument:
    """
    Load and extract content from a file.
    Routes to appropriate loader based on extension.
    Returns RawDocument with populated fields and any extraction errors.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Loading document: {path.name} (type={ext})")

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_markdown,
        ".csv": _load_csv,
        ".pptx": _load_pptx,
        ".png": _load_image,
        ".jpg": _load_image,
        ".jpeg": _load_image,
    }

    loader_fn = loaders[ext]
    doc = loader_fn(path)
    logger.info(
        f"Loaded {path.name}: {len(doc.pages)} pages, "
        f"{len(doc.tables)} tables, {len(doc.images)} images"
    )
    return doc


def _load_pdf(path: Path) -> RawDocument:
    """Extract text, tables, and images from PDF."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="pdf",
    )
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            doc.metadata["total_pages"] = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                page_data: dict[str, Any] = {
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text),
                }
                doc.pages.append(page_data)

                # Extract tables
                for tbl_idx, tbl in enumerate(page.extract_tables() or []):
                    if tbl:
                        doc.tables.append({
                            "page_number": page_num,
                            "table_index": tbl_idx,
                            "headers": tbl[0] if tbl else [],
                            "rows": tbl[1:] if len(tbl) > 1 else [],
                            "source": path.name,
                        })

                # Render page as image for visual content
                try:
                    img = page.to_image(resolution=150)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    doc.images.append({
                        "page_number": page_num,
                        "image_type": "page_render",
                        "image_bytes": buf.getvalue(),
                        "source": path.name,
                        "description": f"Rendered page {page_num} of {path.name}",
                    })
                except Exception as e:
                    logger.debug(f"Page render failed p{page_num}: {e}")

    except ImportError:
        doc.extraction_errors.append("pdfplumber not installed; PDF parsing unavailable")
        logger.error("pdfplumber not installed")
    except Exception as e:
        doc.extraction_errors.append(f"PDF extraction error: {e}")
        logger.error(f"PDF extraction failed for {path.name}: {e}")

    return doc


def _load_docx(path: Path) -> RawDocument:
    """Extract paragraphs, headings, tables, and images from DOCX."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="docx",
    )
    try:
        from docx import Document
        from docx.oxml.ns import qn
        import base64

        word_doc = Document(str(path))
        full_text_parts = []
        para_idx = 0

        for para in word_doc.paragraphs:
            text = para.text.strip()
            if text:
                style_name = para.style.name if para.style else "Normal"
                full_text_parts.append(text)
                para_idx += 1

        doc.pages.append({
            "page_number": 1,
            "text": "\n".join(full_text_parts),
            "char_count": sum(len(t) for t in full_text_parts),
        })

        for tbl_idx, tbl in enumerate(word_doc.tables):
            headers = []
            rows = []
            for row_idx, row in enumerate(tbl.rows):
                cells = [cell.text.strip() for cell in row.cells]
                if row_idx == 0:
                    headers = cells
                else:
                    rows.append(cells)
            doc.tables.append({
                "page_number": 1,
                "table_index": tbl_idx,
                "headers": headers,
                "rows": rows,
                "source": path.name,
            })

        # Extract embedded images
        for rel in word_doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_data = rel.target_part.blob
                    doc.images.append({
                        "page_number": 1,
                        "image_type": "embedded",
                        "image_bytes": img_data,
                        "source": path.name,
                        "description": f"Embedded image from {path.name}",
                    })
                except Exception as e:
                    logger.debug(f"Could not extract image from DOCX: {e}")

    except ImportError:
        doc.extraction_errors.append("python-docx not installed; DOCX parsing unavailable")
    except Exception as e:
        doc.extraction_errors.append(f"DOCX extraction error: {e}")
        logger.error(f"DOCX extraction failed for {path.name}: {e}")

    return doc


def _load_text(path: Path) -> RawDocument:
    """Load plain text file."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="txt",
    )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc.pages.append({
            "page_number": 1,
            "text": text,
            "char_count": len(text),
        })
    except Exception as e:
        doc.extraction_errors.append(f"Text read error: {e}")
        logger.error(f"Text load failed for {path.name}: {e}")

    return doc


def _load_markdown(path: Path) -> RawDocument:
    """Load Markdown file, preserving heading structure."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="md",
    )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc.pages.append({
            "page_number": 1,
            "text": text,
            "char_count": len(text),
        })
        # Extract headings as metadata
        headings = [
            line.strip() for line in text.splitlines()
            if line.startswith("#")
        ]
        doc.metadata["headings"] = headings[:20]  # cap at 20
    except Exception as e:
        doc.extraction_errors.append(f"Markdown read error: {e}")
        logger.error(f"Markdown load failed for {path.name}: {e}")

    return doc


def _load_csv(path: Path) -> RawDocument:
    """Load CSV, preserving column names and row relationships."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="csv",
    )
    try:
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                rows.append(dict(row))

        doc.metadata["columns"] = list(headers)
        doc.metadata["row_count"] = len(rows)

        doc.tables.append({
            "page_number": 1,
            "table_index": 0,
            "headers": list(headers),
            "rows": [[row.get(h, "") for h in headers] for row in rows],
            "source": path.name,
        })

        # Also create a text representation
        lines = [", ".join(str(h) for h in headers)]
        for row in rows[:500]:  # cap for very large CSVs
            lines.append(", ".join(str(v) for v in row.values()))
        text = "\n".join(lines)
        doc.pages.append({
            "page_number": 1,
            "text": text,
            "char_count": len(text),
        })

    except Exception as e:
        doc.extraction_errors.append(f"CSV read error: {e}")
        logger.error(f"CSV load failed for {path.name}: {e}")

    return doc


def _load_pptx(path: Path) -> RawDocument:
    """Extract text, tables, and images from PPTX."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type="pptx",
    )
    try:
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation(str(path))
        doc.metadata["total_slides"] = len(prs.slides)

        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            texts.append(t)
                if shape.has_table:
                    tbl = shape.table
                    headers = [tbl.cell(0, c).text.strip() for c in range(tbl.columns.__len__())]
                    rows = []
                    for r in range(1, tbl.rows.__len__()):
                        rows.append([tbl.cell(r, c).text.strip() for c in range(tbl.columns.__len__())])
                    doc.tables.append({
                        "page_number": slide_num,
                        "table_index": 0,
                        "headers": headers,
                        "rows": rows,
                        "source": path.name,
                    })

            text = "\n".join(texts)
            doc.pages.append({
                "page_number": slide_num,
                "text": text,
                "char_count": len(text),
            })

    except ImportError:
        doc.extraction_errors.append("python-pptx not installed; PPTX parsing unavailable")
        logger.warning("python-pptx not installed")
    except Exception as e:
        doc.extraction_errors.append(f"PPTX extraction error: {e}")
        logger.error(f"PPTX load failed for {path.name}: {e}")

    return doc


def _load_image(path: Path) -> RawDocument:
    """Load standalone image file."""
    doc = RawDocument(
        source_path=str(path),
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
    )
    try:
        img_bytes = path.read_bytes()
        doc.images.append({
            "page_number": 1,
            "image_type": "standalone",
            "image_bytes": img_bytes,
            "source": path.name,
            "description": f"Image file: {path.name}",
        })
        doc.pages.append({
            "page_number": 1,
            "text": f"[Image: {path.name}]",
            "char_count": len(path.name) + 9,
        })
    except Exception as e:
        doc.extraction_errors.append(f"Image load error: {e}")
        logger.error(f"Image load failed for {path.name}: {e}")

    return doc
