import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import check_ollama_server, process_pdf, validate_pdf_path


class PathRequest(BaseModel):
    pdf_path: str


app = FastAPI(
    title="PDF Summarization API",
    version="1.0.0",
    description="Expose main.py summarization pipeline for web clients",
)


def split_summary_into_pages(summary_text: str):
    """Split combined summary into page chunks for easier frontend display."""
    page_entries = extract_page_entries(summary_text)
    if page_entries:
        return [entry["raw_text"] for entry in page_entries]
    sections = [chunk.strip() for chunk in summary_text.split("=" * 60) if chunk.strip()]
    return sections


def extract_page_entries(summary_text: str):
    """Extract page metadata and raw page note text from combined summary."""
    cleaned = summary_text.replace("\r\n", "\n")
    cleaned = re.sub(r"\n?={10,}\n?", "\n", cleaned).strip()

    pattern = re.compile(
        r"(?im)^PAGE\s+(\d+)\s*-\s*Student Notes\s*\((Text|Visual)\)\s*$"
    )
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return []

    pages = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        raw_text = cleaned[start:end].strip()
        pages.append(
            {
                "page_number": int(match.group(1)),
                "source_type": match.group(2).lower(),
                "raw_text": raw_text,
            }
        )

    return pages


def parse_structured_page(raw_text: str):
    """Parse student note text into strict JSON fields."""
    result = {
        "main_topic": "",
        "key_points": [],
        "terms_to_remember": [],
        "exam_tip": "",
        "one_line_recap": "",
        "raw_text": raw_text,
    }

    current = None
    for original_line in raw_text.splitlines():
        line = original_line.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("main topic:"):
            result["main_topic"] = line.split(":", 1)[1].strip()
            current = "main_topic"
            continue
        if low.startswith("main topic/"):
            result["main_topic"] = line.split(":", 1)[1].strip() if ":" in line else line
            current = "main_topic"
            continue
        if low.startswith("key points:"):
            current = "key_points"
            continue
        if low.startswith("terms to remember:"):
            current = "terms_to_remember"
            continue
        if low.startswith("exam tip:"):
            result["exam_tip"] = line.split(":", 1)[1].strip()
            current = "exam_tip"
            continue
        if low.startswith("one-line recap:"):
            result["one_line_recap"] = line.split(":", 1)[1].strip()
            current = "one_line_recap"
            continue

        bullet_text = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", line).strip()
        if current == "key_points":
            if bullet_text:
                result["key_points"].append(bullet_text)
            continue
        if current == "terms_to_remember":
            if bullet_text:
                result["terms_to_remember"].append(bullet_text)
            continue
        if current == "exam_tip":
            result["exam_tip"] = (result["exam_tip"] + " " + line).strip()
            continue
        if current == "one_line_recap":
            result["one_line_recap"] = (result["one_line_recap"] + " " + line).strip()
            continue

    # Fallbacks for partially structured model output.
    if not result["main_topic"]:
        first_line = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), "")
        result["main_topic"] = first_line[:180]

    if not result["key_points"]:
        candidate_lines = [ln.strip(" -•\t") for ln in raw_text.splitlines() if len(ln.strip()) > 20]
        result["key_points"] = candidate_lines[:4]

    if not result["one_line_recap"]:
        result["one_line_recap"] = result["key_points"][0] if result["key_points"] else ""

    return result


def build_structured_pages(summary_text: str):
    """Build strict per-page JSON notes for frontend rendering."""
    page_entries = extract_page_entries(summary_text)
    if not page_entries:
        # If page headers are missing, return one synthesized page.
        return [
            {
                "page_number": 1,
                "source_type": "unknown",
                **parse_structured_page(summary_text.strip()),
            }
        ]

    structured = []
    for entry in page_entries:
        structured.append(
            {
                "page_number": entry["page_number"],
                "source_type": entry["source_type"],
                **parse_structured_page(entry["raw_text"]),
            }
        )

    return structured

# Allow React app (and other clients) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "PDF Summarization API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    ollama_ok = check_ollama_server()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
    }


@app.post("/summarize/upload")
async def summarize_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not available")

    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            data = await file.read()
            temp_pdf.write(data)
            temp_pdf_path = temp_pdf.name

        summary = process_pdf(temp_pdf_path)

        output_path = Path("summary_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL HYBRID SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)

        return {
            "filename": file.filename,
            "summary": summary,
            "pages": split_summary_into_pages(summary),
            "output_file": str(output_path.resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass


@app.post("/summarize/path")
def summarize_path(payload: PathRequest):
    if not validate_pdf_path(payload.pdf_path):
        raise HTTPException(status_code=400, detail="Invalid PDF path")

    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not available")

    try:
        summary = process_pdf(payload.pdf_path)

        output_path = Path("summary_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL HYBRID SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)

        return {
            "pdf_path": str(Path(payload.pdf_path).resolve()),
            "summary": summary,
            "pages": split_summary_into_pages(summary),
            "output_file": str(output_path.resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/summarize/upload/structured")
async def summarize_upload_structured(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not available")

    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            data = await file.read()
            temp_pdf.write(data)
            temp_pdf_path = temp_pdf.name

        summary = process_pdf(temp_pdf_path)
        structured_pages = build_structured_pages(summary)

        output_path = Path("summary_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL HYBRID SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)

        return {
            "filename": file.filename,
            "summary": summary,
            "structured_pages": structured_pages,
            "page_count": len(structured_pages),
            "output_file": str(output_path.resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass


@app.post("/summarize/path/structured")
def summarize_path_structured(payload: PathRequest):
    if not validate_pdf_path(payload.pdf_path):
        raise HTTPException(status_code=400, detail="Invalid PDF path")

    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not available")

    try:
        summary = process_pdf(payload.pdf_path)
        structured_pages = build_structured_pages(summary)

        output_path = Path("summary_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL HYBRID SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)

        return {
            "pdf_path": str(Path(payload.pdf_path).resolve()),
            "summary": summary,
            "structured_pages": structured_pages,
            "page_count": len(structured_pages),
            "output_file": str(output_path.resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
