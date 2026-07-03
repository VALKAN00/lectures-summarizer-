# How to Run This Project

This project summarizes PDF slides using:
- **T5** for text-heavy pages
- **LLaVA (via Ollama)** for visual/diagram-heavy pages
- **OCR (Tesseract + pytesseract)** for pages with poor text extraction

## 1) Open project folder

Use PowerShell and go to the workspace folder:

```powershell
cd "D:\summary model\summarization"
```

## 2) Activate virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## 3) Install Python dependencies

```powershell
pip install -r requirements.txt
```

## 4) Install required system tools

### A) Ollama (required)
1. Install Ollama: https://ollama.com/download
2. Start Ollama (desktop app or service)
3. Pull the vision model used by this project:

```powershell
ollama pull llava
```

### B) Tesseract OCR (recommended; improves extraction)
Follow the detailed guide in `OCR_SETUP.md`.

Quick Windows steps:
1. Install Tesseract OCR (UB Mannheim build): https://github.com/UB-Mannheim/tesseract/wiki
2. Default path should be:
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Verify install:

```powershell
tesseract --version
```

### C) Poppler for pdf2image (required for image conversion)
`pdf2image` needs Poppler binaries on Windows.

1. Download a Windows Poppler build (example source):
   - https://github.com/oschwartz10612/poppler-windows/releases
2. Extract it, then add the `bin` folder to PATH (for example `C:\poppler\Library\bin`)
3. Restart terminal and verify:

```powershell
where.exe pdftoppm
```

If command is found, Poppler is correctly set.

## 5) Run the summarizer

### Option 1: Use default test file
The script defaults to `test.pdf` when no argument is provided.

```powershell
python main.py
```

### Option 2: Pass your own PDF

```powershell
python main.py "path\to\your_file.pdf"
```

Example:

```powershell
python main.py "test.pdf"
```

## 6) Run as API (FastAPI) for React app

Start the API server:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger docs:
- `http://localhost:8000/docs`

Available endpoints:
- `GET /health` -> checks API + Ollama availability
- `POST /summarize/upload` -> upload a PDF file as `multipart/form-data`
- `POST /summarize/path` -> summarize a local PDF path (server-side path)
- `POST /summarize/upload/structured` -> upload a PDF and get strict per-page JSON notes
- `POST /summarize/path/structured` -> summarize server-side PDF path and get strict per-page JSON notes

Student-friendly response notes:
- `summary`: Full combined student notes text
- `pages`: List of page-wise note blocks (easier to render in React cards/tabs)
- `structured_pages`: Strict page objects with `main_topic`, `key_points`, `terms_to_remember`, `exam_tip`, `one_line_recap`
- `page_count`: Number of structured pages in response

Example upload request (PowerShell):

```powershell
curl.exe -X POST "http://localhost:8000/summarize/upload" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@test.pdf"
```

Example path request:

```powershell
curl.exe -X POST "http://localhost:8000/summarize/path" ^
  -H "Content-Type: application/json" ^
  -d "{\"pdf_path\":\"test.pdf\"}"
```

React frontend base URL:
- `http://localhost:8000`

Notes:
- Keep Ollama running before calling the API.
- First request can be slow because models need to load.
- Visual pages are now formatted into student study notes with sections like main topic, key points, terms, and recap.

## 7) Output files

After a successful run:
- Summary text is written to: `summary_output.txt`
- Logs are written to: `summarization.log`

## 8) Common issues

- **Cannot connect to Ollama server**
  - Start Ollama, then run `ollama list`.

- **tesseract is not installed**
  - Install Tesseract and verify path/version.

- **PDF image conversion errors (pdftoppm not found)**
  - Install Poppler and ensure its `bin` path is in PATH.

- **Large model download / slow first run**
  - First run may take longer due to model downloads and initialization.

---

You can now run the pipeline end-to-end with:

```powershell
python main.py "test.pdf"
```
