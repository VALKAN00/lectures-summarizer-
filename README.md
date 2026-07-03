# Summarization Model Documentation

## 1. Overview
This project provides an intelligent and hybrid PDF summarization pipeline built primarily for educational slides and student notes. It adapts to the content type of each page, seamlessly transitioning between layout-aware text extraction, Optical Character Recognition (OCR), Text-to-Text inference, and Multimodal Vision inference. 

By exposing the core Python logic (`main.py`) via a FastAPI backend (`api.py`), the model can be used dynamically as a standalone script or as a backend microservice.

---

## 2. Architecture & Core Technologies

The summarization pipeline utilizes a robust tech stack to handle varying PDF inputs:

*   **Text Outline Summarization (Transformers)**: Uses Hugging Face's `t5-base` to compress and synthesize text-heavy pages.
*   **Visual Data interpretation (Ollama)**: Uses `llava` to act as a fallback and visionary engine. It interprets spatial relationships, graphical flows, or diagrams on pages that contain few readable text blocks.
*   **Text & Image Extraction**: 
    *   `pdfplumber`: Programmatically extracts accessible text elements.
    *   `pytesseract` + Tesseract Engine: Acts automatically as a fallback OCR engine when `pdfplumber` detects fewer words than the defined threshold.
    *   `pdf2image` + Poppler: Converts raw PDF pages into images essential for OCR testing and the LLaVA vision model.
*   **Web API Framework**: Built on `FastAPI` (running on `uvicorn`), yielding performant stateless APIs.

---

## 3. Setup and Prerequisites

To deploy and test the model, a specific environment setup is required.

### System-Level Dependencies:
1.  **Ollama**: Install the Ollama service locally and pull the vision model:
    ```bash
    ollama pull llava
    ```
2.  **Tesseract OCR** *(Recommended)*: Enhances the system's ability to read scanned documents.
    *   **Windows**: Download installer, configure PATH (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
3.  **Poppler**: Required by `pdf2image`. 
    *   **Windows**: Extract binaries and attach the `/bin` directory to the system PATH.

### Python Environment:
Activate a virtual environment and install dependencies listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## 4. Pipeline Logic & Configuration

The entry point configuration is nested within a `CONFIG` dictionary in `main.py`. This orchestrates how the model makes decisions handling different data representations.

### Key Logic Gates:
*   **Word Threshold** (`word_threshold = 30`): If a PDF page generates `< 30` words natively, the system delegates that specific page to OCR and alternatively the `llava` vision model.
*   **OCR Enhancements** (`use_ocr = True`): Turns on `pytesseract` to scrape text hidden in images before fully reverting to visual summarization.

### Tuning Summaries:
*   `model_name`: Set to `"t5-base"` but customizable.
*   **Beam Search & Penalties**: Variables like `min_beams` (6), `length_penalty` (2.0), and `temperature` (0.3) force the model into logical, deterministic behaviors that combat summarization repetition and fragmentation.
*   **Student Formatting**: The model aims explicitly to format outputs as study notes (`preserve_structure = True`, `student_friendly = True`), prioritizing bullet points, main topics, and practical summaries.

---

## 5. API Integrations

The FastAPI application (`api.py`) exposes several endpoints, making the models externally consumable. Start the server using:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:
1.  **GET `/health`**: Validates whether the application is running and if the Ollama daemon successfully responds.
2.  **POST `/summarize/upload`**: Takes a physical file upload representing the PDF (`multipart/form-data`) and returns a flat Markdown/Text summary.
3.  **POST `/summarize/path`**: Receives a `pdf_path` payload in JSON and operates locally on the server file structure.
4.  **POST `/summarize/upload/structured`** / **POST `/summarize/path/structured`**:
    Highly customized endpoints built to parse the text summary into predictable JSON keys suited for frontend display mapping.

### Structured JSON Response Structure
The structured API calls return the payload divided accurately per page containing:
```json
{
  "structured_pages": [
    {
      "page_number": 1,
      "source_type": "text",
      "main_topic": "Introduction to AI",
      "key_points": ["Point 1", "Point 2"],
      "terms_to_remember": ["AI", "Machine Learning"],
      "exam_tip": "Focus on definitions.",
      "one_line_recap": "AI is the simulation of human intelligence.",
      "raw_text": "..."
    }
  ],
  "page_count": 1
}
```

## 6. Access and Testing
A Postman collection is attached in the `/postman` directory to speed up request prototyping. Import `PDF_Summarization_API.postman_collection.json` alongside the environment keys to validate endpoints locally.
