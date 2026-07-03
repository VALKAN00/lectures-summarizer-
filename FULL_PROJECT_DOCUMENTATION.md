# 📚 Hybrid PDF Summarization System - Full Documentation

This document is designed to give you a complete, top-to-bottom understanding of how your project works. Use this to prepare for the discussion with your doctor/supervisor. 

---

## 1. Project Overview
**What does this project do?**
It is a locally-hosted, AI-powered pipeline that takes a PDF document (like a university lecture slide deck) and converts it into structured, easy-to-read "Student Notes". 

**What is the core innovation?**
Instead of blindly sending everything to one AI model, your project uses a **"Smart Hybrid Router"**. It analyzes each page of the PDF individually and decides the best, most efficient way to summarize it:
1. **Text-heavy pages** are routed to a fast Text Model (**T5**).
2. **Diagram or image-heavy pages** are routed to a Vision Model (**LLaVA via Ollama**).

---

## 2. System Architecture & Workflow
When a user uploads a PDF, the system follows this exact step-by-step process (found in `main.py`):

1. **PDF Parsing:** The system opens the PDF using `pdfplumber`. It loops through the document page by page.
2. **Text Extraction & OCR:** It tries to extract text. If the text count is extremely low (under 20 words), it assumes the page might be a scanned image and falls back to **Tesseract OCR** to pull the text from the image.
3. **Visual Layout Detection:** It checks if the page has diagrams (by counting PDF curves, lines, rects, and scattered text).
4. **The Router:** Based on the extracted text and visual layout, the router decides: *T5 or LLaVA?*
5. **Summarization:** The chosen model generates the summary.
6. **Post-Processing:** The raw summary is cleaned and formatted into a specific "Student Note" structure.
7. **JSON Structuring (API):** The FastAPI backend (`api.py`) parses this text and converts it into a clean JSON response for front-end web applications.

---

## 3. The "Smart Router" (How it decides)
This is exactly how the system intelligently routes pages (in `main.py -> process_pdf`):

*   **Rule 1 - Diagrams:** If `detect_visual_layout()` finds > 3 curves, > 5 lines, or text scattered everywhere in 2D space, it flags the page as a diagram. -> **Routes to LLaVA (Vision)**
*   **Rule 2 - Low Word Count:** If the page has <= 30 words. -> **Routes to LLaVA (Vision)**
*   **Rule 3 - Has Images/Tables:** If the page has images and < 100 words, or tables and < 80 words. -> **Routes to LLaVA (Vision)**
*   **Rule 4 - Text Heavy:** If none of the above are true, the page is mostly text. -> **Routes to T5 (Text)**

---

## 4. The AI Models Explained

### A. The Text Model (T5)
*   **What it is:** T5 (Text-to-Text Transfer Transformer) by Google, loaded via HuggingFace `transformers`.
*   **Why use it:** It's very fast, runs locally on the CPU/GPU, and excels at standard text summarization.
*   **How it's controlled:** We use specific algorithms to ensure quality:
    *   *Beam Search (`min_beams=6`):* Explores multiple possible sentences before picking the best one to ensure high quality.
    *   *Repetition Penalty (`repetition_penalty=2.0`):* Prevents the model from repeating the same sentence forever.

### B. The Vision Model (LLaVA)
*   **What it is:** A Large Language-and-Vision Assistant, served via **Ollama**.
*   **Why use it:** T5 cannot "see" images. LLaVA can analyze flowcharts, graphs, and visual relationships. 
*   **The "Text Grounding" Trick:** To prevent LLaVA from hallucinating (making things up), we pass the text we extracted via OCR *into the prompt* alongside the image. This forces LLaVA to stay factual to the actual slide.

---

## 5. The API Backend (`api.py`)

You built a **FastAPI** backend to expose this pipeline to the web. 
It has specialized endpoints for different use cases:
*   `POST /summarize/upload`: Takes a physical PDF upload, runs the pipeline, and returns the raw text summary separated by page.
*   `POST /summarize/upload/structured`: Does the same thing, but it runs the text through `parse_structured_page()`. This function uses Regular Expressions (Regex) to extract the "Main Topic", "Key Points", "Terms", and "Exam Tip" strictly into a JSON dictionary so a React frontend can easily render it as UI cards.

---

## 6. Key Optimizations (Important for your discussion)

Your doctor will want to know *why* your code is good. Mention these optimizations:

1. **Lazy Image Conversion:** Converting PDFs to images takes a lot of RAM. Your code *only* converts a page to an image IF the router decides it needs the Vision model. If it routes to T5, no image is generated. This saves massive amounts of memory.
2. **OCR Fallback:** Instead of failing on scanned PDFs, the system automatically detects low word counts and activates Tesseract OCR to read text from pixels.
3. **Student-Centric Prompting:** Instead of general summaries, the system prompts the Vision model specifically to output "Main topics", "Terms to remember", and "Exam tips". 
4. **Deterministic Generation:** For the Vision model, you set the `temperature` to `0.1`. This makes the model highly analytical and factual, avoiding creative hallucinations.

---

## 7. Q&A: Prepare for your Doctor's Questions

**Q: Why didn't you just send the whole PDF to ChatGPT?**
*Answer:* "Privacy, cost, and efficiency. By running models locally (T5 and Ollama/LLaVA), sensitive university documents don't leave the user's computer. Furthermore, using a small, fast model (T5) for text pages is highly resource-efficient, saving the heavy Vision model only for pages that actually have diagrams."

**Q: How does the system know what is a diagram?**
*Answer:* "We use `pdfplumber` to analyze the hidden geometry of the PDF. We look at the coordinates of text (is it scattered?), and we count the number of vector lines, curves, and rectangles on the page. If it crosses our threshold, we flag it as a visual layout."

**Q: What happens if the text extraction fails?**
*Answer:* "The system monitors the word count extracted by `pdfplumber`. If the word count drops below 20, it suspects the page is a flat, scanned image. It automatically triggers `pytesseract` (OCR) to dynamically pull the text from the pixels."

**Q: How do you prevent the Vision model (LLaVA) from hallucinating information on complex graphs?**
*Answer:* "We use a technique called 'Text Grounding'. Before sending the image to LLaVA, we extract any available text from the page. We inject that text directly into LLaVA's prompt and explicitly command it: 'Be factual and avoid adding information that is not visible'. This anchors the visual analysis to grounded text."

**Q: How does the frontend handle the data?**
*Answer:* "Instead of returning a giant wall of text, `api.py` includes a custom parser that reads the generated notes, breaks them up by page, and converts the bullet points into a strictly typed JSON object (Arrays for key points, Strings for main topics). This makes it trivial to bind the data to an interactive UI like React or Vue."
