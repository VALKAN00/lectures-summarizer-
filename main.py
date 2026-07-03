import os
import sys
import logging
import re
from pathlib import Path
import torch
import pdfplumber
from pdf2image import convert_from_path
import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
    # Uncomment and set the path if tesseract is not in system PATH (Windows)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("pytesseract not installed. OCR preprocessing disabled. Install with: pip install pytesseract pillow")

# --- CONFIGURATION ---
CONFIG = {
    "model_name": "t5-base",
    "max_source_len": 768,
    "max_target_len": 250,
    "word_threshold": 30,  # Pages with <= this many words use vision model
    "dpi": 200,  # Increased for better image quality
    "temp_dir": "temp_slides",
    "output_file": "summary_output.txt",
    "ollama_model": "llava",
    "min_beams": 6,  # More beams for better quality
    "length_penalty": 2.0,  # Encourage longer, more complete summaries
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 2.0,  # Stronger penalty against repetition
    "min_summary_length": 20,  # Longer minimum to avoid truncated thoughts
    "temperature": 0.3,  # Lower temperature for more conservative/factual generation
    "do_sample": False,  # Deterministic generation
    "early_stopping": True,
    "use_ocr": True,  # Enable OCR preprocessing for better text extraction
    "ocr_confidence_threshold": 60,  # Minimum OCR confidence (0-100)
    "student_friendly": True,  # Format output for students with structure
    "preserve_structure": True,  # Keep headings, bullets, and sections
    "detect_diagrams": True,  # Detect pages with diagrams/flowcharts
    "diagram_curve_threshold": 3  # Min number of curves to consider it a diagram
}

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('summarization.log')
    ]
)
logger = logging.getLogger(__name__)

# --- 1. SETUP T5 FOR TEXT ---
MODEL_NAME = CONFIG["model_name"]
MAX_SOURCE_LEN = CONFIG["max_source_len"]
MAX_TARGET_LEN = CONFIG["max_target_len"]

def initialize_models():
    """Initialize T5 model and tokenizer with error handling.
    
    Returns:
        tuple: (tokenizer, model, device) or (None, None, None) if failed
    """
    try:
        logger.info(f"Loading T5 model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        logger.info(f"✅ T5 Text Model loaded on: {device}")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to initialize T5 model: {e}")
        return None, None, None

tokenizer, model, device = initialize_models()
if model is None:
    logger.critical("Cannot proceed without T5 model. Exiting.")
    sys.exit(1)

# --- 2. VALIDATION FUNCTIONS ---
def validate_pdf_path(pdf_path):
    """Validate that the PDF file exists and is readable.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bool: True if valid, False otherwise
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    if not path.is_file():
        logger.error(f"Path is not a file: {pdf_path}")
        return False
    if path.suffix.lower() != '.pdf':
        logger.warning(f"File may not be a PDF: {pdf_path}")
    return True

def check_ollama_server():
    """Check if Ollama server is accessible.
    
    Returns:
        bool: True if server is reachable, False otherwise
    """
    try:
        # Try to list models as a health check
        ollama.list()
        logger.info("✅ Ollama server is accessible")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama server: {e}")
        logger.error("Please start Ollama server before running this script.")
        return False

def extract_text_with_ocr(image_path):
    """Extract text from image using OCR (pytesseract).
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text or empty string if failed
    """
    if not PYTESSERACT_AVAILABLE:
        return ""
    
    try:
        img = Image.open(image_path)
        # Use pytesseract with detailed output for confidence filtering
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Filter by confidence threshold and reconstruct text
        filtered_text = []
        for i, conf in enumerate(ocr_data['conf']):
            if int(conf) > CONFIG["ocr_confidence_threshold"]:
                text = ocr_data['text'][i]
                if text.strip():
                    filtered_text.append(text)
        
        result = ' '.join(filtered_text)
        logger.info(f"OCR extracted {len(filtered_text)} high-confidence words")
        return result
        
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""

def preprocess_page_text(page, page_num, pdf_path):
    """Extract and preprocess text from a PDF page with OCR fallback.
    
    Args:
        page: pdfplumber page object
        page_num (int): Page number
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Preprocessed and cleaned text
    """
    # Primary extraction with pdfplumber
    text_pdfplumber = page.extract_text() or ""
    word_count_plumber = len(text_pdfplumber.split())
    
    # If OCR is enabled and pdfplumber extraction is poor, use OCR
    if CONFIG["use_ocr"] and PYTESSERACT_AVAILABLE and word_count_plumber < 20:
        logger.info(f"Page {page_num}: pdfplumber extracted only {word_count_plumber} words. Using OCR...")
        
        # Convert page to image for OCR
        temp_dir = Path(CONFIG["temp_dir"])
        temp_dir.mkdir(exist_ok=True)
        ocr_img_path = temp_dir / f"ocr_page_{page_num}.png"
        
        try:
            page_images = convert_from_path(
                pdf_path,
                dpi=CONFIG["dpi"],
                first_page=page_num,
                last_page=page_num
            )
            page_images[0].save(ocr_img_path, 'PNG')
            
            text_ocr = extract_text_with_ocr(str(ocr_img_path))
            word_count_ocr = len(text_ocr.split())
            
            # Clean up OCR image
            if ocr_img_path.exists():
                ocr_img_path.unlink()
            
            # Use OCR text if it extracted more content
            if word_count_ocr > word_count_plumber:
                logger.info(f"OCR improvement: {word_count_plumber} → {word_count_ocr} words")
                return text_ocr
            else:
                logger.info(f"pdfplumber extraction was better: {word_count_plumber} words")
                return text_pdfplumber
                
        except Exception as e:
            logger.warning(f"OCR preprocessing failed: {e}")
            return text_pdfplumber
    
    return text_pdfplumber

def detect_visual_layout(page):
    """Detect if a page has visual elements like diagrams, flowcharts, or special layouts.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        tuple: (has_diagram, description)
    """
    if not CONFIG["detect_diagrams"]:
        return False, ""
    
    try:
        # 1. Check for curves/lines (arrows, connectors in diagrams)
        curves = page.curves if hasattr(page, 'curves') else []
        lines = page.lines if hasattr(page, 'lines') else []
        rects = page.rects if hasattr(page, 'rects') else []
        
        # 2. Check for images
        images = page.images
        
        # 3. Analyze text positioning - diagrams have scattered text
        chars = page.chars
        if len(chars) > 0:
            # Get y-positions of text
            y_positions = [char['y0'] for char in chars]
            # Calculate how spread out the text is vertically
            y_spread = max(y_positions) - min(y_positions) if y_positions else 0
            
            # Get x-positions to check horizontal spread
            x_positions = [char['x0'] for char in chars]
            x_spread = max(x_positions) - min(x_positions) if x_positions else 0
            
            # Diagrams typically have text scattered in 2D space
            is_scattered = y_spread > 200 and x_spread > 300
        else:
            is_scattered = False
        
        # Detect diagrams based on multiple criteria
        has_many_curves = len(curves) >= CONFIG["diagram_curve_threshold"]
        has_many_lines = len(lines) >= 5
        has_shapes = len(rects) >= 3
        
        reasons = []
        if has_many_curves:
            reasons.append(f"{len(curves)} curves/arrows")
        if has_many_lines:
            reasons.append(f"{len(lines)} lines")
        if has_shapes:
            reasons.append(f"{len(rects)} shapes")
        if is_scattered:
            reasons.append("scattered text layout")
        if images:
            reasons.append(f"{len(images)} images")
        
        # Consider it a diagram if it has multiple visual indicators
        is_diagram = (has_many_curves or has_many_lines or has_shapes or 
                     (is_scattered and len(images) > 0))
        
        description = ", ".join(reasons) if reasons else "no visual elements"
        return is_diagram, description
        
    except Exception as e:
        logger.warning(f"Error detecting visual layout: {e}")
        return False, ""

def parse_document_structure(text):
    """Parse text to identify structure: headings, bullets, sections.
    
    Args:
        text (str): Raw text from PDF
        
    Returns:
        dict: Structured representation with sections and content
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    structure = {
        'sections': [],
        'has_structure': False
    }
    
    # Common heading indicators
    heading_keywords = ['topics:', 'grading:', 'textbook:', 'outline:', 'course:', 
                       'objectives:', 'introduction:', 'summary:', 'references:',
                       'agenda:', 'overview:', 'goals:', 'requirements:']
    
    current_section = None
    
    for line in lines:
        line_lower = line.lower()
        
        # Check if it's a heading
        is_heading = any(line_lower.startswith(kw) or line_lower == kw.rstrip(':') 
                        for kw in heading_keywords)
        
        # Also check for ALL CAPS headings (common in slides)
        if not is_heading and line.isupper() and len(line.split()) <= 5:
            is_heading = True
        
        if is_heading:
            structure['has_structure'] = True
            current_section = {
                'heading': line,
                'content': []
            }
            structure['sections'].append(current_section)
        elif current_section:
            current_section['content'].append(line)
        else:
            # No section yet, create a default one
            if not structure['sections']:
                structure['sections'].append({
                    'heading': None,
                    'content': []
                })
            structure['sections'][0]['content'].append(line)
    
    return structure

def format_for_students(text, page_structure=None):
    """Format text in a student-friendly way with proper structure.
    
    Args:
        text (str): Text to format
        page_structure (dict): Parsed structure from parse_document_structure
        
    Returns:
        str: Formatted, student-friendly text
    """
    if not CONFIG["student_friendly"]:
        return text
    
    if page_structure and page_structure['has_structure']:
        # Format with preserved structure
        formatted_parts = []
        
        for section in page_structure['sections']:
            if section['heading']:
                formatted_parts.append(f"\n📌 **{section['heading'].title()}**")
            
            # Identify and format bullets
            content_items = section['content']
            if content_items:
                # Check if items look like a list
                if len(content_items) > 2:
                    formatted_parts.append("")
                    for i, item in enumerate(content_items, 1):
                        # Clean up the item
                        item = item.strip('•*-›→')
                        if item:
                            # Use bullets for short items, numbers for longer lists
                            if len(content_items) <= 6:
                                formatted_parts.append(f"   • {item}")
                            else:
                                formatted_parts.append(f"   {i}. {item}")
                else:
                    # Regular paragraph
                    formatted_parts.append(f"   {' '.join(content_items)}")
        
        return '\n'.join(formatted_parts)
    else:
        # No clear structure, just make it more readable
        sentences = [s.strip() for s in text.replace('. ', '.\n').split('\n') if s.strip()]
        if len(sentences) > 3:
            # Use bullet points for multiple items
            return '\n'.join([f"• {s}" for s in sentences])
        else:
            return text

def postprocess_student_summary(text):
    """Clean and normalize model output for easier student reading.

    Args:
        text (str): Raw model output

    Returns:
        str: Cleaned, student-friendly output
    """
    if not text:
        return ""

    cleaned = text.replace("\\n", "\n").replace("\r\n", "\n")

    # Convert visual bullets/markdown to cleaner plain-text notes.
    cleaned = cleaned.replace("•", "-")
    cleaned = cleaned.replace("📌", "")
    cleaned = cleaned.replace("**", "")

    # Keep whitespace compact and readable.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)

    return cleaned.strip()

# --- 3. SUMMARIZE TEXT (T5) ---
def summarize_text(text, preserve_structure=True):
    """Summarize text using T5 model with optional structure preservation.
    
    Args:
        text (str): Input text to summarize
        preserve_structure (bool): Whether to preserve document structure
        
    Returns:
        str: Summary text or error message
    """
    try:
        if not text.strip():
            return "[Empty text - no summary generated]"
        
        word_count = len(text.split())
        
        # Parse structure if enabled
        structure = None
        if preserve_structure and CONFIG["preserve_structure"]:
            structure = parse_document_structure(text)
        
        # For structured content (like course outlines), use extractive approach
        if structure and structure['has_structure']:
            logger.info("Structured content detected - using extractive approach")
            formatted = format_for_students(text, structure)
            return formatted
        
        # For very short texts, use extractive approach instead
        if word_count < 50:
            # Return key sentences instead of generating
            sentences = text.split('.')
            important_sentences = [s.strip() for s in sentences if len(s.split()) > 3][:2]
            if important_sentences:
                return '. '.join(important_sentences) + '.'
            
        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LEN
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=MAX_TARGET_LEN,
                min_length=CONFIG["min_summary_length"],
                num_beams=CONFIG["min_beams"],
                length_penalty=CONFIG["length_penalty"],
                no_repeat_ngram_size=CONFIG["no_repeat_ngram_size"],
                repetition_penalty=CONFIG["repetition_penalty"],
                temperature=CONFIG["temperature"] if CONFIG["do_sample"] else 1.0,
                do_sample=CONFIG["do_sample"],
                early_stopping=CONFIG["early_stopping"]
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Make summary student-friendly
        if CONFIG["student_friendly"]:
            summary = format_for_students(summary)
            summary = postprocess_student_summary(summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in text summarization: {e}")
        return f"[Error generating summary: {str(e)}]"

# --- 4. SUMMARIZE IMAGES/GRAPHS (LLaVA via Ollama) ---
def summarize_image(image_path, extracted_text=""):
    """Summarize an image using LLaVA model via Ollama with grounding in extracted text.
    
    Args:
        image_path (str): Path to the image file
        extracted_text (str): Text extracted from the page to ground the vision model
        
    Returns:
        str: Summary of the image or error message
    """
    logger.info(f"🧠 Passing {image_path} to LLaVA...")
    try:
        if not Path(image_path).exists():
            return f"[Error: Image file not found: {image_path}]"
        
          # Student-friendly structured prompt
        if extracted_text.strip():
                prompt = f"""You are creating study notes for university students from a lecture slide.

Write in clear and simple language suitable for a student who is learning this topic for the first time.

Text extracted from this slide:
{extracted_text[:500]}

Output format (plain text only, no markdown, no JSON):
Main topic:
Key points:
- point 1
- point 2
Terms to remember:
- term 1
- term 2
Exam tip:
One-line recap:

IMPORTANT INSTRUCTIONS FOR DIAGRAMS:
- Preserve original numbering when visible.
- For flowcharts/concept maps, explain how items connect.
- Be factual and avoid adding information that is not visible.

Keep it concise and study-ready."""
        else:
                prompt = """You are creating study notes for university students from a lecture slide.

Write in clear and simple language suitable for a student who is learning this topic for the first time.

Output format (plain text only, no markdown, no JSON):
Main topic:
Key points:
- point 1
- point 2
Terms to remember:
- term 1
- term 2
Exam tip:
One-line recap:

CRITICAL INSTRUCTIONS FOR DIAGRAMS:
- If you see arrows, describe those connections clearly.
- Keep visible numbering exactly as shown.
- Preserve diagram meaning instead of turning everything into a simple list.
- Be factual and avoid adding information that is not visible.

Keep it concise and study-ready."""
            
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }],
            options={
                'temperature': 0.1,  # Very low temperature for factual output
                'top_p': 0.9,
                'top_k': 40
            }
        )
        return postprocess_student_summary(response['message']['content'])
    except Exception as e:
        logger.error(f"Error in image summarization: {e}")
        return f"[Error summarizing image: {str(e)}]"

# --- 5. MAIN PIPELINE ---
def process_pdf(pdf_path):
    """Process a PDF file and generate summaries for each page.
    
    Uses T5 for text-heavy pages and LLaVA for diagram-heavy pages.
    Implements lazy image conversion - only converts pages that need vision processing.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Combined summary of all pages
    """
    final_output = []
    temp_dir = Path(CONFIG["temp_dir"])
    temp_dir.mkdir(exist_ok=True)
    
    try:
        logger.info(f"Opening PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages...")
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing Page {page_num}/{total_pages}...")
                
                try:
                    # Use OCR-enhanced text extraction
                    text = preprocess_page_text(page, page_num, pdf_path)
                    word_count = len(text.split())
                    
                    # Check if page has images/tables for smarter routing
                    has_images = len(page.images) > 0
                    has_tables = len(page.extract_tables()) > 0
                    
                    # Detect diagrams and visual layouts
                    is_diagram, diagram_info = detect_visual_layout(page)
                    
                    # SMARTER ROUTER: Consider text, images, tables, AND visual layout
                    use_vision = False
                    reason = ""
                    
                    if is_diagram:
                        use_vision = True
                        reason = f"diagram/flowchart detected ({diagram_info})"
                    elif word_count <= CONFIG["word_threshold"]:
                        use_vision = True
                        reason = f"low word count ({word_count} words)"
                    elif has_images and word_count < 100:
                        use_vision = True
                        reason = f"contains images with moderate text ({word_count} words)"
                    elif has_tables and word_count < 80:
                        use_vision = True
                        reason = f"contains tables with moderate text ({word_count} words)"
                    else:
                        reason = f"text-heavy ({word_count} words)"
                    
                    if not use_vision:
                        logger.info(f"📄 Page {page_num}: Using T5 - {reason}")
                        summary = summarize_text(text, preserve_structure=True)
                        
                        # Add student-friendly header
                        if CONFIG["student_friendly"]:
                            final_output.append(f"\n{'='*60}")
                            final_output.append(f"PAGE {page_num} - Student Notes (Text)")
                            final_output.append(f"{'='*60}\n")
                            final_output.append(summary)
                            final_output.append("")
                        else:
                            final_output.append(f"📄 Page {page_num} (Text Summary):\n{summary}\n")
                    else:
                        logger.info(f"📊 Page {page_num}: Using LLaVA - {reason}")
                        # LAZY IMAGE CONVERSION: Only convert this specific page
                        img_path = temp_dir / f"page_{page_num}.png"
                        try:
                            # Convert only the current page (pages are 1-indexed in convert_from_path)
                            page_images = convert_from_path(
                                pdf_path, 
                                dpi=CONFIG["dpi"],
                                first_page=page_num,
                                last_page=page_num
                            )
                            page_images[0].save(img_path, 'PNG')
                            
                            # Pass extracted text to ground the vision model and reduce hallucinations
                            summary = summarize_image(str(img_path), extracted_text=text)
                            
                            # Add student-friendly header
                            if CONFIG["student_friendly"]:
                                final_output.append(f"\n{'='*60}")
                                final_output.append(f"PAGE {page_num} - Student Notes (Visual)")
                                final_output.append(f"{'='*60}\n")
                                final_output.append(summary)
                                final_output.append("")
                            else:
                                final_output.append(f"📊 Page {page_num} (Vision Summary):\n{summary}\n")
                        finally:
                            # Clean up the image immediately to save space
                            if img_path.exists():
                                try:
                                    img_path.unlink()
                                except Exception as e:
                                    logger.warning(f"Could not delete {img_path}: {e}")
                                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    final_output.append(f"❌ Page {page_num}: Error - {str(e)}\n")
                    continue

    except Exception as e:
        logger.error(f"Critical error processing PDF: {e}")
        return f"❌ Failed to process PDF: {str(e)}"
    
    finally:
        # CLEANUP: Ensure temp directory is cleaned up
        try:
            if temp_dir.exists():
                for file in temp_dir.glob('*'):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete {file}: {e}")
                # Try to remove the directory itself
                try:
                    temp_dir.rmdir()
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove temp directory: {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    return "\n".join(final_output)

# --- 6. EXECUTION ---
if __name__ == "__main__":
    # Get PDF path from command line or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    
    logger.info("="*60)
    logger.info("HYBRID PDF SUMMARIZATION PIPELINE")
    logger.info("="*60)
    
    # Validate inputs
    if not validate_pdf_path(pdf_path):
        logger.critical("PDF validation failed. Exiting.")
        sys.exit(1)
    
    if not check_ollama_server():
        logger.critical("Ollama server check failed. Exiting.")
        sys.exit(1)
    
    # Process the PDF
    try:
        result = process_pdf(pdf_path)
        
        # Save to file
        output_file = Path(CONFIG["output_file"])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FINAL HYBRID SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(result)
        
        logger.info("\n" + "="*60)
        logger.info("FINAL HYBRID SUMMARY")
        logger.info("="*60 + "\n")
        print(result)
        logger.info(f"\n✅ Summary saved to: {output_file.absolute()}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)