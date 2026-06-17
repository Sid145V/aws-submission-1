import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from backend.utils.logger import get_logger

logger = get_logger("pdf_loader")

class PDFLoader:
    """
    Service to load and extract text from PDF files, preserving page numbers 
    and tracking section hierarchy.
    """
    
    # Regex to detect section headers (e.g., "1. Use of the Services.", "Section 7. Term and Termination.")
    SECTION_PATTERN = re.compile(
        r"^\s*(?:Section\s+)?(\d+)\.\s+([A-Z][a-zA-Z\s,;&()\-–/]+)(?:\.|\b)",
        re.MULTILINE
    )

    @classmethod
    def load_pdf(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads PDF, cleans text page by page, and detects sections.
        Returns a list of dicts: [ { "page_num": int, "text": str, "sections": List[str] }, ... ]
        """
        logger.info(f"Starting PDF extraction for: {file_path}")
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF file at {file_path}: {e}")
            raise FileNotFoundError(f"PDF file not found at {file_path}") from e

        extracted_pages = []
        current_section = "Preamble"
        
        for i in range(len(doc)):
            page = doc.load_page(i)
            page_text = page.get_text("text")
            page_num = i + 1
            
            # Clean text (remove invalid characters, fix double whitespaces/lines)
            cleaned_text = cls._clean_text(page_text)
            
            # Identify section headers on this page
            sections_found = cls._detect_sections(cleaned_text)
            
            # If section headers are found on this page, update current_section
            if sections_found:
                current_section = sections_found[-1]
                
            extracted_pages.append({
                "page_num": page_num,
                "text": cleaned_text,
                "current_section": current_section,
                "sections_on_page": sections_found
            })
            
        logger.info(f"Successfully extracted {len(extracted_pages)} pages from {file_path}")
        return extracted_pages

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """
        Applies basic text cleaning to improve text consistency and chunking quality.
        """
        if not text:
            return ""
        
        # Replace multiple spaces with a single space
        text = re.sub(r"[ \t]+", " ", text)
        
        # Replace multiple newlines with double newlines (to preserve paragraphs)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        
        # Replace standard quotes and single dashes
        text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        
        # Fix split words at end of line (e.g. con- \n tract -> contract)
        text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
        
        return text.strip()

    @classmethod
    def _detect_sections(cls, text: str) -> List[str]:
        """
        Helper method to detect section headers on a given page text.
        """
        sections = []
        # Find all lines matching the section pattern
        for match in cls.SECTION_PATTERN.finditer(text):
            sec_num = match.group(1)
            sec_name = match.group(2).strip()
            
            # Filter out common false positives (like date numbers, page numbers, or short uppercase strings)
            if len(sec_name) > 3 and not sec_name.upper().startswith("PAGE"):
                sections.append(f"Section {sec_num}: {sec_name}")
                
        return sections
