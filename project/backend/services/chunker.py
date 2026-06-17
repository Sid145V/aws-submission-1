from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.utils.logger import get_logger

logger = get_logger("chunker")

class DocumentChunker:
    """
    Service to split extracted PDF pages into semantic chunks of a fixed size and overlap,
    while enriching chunks with metadata to preserve the section hierarchy.
    """
    
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    @classmethod
    def chunk_pages(
        cls, 
        pages: List[Dict[str, Any]], 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[Document]:
        """
        Splits extracted pages into chunks page-by-page to ensure clear page-level citations.
        Enriches each chunk's content with section information for improved embedding matching.
        """
        logger.info(f"Chunking {len(pages)} pages with size={chunk_size}, overlap={chunk_overlap}")
        
        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        all_chunks = []
        
        for page in pages:
            page_num = page["page_num"]
            page_text = page["text"]
            current_section = page["current_section"]
            
            if not page_text.strip():
                continue
            
            # Split the page text
            splits = text_splitter.split_text(page_text)
            
            for idx, split_text in enumerate(splits):
                # Build context header to prepend to the chunk content.
                # This injects section context into the vector space and LLM context.
                context_header = f"[AWS Customer Agreement | {current_section} | Page {page_num}]\n"
                enriched_content = context_header + split_text
                
                # Create LangChain Document
                doc = Document(
                    page_content=enriched_content,
                    metadata={
                        "page_num": page_num,
                        "section": current_section,
                        "chunk_index": idx,
                        "raw_content": split_text  # store the raw text without prepended context
                    }
                )
                all_chunks.append(doc)
                
        logger.info(f"Created a total of {len(all_chunks)} chunks from PDF pages.")
        return all_chunks
