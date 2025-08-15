# Page Number Tracking for Deep-Linking

## Implementation Status ✅

### Database Schema
- Added `page_index` column to `doc_chunks` table
- **Manual step required**: Run this SQL in Supabase SQL editor:
  ```sql
  ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS page_index int;
  ```

### PDF Processing Enhancement
- **Page Extraction**: `extract_pages_from_pdf()` now returns `[(text, page_number), ...]`
- **Smart Chunking**: `smart_chunks_with_pages()` preserves page numbers from source tokens
- **Page Inheritance**: Each chunk inherits the page number from its first token

### RAG System Integration
- **Metadata Collection**: Page numbers included in citation metadata
- **Future Deep-Linking**: Ready for PDF viewer integration with `#page=N` URLs

### Frontend Display
- **Citation Enhancement**: Shows "(page X)" when page numbers available
- **Example**: `[1] Board Minutes 2024 (page 3) — open`

## Usage Notes

### For New Documents
- All new PDF uploads will automatically capture page numbers
- Each chunk stores the page from its first token
- Multi-page chunks show the starting page number

### Future Deep-Linking
When hosting PDFs with a viewer that supports `#page=N`:
```javascript
// Example: PDF.js integration
const pdfUrl = meta.url + `#page=${meta.page_index}`;
```

### Example Output
**Before**: `[1] Board Meeting Minutes 2024-03-15.pdf — open`
**After**: `[1] Board Meeting Minutes 2024-03-15.pdf (page 3) — open`

## Technical Details

### Page Number Heuristic
- Uses PyPDF2 page-by-page extraction for accurate page tracking
- Fallback to pdfminer assumes single page (page 1)
- Chunks inherit page number from first token position
- 1-based page numbering (page 1, 2, 3...)

### Token-Level Precision
- Builds complete token sequence with page mapping
- Each token knows its source page
- Chunk boundary at token 2,847 → gets page from token 2,847
- Preserves overlap while maintaining page accuracy