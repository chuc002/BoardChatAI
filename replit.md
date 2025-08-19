# Board Continuity MVP

## Overview
Board Continuity MVP is a document intelligence application enabling users to upload PDF documents and interact with them via AI-powered chat. It functions as a knowledge base system, processing PDFs to extract text, chunk content for retrieval, generate embeddings for semantic search, and provide natural language responses with contextual citations. The system aims to provide intelligent answers to queries about document content, citing relevant sections and page numbers. The overarching vision is to provide perfect recall with 30-year veteran wisdom, comprehensive validation, system integrity monitoring, and complete institutional intelligence synthesis for board-related decisions.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture
The system employs a 5-layer architecture: Ingestion (100% capture), Memory (perfect storage), Intelligence (30-year wisdom), Synthesis (perfect responses), and Validation (100% accuracy). It handles multi-format processing (PDFs, images, audio, video) with perfect extraction. A Perfect Recall Engine coordinates subsystems for veteran-level responses with institutional context. System Integrity Monitoring ensures continuous validation across 8 subsystems. Comprehensive analysis combines pattern recognition, knowledge graphs, governance intelligence, memory synthesis, and perfect RAG. A Validation Framework performs fact-checking, consistency verification, completeness scoring, and accuracy measurement.

### Frontend Architecture
The frontend features a NotebookLM-inspired interface built with Bootstrap 5 and a custom design system, emphasizing an AI-first conversational approach. The primary interface is a chat-driven AI assistant. HTMX provides dynamic interactions, enhancing elements like auto-resizing textareas and real-time message rendering. UI components include a chat-first layout with a sidebar for sources/upload, styled with modern gradient themes and a professional purple-blue AI branding. The user experience focuses on conversational document exploration, intelligent suggestions, confidence indicators, and seamless source citations.

### Backend Architecture
Built with Flask, the backend uses a modular route organization and a service layer pattern to separate business logic. Flask-Login manages user sessions (though bypassed in current MVP). File handling uses Werkzeug for secure processing, supporting multi-file uploads up to 16MB. Comprehensive logging and user-friendly error messages are implemented. Persistent Q&A storage preserves full citation metadata. A background worker (worker.py) handles asynchronous document ingestion, with real-time job status tracking via API endpoints.

### Data Storage Solutions
Supabase (PostgreSQL) is the primary database for document metadata, chat history, and institutional memory models. Document storage uses the local file system. Vector storage is embedded within Supabase for semantic search. Core data models include Document, DocumentChunk, and QA_History, with org_id/created_by relationships. Institutional Memory Models capture decision registries, historical patterns, institutional knowledge, and board member insights for comprehensive organizational recall and cross-referential analytics.

### Authentication and Authorization
In the current MVP development mode, authentication uses environment variables (DEV_ORG_ID, DEV_USER_ID), bypassing traditional login/registration. All document and chat operations are scoped to a development organization, allowing open access to all routes for simplified development.

### Document Processing Pipeline
PDF extraction utilizes PyPDF2 for text extraction with page-level granularity. A section-aware chunking system (lib/enhanced_ingest.py) employs regex patterns to preserve complete policy sections and prevent mid-sentence breaks of critical information, ensuring intelligent chunking that respects contextual boundaries rather than arbitrary character limits. Percentage sequences are intelligently detected and preserved. Each chunk receives a `section_completeness_score`. Chunks are validated to prevent ending with incomplete numbers, percentages, or conjunctions. OpenAI `text-embedding-3-small` generates 1536-dimensional embeddings. GPT-4o-mini is used for pre-computed chunk summaries. Enhanced chunk storage includes section metadata, percentage lists, and completeness scores. SHA256-based file deduplication prevents reprocessing identical documents.

### AI Integration
OpenAI GPT-4o is the primary language model for chat responses, with automatic fallback to GPT-4o-mini. Strict token budgeting is implemented with configurable limits (e.g., 2400 summary tokens, 4800 final prompt tokens). Retrieval uses vector similarity search with MMR reranking and a hybrid keyword fallback. MMR (Maximal Marginal Relevance) balances relevance and diversity. Graceful degradation ensures a fallback to keyword search if the vector index is unavailable. Real-time performance logging tracks retrieval and processing steps. Context assembly uses a summary-first approach. Response generation includes retry logic with exponential backoff and automatic model downgrading. Citation management de-duplicates sources with document titles and page-specific deep links.

## External Dependencies

### Core Services
- **Supabase**: PostgreSQL database hosting and management.
- **OpenAI API**: Provides GPT models (GPT-4o, GPT-4o-mini) for chat completion and text-embedding-3-small for embeddings.

### Python Libraries
- **Flask**: Web framework.
- **Flask-Login**: User session management.
- **PyPDF2**: PDF text extraction.
- **Supabase Python Client**: Database interaction and vector search.
- **OpenAI Python SDK**: AI model interactions.
- **Werkzeug**: Secure filename processing.

### Frontend Dependencies
- **Bootstrap 5**: UI framework.
- **Font Awesome**: Icon library.
- **HTMX**: Dynamic HTML enhancement.

### Environment Configuration (Required Secrets)
- **OPENAI_API_KEY**
- **SUPABASE_URL**
- **SUPABASE_SERVICE_ROLE**
- **DATABASE_URL**
- **SESSION_SECRET**
```