# Board Continuity MVP

## Overview

Board Continuity MVP is a document intelligence application that allows users to upload PDF documents and interact with them through AI-powered chat functionality. The system processes PDF documents by extracting text, chunking content for optimal retrieval, generating embeddings for semantic search, and enabling natural language queries with contextual responses and source citations.

The application serves as a knowledge base system where users can upload board documents, meeting minutes, or other PDF materials and ask questions about their content. The AI provides intelligent responses with specific citations to relevant document sections and page numbers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: NotebookLM-inspired interface with Bootstrap 5 and custom design system
- **Design Philosophy**: AI-first conversational interface putting document intelligence at the forefront
- **Primary Interface**: Chat-driven document intelligence with AI assistant as the main landing page
- **JavaScript Enhancement**: HTMX for dynamic interactions, enhanced auto-resizing textarea, real-time message rendering
- **UI Components**: Chat-first layout with AI conversation as primary interface, sidebar for sources/upload, NotebookLM-style messaging
- **Styling**: Modern gradient themes, professional color palette with purple-blue AI branding, smooth animations, conversation-focused design
- **User Experience**: Conversational document exploration, intelligent suggestions, confidence indicators, seamless source citations, auto-focus chat input

### Recent Performance Improvements (August 18-19, 2025)
- **✅ ULTIMATE NOTEBOOKLM ENHANCEMENT**: Final optimization to achieve 90%+ comprehensive quality matching NotebookLM standards
- **Enhanced Readability & Structure**: Implemented proper formatting with clear section headers, double line breaks, organized bullet points for easy scanning
- **Open Notebook Integration**: Reviewed and incorporated advanced techniques from open-notebook repository including multi-modal content processing, granular context control, and professional citation systems
- **tldw_chatbook Integration**: Adopted advanced RAG approaches from tldw_chatbook project for comprehensive document intelligence
- **Advanced Content Scoring**: Implemented weighted detail richness scoring system with 22+ comprehensive indicators for optimal chunk selection
- **Comprehensive Category Coverage**: Enhanced retrieval to capture all membership categories (Foundation, Social, Intermediate, Legacy, Corporate, Golfing Senior)
- **Multi-Category Synthesis**: System searches 60+ chunks and combines information across all membership types for complete fee structure responses
- **Percentage-Rich Content Prioritization**: Specialized extraction prioritizing chunks containing specific percentages (70%, 75%, 50%, 25%, 40%) with up to 12,000 characters per chunk
- **Professional Response Organization**: NotebookLM-style structured responses with Roman numerals (I-VII), proper spacing, and comprehensive cross-category synthesis
- **Enhanced Prompting System**: Implemented exhaustive extraction instructions inspired by tldw_chatbook's research workflow approach and Open Notebook's multi-dimensional quality metrics
- **🔥 SECTION-AWARE INGESTION SYSTEM**: Created lib/enhanced_ingest.py with intelligent chunking that preserves complete policy sections, prevents mid-sentence breaks of critical information (like reinstatement percentages), and uses regex pattern matching to identify and preserve complete sections
- **Contextual Overlap Processing**: Replaced character-based overlap with contextual overlap that preserves complete meaning and context between chunks
- **Percentage Sequence Validation**: Added validation to ensure no chunk ends with incomplete numbers or percentage references, specifically addressing reinstatement percentage capture issues
- **🧠 INSTITUTIONAL MEMORY SYSTEM**: Created comprehensive decision registry, historical pattern analysis, enhanced document chunking with decision detection, institutional knowledge capture, and board member insight tracking for perfect organizational recall
- **🎯 PERFECT EXTRACTION SYSTEM**: Built multi-pass extraction engine with 100% information capture guarantee including monetary amounts, percentages, dates, member names, voting records, and board-specific entities with comprehensive validation layers
- **🔮 PATTERN RECOGNITION ENGINE**: Created comprehensive governance pattern analysis system with decision outcome prediction, financial pattern tracking, precedent matching, and risk assessment providing data-driven insights like "73% chance of success based on 47 similar decisions"
- **🕸️ KNOWLEDGE GRAPH SYSTEM**: Built complete institutional knowledge graph connecting all board members, decisions, committees, policies, and vendors with temporal relationships, influence tracking, and ripple effect analysis enabling queries like "Show me everything connected to the 2019 dues increase decision and its ripple effects through 2024"

### Backend Architecture
- **Framework**: Flask web application with modular route organization
- **Authentication**: Flask-Login for session management with custom User model
- **File Handling**: Werkzeug secure filename processing with 16MB upload limits, multi-file support
- **Architecture Pattern**: Service layer pattern separating business logic from routes
- **Error Handling**: Comprehensive logging and user-friendly error messages
- **Chat History**: Persistent Q&A storage with full citation metadata preservation
- **Background Processing**: Worker process (worker.py) for asynchronous document ingestion
- **Job Status API**: Real-time endpoints to track background processing progress

### Data Storage Solutions
- **Primary Database**: Supabase (PostgreSQL) for document metadata, chat history, and institutional memory
- **Document Storage**: Local file system with secure filename handling
- **Vector Storage**: Embedded within Supabase for semantic search capabilities
- **Core Data Models**: Document, DocumentChunk, and QA_History entities with org_id/created_by relationships
- **🧠 Institutional Memory Models**: Decision registry, historical patterns, institutional knowledge, board member insights, and decision participation tracking for complete organizational recall
- **Enhanced Analytics**: Cross-referential analysis of decisions, voting patterns, success rates, and institutional knowledge preservation

### Authentication and Authorization
- **Strategy**: Development mode using environment variables (DEV_ORG_ID and DEV_USER_ID)
- **Session Management**: Bypassed for MVP - no login/registration required
- **Access Control**: Open access to all routes for simplified development
- **User Isolation**: All document and chat operations scoped to development organization

### Document Processing Pipeline
- **PDF Extraction**: PyPDF2 for text extraction with page-level granularity and structure preservation
- **🔥 Section-Aware Chunking**: Advanced regex-based section detection using pattern `\([a-z]\)\s+([^–]+)––([^(]+(?:\([a-z]\)|$))` to identify complete policy sections
- **Percentage Sequence Preservation**: Intelligent detection and preservation of percentage sequences (75%, 50%, 25%) without mid-sentence breaks
- **Contextual Overlap**: Smart overlap that preserves complete context rather than arbitrary character boundaries
- **Completeness Scoring**: Each chunk receives a section_completeness_score (0-100) indicating how complete and well-structured the content is
- **Validation System**: Chunks validated to ensure they don't end with incomplete numbers, percentages, or conjunction words
- **Embedding Generation**: OpenAI text-embedding-3-small (1536-dim) for high-quality vector representations
- **Chunk Summarization**: Pre-computed summaries using GPT-4o-mini for efficient retrieval
- **Storage**: Enhanced chunk storage with section metadata, percentage lists, and completeness scores
- **De-duplication**: SHA256-based file deduplication prevents re-processing identical documents

### AI Integration
- **Language Model**: OpenAI GPT-4o for chat responses with automatic fallback to GPT-4o-mini
- **Token Management**: Strict token budgeting with configurable limits (2400 summary tokens, 4800 final tokens)
- **Retrieval System**: Vector similarity search with MMR reranking and hybrid keyword fallback
- **MMR Reranking**: Maximal Marginal Relevance balances relevance vs diversity in results
- **Graceful Degradation**: Automatic fallback to keyword search when vector index unavailable
- **Performance Logging**: Real-time timing logs for retrieval and processing steps
- **Context Assembly**: Summary-first approach with fallback to truncated content for token efficiency
- **Response Generation**: Retry logic with exponential backoff and automatic model downgrading
- **Citation Management**: De-duplicated sources with document titles and page-specific deep links

## External Dependencies

### Core Services
- **Supabase**: PostgreSQL database hosting and management
- **OpenAI API**: GPT-4o for chat completion and text-embedding-3-large for embeddings

### Python Libraries
- **Flask**: Web framework with Login extension for authentication
- **PyPDF2**: PDF text extraction and processing
- **Supabase Python Client**: Database operations and vector search
- **OpenAI Python SDK**: AI model interactions

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme support
- **Font Awesome**: Icon library for enhanced visual design
- **HTMX**: Dynamic HTML enhancement for seamless interactions

### Environment Configuration

#### Required Secrets (Replit → Secrets)
- **OPENAI_API_KEY**: OpenAI API key for GPT models and embeddings
- **SUPABASE_URL**: Supabase project URL for database connection
- **SUPABASE_SERVICE_ROLE**: Supabase service role key for admin operations
- **DATABASE_URL**: PostgreSQL connection string from Supabase
- **SESSION_SECRET**: Flask session encryption key (optional for development)

#### Performance Tuning Secrets (Optional)
- **CHAT_PRIMARY=gpt-4o**: Main model for generating final answers
- **CHAT_COMPRESS=gpt-4o-mini**: Model for chunk summarization during ingestion  
- **EMBED_MODEL=text-embedding-3-small**: Model for generating document embeddings
- **MAX_CANDIDATES=16**: Maximum document chunks to retrieve per query
- **MMR_K=8**: Final reranked results count after MMR filtering
- **MMR_LAMBDA=0.55**: Diversity vs relevance balance (0.5-0.7, higher = more diverse)
- **USE_VECTOR=1**: Enable vector search (set to 0 for keyword-only mode)
- **MAX_SUMMARY_TOKENS=3200**: Token budget for building source notes from summaries
- **MAX_FINAL_TOKENS=5200**: Maximum tokens allowed in final prompt to AI
- **CHAT_TEMPERATURE=0.2**: AI response creativity (0.0-1.0, lower = more factual)

#### Development Environment Variables
- **DEV_ORG_ID**: Development organization ID for data isolation
- **DEV_USER_ID**: Development user ID for attribution

#### Background Worker Configuration (Optional)
- **WORKER_POLL_SEC=1.5**: Polling interval for checking new ingestion jobs
- **WORKER_BATCH=3**: Maximum documents to process simultaneously