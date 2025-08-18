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

### Recent Performance Improvements (August 18, 2025)
- **✅ ULTIMATE NOTEBOOKLM ENHANCEMENT**: Final optimization to achieve 90%+ comprehensive quality matching NotebookLM standards
- **Comprehensive Category Coverage**: Enhanced retrieval to capture all membership categories (Foundation, Social, Intermediate, Legacy, Corporate, Golfing Senior)
- **Multi-Category Synthesis**: System now searches up to 60 chunks and combines information across all membership types for complete fee structure responses
- **Advanced Content Extraction**: Specialized extraction for each membership category with complete fee details, age requirements, and restrictions
- **Professional Response Organization**: Structured responses with clear section headers, bullet points, and comprehensive cross-category information synthesis

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
- **Primary Database**: Supabase (PostgreSQL) for document metadata, chat history
- **Document Storage**: Local file system with secure filename handling
- **Vector Storage**: Embedded within Supabase for semantic search capabilities
- **Data Models**: Document, DocumentChunk, and QA_History entities with org_id/created_by relationships

### Authentication and Authorization
- **Strategy**: Development mode using environment variables (DEV_ORG_ID and DEV_USER_ID)
- **Session Management**: Bypassed for MVP - no login/registration required
- **Access Control**: Open access to all routes for simplified development
- **User Isolation**: All document and chat operations scoped to development organization

### Document Processing Pipeline
- **PDF Extraction**: PyPDF2 for text extraction with page-level granularity
- **Text Chunking**: Intelligent chunking with overlap preservation for context continuity
- **Embedding Generation**: OpenAI text-embedding-3-small (1536-dim) for high-quality vector representations
- **Chunk Summarization**: Pre-computed summaries using GPT-4o-mini for efficient retrieval
- **Storage**: Chunked content stored with summaries and metadata linking to source documents
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