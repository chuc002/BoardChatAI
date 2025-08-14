# Board Continuity MVP

## Overview

Board Continuity MVP is a document intelligence application that allows users to upload PDF documents and interact with them through AI-powered chat functionality. The system processes PDF documents by extracting text, chunking content for optimal retrieval, generating embeddings for semantic search, and enabling natural language queries with contextual responses and source citations.

The application serves as a knowledge base system where users can upload board documents, meeting minutes, or other PDF materials and ask questions about their content. The AI provides intelligent responses with specific citations to relevant document sections and page numbers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Traditional server-rendered Flask templates with modern card-based design
- **JavaScript Enhancement**: HTMX for dynamic interactions, drag-and-drop file upload functionality
- **UI Components**: Card-based layout with drag-drop upload, document management, chat interface, and history panel
- **Styling**: Clean modern design with custom CSS for cards, citations, and interactive elements

### Backend Architecture
- **Framework**: Flask web application with modular route organization
- **Authentication**: Flask-Login for session management with custom User model
- **File Handling**: Werkzeug secure filename processing with 16MB upload limits, multi-file support
- **Architecture Pattern**: Service layer pattern separating business logic from routes
- **Error Handling**: Comprehensive logging and user-friendly error messages
- **Chat History**: Persistent Q&A storage with full citation metadata preservation

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
- **Retrieval System**: Vector similarity search with hybrid keyword fallback using pre-computed summaries
- **Context Assembly**: Summary-first approach with fallback to truncated content for token efficiency
- **Response Generation**: Retry logic with exponential backoff and automatic model downgrading
- **Citation Management**: De-duplicated sources with document titles and download links

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
- Required environment variables: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
- Optional: SESSION_SECRET for production security