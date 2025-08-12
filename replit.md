# Board Continuity MVP

## Overview

Board Continuity MVP is a document intelligence application that allows users to upload PDF documents and interact with them through AI-powered chat functionality. The system processes PDF documents by extracting text, chunking content for optimal retrieval, generating embeddings for semantic search, and enabling natural language queries with contextual responses and source citations.

The application serves as a knowledge base system where users can upload board documents, meeting minutes, or other PDF materials and ask questions about their content. The AI provides intelligent responses with specific citations to relevant document sections and page numbers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Traditional server-rendered Flask templates with Bootstrap 5 dark theme
- **JavaScript Enhancement**: HTMX for dynamic interactions, custom JavaScript for chat and upload functionality
- **UI Components**: Responsive design with dashboard, chat interface, authentication pages, and document management
- **Styling**: Bootstrap-based with custom CSS for chat bubbles, citations, and enhanced UX elements

### Backend Architecture
- **Framework**: Flask web application with modular route organization
- **Authentication**: Flask-Login for session management with custom User model
- **File Handling**: Werkzeug secure filename processing with 16MB upload limits
- **Architecture Pattern**: Service layer pattern separating business logic from routes
- **Error Handling**: Comprehensive logging and user-friendly error messages

### Data Storage Solutions
- **Primary Database**: Supabase (PostgreSQL) for user accounts, document metadata, chat history
- **Document Storage**: Local file system with secure filename handling
- **Vector Storage**: Embedded within Supabase for semantic search capabilities
- **Data Models**: User, Document, DocumentChunk, and ChatMessage entities with proper relationships

### Authentication and Authorization
- **Strategy**: Email/password authentication with hashed passwords
- **Session Management**: Flask-Login with secure session handling
- **Access Control**: Login-required decorators protecting authenticated routes
- **User Isolation**: All document and chat operations scoped to authenticated user

### Document Processing Pipeline
- **PDF Extraction**: PyPDF2 for text extraction with page-level granularity
- **Text Chunking**: Intelligent chunking with overlap preservation for context continuity
- **Embedding Generation**: OpenAI text-embedding-3-large for high-quality vector representations
- **Storage**: Chunked content stored with metadata linking to source documents and page numbers

### AI Integration
- **Language Model**: OpenAI GPT-4o for chat responses and document analysis
- **Retrieval System**: Vector similarity search to find relevant document chunks
- **Context Assembly**: Multi-chunk context building with document and page citations
- **Response Generation**: Contextual responses with source attribution and citation links

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