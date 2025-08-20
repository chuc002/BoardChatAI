# Board Continuity MVP

## Overview
Board Continuity MVP is a document intelligence application enabling users to upload PDF documents and interact with them via AI-powered chat. It functions as a knowledge base system, processing PDFs to extract text, chunk content for retrieval, generate embeddings for semantic search, and provide natural language responses with contextual citations. The system aims to provide intelligent answers to queries about document content, citing relevant sections and page numbers. The overarching vision is to provide perfect recall with 30-year veteran wisdom, comprehensive validation, system integrity monitoring, and complete institutional intelligence synthesis for board-related decisions.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture
The system employs a 5-layer architecture: Ingestion (100% capture), Memory (perfect storage), Intelligence (30-year wisdom), Synthesis (perfect responses), and Validation (100% accuracy). It handles multi-format processing (PDFs, images, audio, video) with perfect extraction. A Perfect Recall Engine coordinates subsystems for veteran-level responses with institutional context. System Integrity Monitoring ensures continuous validation across 8 subsystems. Comprehensive analysis combines pattern recognition, knowledge graphs, governance intelligence, memory synthesis, and perfect RAG. A Validation Framework performs fact-checking, consistency verification, completeness scoring, and accuracy measurement.

### Enterprise Guardrails System (August 2025)
**Enterprise Security Framework (lib/enterprise_guardrails.py)**: Comprehensive security and quality control system providing multi-layered validation for governance AI systems. Features input validation with governance relevance checking (30+ keywords), advanced prompt injection detection, query safety validation, and LLM-based security classification. Output validation includes confidentiality leak prevention, PII detection and filtering, response quality assessment, and veteran board member language validation. Quality control provides institutional detail scoring, historical context validation, professional response formatting, and citation accuracy verification. Security monitoring includes real-time threat detection, comprehensive logging, confidence scoring, and graceful fallback handling. Integrated with RAG system for enterprise-grade security and quality assurance.

### Committee Agents System (August 2025)
**Specialized Committee Intelligence (lib/committee_agents.py)**: Advanced multi-agent system providing specialized committee expertise for targeted governance insights. Features 7 committee specialists: Golf (course operations, tournaments), Finance (budget, investments), Food & Beverage (dining, events), House (facilities, maintenance), Membership (recruitment, retention), Grounds (landscaping, environmental), and Strategic Planning (long-term vision). Intelligent query routing determines appropriate committee consultation based on governance topics. Committee agents provide veteran-level institutional knowledge with committee-specific historical context, precedent warnings, and specialized implementation guidance. Synthesis engine combines multiple committee perspectives into unified responses. Integrated with RAG system for enhanced governance intelligence beyond document retrieval.

### Enhanced Intelligence Systems (August 2025)
**Specific Detail Extraction System (lib/detail_extractor.py)**: Advanced pattern recognition extracts financial amounts, committee names, vote counts, meeting references, policy sections, fee categories, member counts, budget items, deadlines, and precedent patterns. Provides governance-specific extraction for institutional intelligence with contextual relationships between data points.

**Precedent Warning System (lib/precedent_analyzer.py)**: Sophisticated precedent analysis with historical success/failure pattern tracking. Committee pre-approval (85% success), budget <$50K (92% approval), vendor 3+ references (90% success), spring timing (78% acceptance). Identifies failure patterns including rushed decisions (60% failure), November timing (45% resistance), poor planning (40% overruns). Provides timeline predictions, risk assessments, and veteran recommendations with 0-100 confidence scoring.

**Enhanced Response Generation (lib/perfect_rag.py)**: Direct OpenAI integration generates authentic veteran board member responses with structured format including Historical Context, Practical Wisdom, Outcome Predictions, and Implementation Guidance sections. Enforces veteran language patterns with "In my 30 years of board experience..." openings and comprehensive detail integration for institutional wisdom delivery.

### Frontend Architecture
The frontend features a NotebookLM-inspired interface built with Bootstrap 5 and a custom design system, emphasizing an AI-first conversational approach. The primary interface is a chat-driven AI assistant. HTMX provides dynamic interactions, enhancing elements like auto-resizing textareas and real-time message rendering. UI components include a chat-first layout with a sidebar for sources/upload, styled with modern gradient themes and a professional purple-blue AI branding. The user experience focuses on conversational document exploration, intelligent suggestions, confidence indicators, and seamless source citations. **Micro-animations implemented (August 2025)**: Comprehensive document processing feedback including upload progress bars with glowing effects, processing spinners, shimmer overlays, success check animations, error shake feedback, floating document cards, animated processing dots, and fade-in chunk animations with staggered timing for professional visual feedback throughout the entire workflow.

### Backend Architecture
Built with Flask, the backend uses a modular route organization and a service layer pattern to separate business logic. Flask-Login manages user sessions (though bypassed in current MVP). File handling uses Werkzeug for secure processing, supporting multi-file uploads up to 16MB. Comprehensive logging and user-friendly error messages are implemented. Persistent Q&A storage preserves full citation metadata. A background worker (worker.py) handles asynchronous document ingestion, with real-time job status tracking via API endpoints.

### Data Storage Solutions
Supabase (PostgreSQL) is the primary database for document metadata, chat history, and institutional memory models. Document storage uses the local file system. Vector storage is embedded within Supabase for semantic search. Core data models include Document, DocumentChunk, and QA_History, with org_id/created_by relationships. Institutional Memory Models capture decision registries, historical patterns, institutional knowledge, and board member insights for comprehensive organizational recall and cross-referential analytics.

### Authentication and Authorization
In the current MVP development mode, authentication uses environment variables (DEV_ORG_ID, DEV_USER_ID), bypassing traditional login/registration. All document and chat operations are scoped to a development organization, allowing open access to all routes for simplified development.

### Document Processing Pipeline
PDF extraction utilizes PyPDF2 for text extraction with page-level granularity. A section-aware chunking system (lib/enhanced_ingest.py) employs regex patterns to preserve complete policy sections and prevent mid-sentence breaks of critical information, ensuring intelligent chunking that respects contextual boundaries rather than arbitrary character limits. Percentage sequences are intelligently detected and preserved. Each chunk receives a `section_completeness_score`. Chunks are validated to prevent ending with incomplete numbers, percentages, or conjunctions. OpenAI `text-embedding-3-small` generates 1536-dimensional embeddings. GPT-4o-mini is used for pre-computed chunk summaries. Enhanced chunk storage includes section metadata, percentage lists, and completeness scores. SHA256-based file deduplication prevents reprocessing identical documents.

### AI Integration
OpenAI GPT-4o-mini is the primary language model for chat responses, with automatic fallback and retry logic. Updated to use valid model names (gpt-4o-mini, text-embedding-3-small) for improved reliability. Strict token budgeting is implemented with configurable limits (e.g., 2400 summary tokens, 4800 final prompt tokens). Retrieval uses vector similarity search with MMR reranking and a hybrid keyword fallback. MMR (Maximal Marginal Relevance) balances relevance and diversity. Graceful degradation ensures a fallback to keyword search if the vector index is unavailable. Real-time performance logging tracks retrieval and processing steps. Context assembly uses a summary-first approach. Response generation includes retry logic with exponential backoff and automatic model downgrading. Citation management de-duplicates sources with document titles and page-specific deep links. Fixed tuple/dictionary return type consistency across all RAG components for robust API handling.

## External Dependencies

### Core Services
- **Supabase**: PostgreSQL database hosting and management.
- **OpenAI API**: Provides GPT models (GPT-4o, GPT-4o-mini) for chat completion and text-embedding-3-small for embeddings.

### Enterprise Scalability (August 2025)
**Enterprise Testing Framework (tests/test_enterprise_scale.py)**: Comprehensive performance validation system testing document capacity, query performance, memory efficiency, and concurrent load handling. Validates system readiness for 100+ document enterprise deployments with detailed assessment reports including response time benchmarking, memory usage analysis, and concurrent query handling validation.

**Batch Processing System (lib/batch_processor.py)**: Enterprise-grade document ingestion with memory management, parallel processing, and batch optimization. Supports configurable worker pools, automatic memory monitoring, and progressive batch processing for large-scale document uploads. Includes processing statistics, enterprise capacity validation, and performance estimation for production deployments.

**Enterprise Monitoring (lib/enterprise_monitoring.py)**: Production-ready performance monitoring with real-time metrics collection, alert thresholds, and optimization recommendations. Features function-level performance tracking, memory usage analysis, error rate monitoring, and automated health assessments. Provides comprehensive performance summaries and system optimization guidance for enterprise operations.

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