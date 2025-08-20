# BoardContinuity AI v4.2 - Enterprise Platform

## Overview
BoardContinuity AI is a comprehensive enterprise-grade governance intelligence platform featuring sophisticated multi-agent coordination, advanced document intelligence, and professional-grade human oversight capabilities. The system provides 30-year veteran board member wisdom through a 5-level intelligence architecture coordinating specialized committee agents, enterprise security guardrails, intelligent human intervention, and real-time performance monitoring. Built for enterprise deployment with production-optimized configuration, automated verification, and comprehensive API ecosystem supporting both AI-first efficiency and appropriate human oversight for sensitive governance scenarios.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture
The system utilizes a 5-layer architecture: Ingestion, Memory, Intelligence, Synthesis, and Validation, designed for 100% capture and perfect storage. It processes multi-format documents (PDFs, images, audio, video) with perfect extraction. A Perfect Recall Engine coordinates subsystems for veteran-level responses. System Integrity Monitoring ensures continuous validation across 8 subsystems. A Validation Framework performs fact-checking, consistency verification, completeness scoring, and accuracy measurement.

### Enterprise Guardrails System
A comprehensive security and quality control system (`lib/enterprise_guardrails.py`) provides multi-layered validation for governance AI. It includes input validation (governance relevance, prompt injection), query safety, LLM-based security classification, and output validation (confidentiality leak prevention, PII detection, response quality, veteran language validation). Quality control covers institutional detail scoring, historical context, professional formatting, and citation accuracy. Real-time security monitoring, logging, confidence scoring, and graceful fallback are integrated.

### Committee Agents System
An advanced multi-agent system (`lib/committee_agents.py`) provides specialized committee expertise for targeted governance insights. It features 7 committee specialists (Golf, Finance, Food & Beverage, House, Membership, Grounds, Strategic Planning). Intelligent query routing directs queries to relevant committees, which provide veteran-level institutional knowledge and implementation guidance. A synthesis engine combines multiple committee perspectives.

### Enterprise RAG Agent System
A comprehensive enterprise-grade RAG agent (`lib/enterprise_rag_agent.py`) orchestrates the full AI workflow, integrating enterprise guardrails, committee specialists, veteran board member intelligence, and human intervention. It features intelligent query routing, multi-agent consultation synthesis, enterprise-grade validation, and intelligent escalation. It provides five levels of governance intelligence: Document, Committee Expertise, Multi-Agent Synthesis, Veteran Wisdom, and Enterprise Validation.

### Human Intervention System
An intelligent escalation framework (`lib/human_intervention.py`) provides seamless handoff to human specialists for high-risk governance scenarios. Triggers include high-risk actions, financial thresholds, legal concerns, low confidence responses, guardrail failures, complex edge cases, and explicit user requests. It monitors 40+ high-risk keywords and provides context-rich escalation with specialist type determination and urgency classification.

### Enhanced Intelligence Systems
**Specific Detail Extraction System (`lib/detail_extractor.py`)**: Extracts governance-specific details like financial amounts, committee names, vote counts, and precedent patterns.
**Precedent Warning System (`lib/precedent_analyzer.py`)**: Analyzes historical success/failure patterns, provides timeline predictions, risk assessments, and veteran recommendations with confidence scoring.
**Enhanced Response Generation (`lib/perfect_rag.py`)**: Direct OpenAI integration generates authentic veteran board member responses with structured formats including Historical Context, Practical Wisdom, Outcome Predictions, and Implementation Guidance.

### Bulletproof Document Processing System (August 2025)
**Multi-Strategy Extraction System (`lib/bulletproof_processing.py`)**: Enterprise-grade document processing system guaranteeing maximum coverage through intelligent 4-strategy fallback chain: PyPDF2 (primary fast extraction), PDFMiner (advanced layout analysis with configurable parameters), PyMuPDF (high-performance processing with metadata), and OCR (Tesseract for scanned documents). Features smart file path resolution with automatic Supabase storage download, content quality validation with minimum character thresholds, comprehensive error handling with detailed logging, and real-time processing status tracking. Includes intelligent chunking with tiktoken tokenization, embedding generation with OpenAI, and database storage with metadata preservation.

**Automated Processing Queue System (`lib/processing_queue.py`)**: Advanced queue-based document processing system with intelligent batch processing, parallel worker execution (3 workers by default), and real-time progress monitoring. Features automatic document discovery, priority-based ordering, async processing with bulletproof extraction integration, comprehensive error handling with retry logic, and real-time statistics tracking. Queue system enables non-blocking document processing with automatic failover, progress monitoring, and intelligent resource management. API endpoints provide queue management (`/api/queue-documents`), status monitoring (`/api/queue-status`), and automated processing (`/api/auto-process-documents`). Frontend integration includes real-time progress display, queue monitoring, and automatic coverage updates. System ensures maximum processing efficiency while maintaining enterprise-grade reliability and user experience.

### Frontend Architecture
A NotebookLM-inspired interface built with Bootstrap 5 and a custom design system emphasizes an AI-first conversational approach. The primary interface is a chat-driven AI assistant, utilizing HTMX for dynamic interactions like auto-resizing textareas and real-time message rendering. UI components include a chat-first layout with a sidebar for sources/upload, styled with modern gradient themes. The UX focuses on conversational document exploration, intelligent suggestions, confidence indicators, and seamless source citations. Comprehensive micro-animations provide visual feedback for document processing.

### Enhanced API Architecture
A comprehensive Flask API architecture (`app.py`) supports standard and enterprise-grade governance intelligence. Standard endpoints (`/api/query`) provide backward compatibility. Enhanced enterprise endpoints (`/api/enterprise-query`, `/api/evaluate-agent`, `/api/agent-status`) offer full multi-agent processing, performance monitoring, and health status, including detailed response metadata and human intervention escalation with context preservation.

### Backend Architecture
Built with Flask, the backend uses modular route organization and a service layer pattern. File handling uses Werkzeug, supporting multi-file uploads. Comprehensive logging and user-friendly error messages are implemented. Persistent Q&A storage preserves full citation metadata. A background worker (`worker.py`) handles asynchronous document ingestion with real-time job status tracking.

### Data Storage Solutions
Supabase (PostgreSQL) is the primary database for document metadata, chat history, and institutional memory models. Document storage uses the local file system. Vector storage is embedded within Supabase for semantic search. Core data models include Document, DocumentChunk, and QA_History, with org_id/created_by relationships. Institutional Memory Models capture decision registries and historical patterns.

### Authentication and Authorization
In the MVP, authentication uses environment variables (DEV_ORG_ID, DEV_USER_ID), bypassing traditional login/registration. All operations are scoped to a development organization, allowing open access for simplified development.

### Document Processing Pipeline
PDF extraction uses PyPDF2 with page-level granularity. A section-aware chunking system (`lib/enhanced_ingest.py`) employs regex patterns to preserve complete policy sections and prevent mid-sentence breaks, ensuring intelligent chunking with `section_completeness_score`. Chunks are validated to prevent ending with incomplete numbers, percentages, or conjunctions. OpenAI `text-embedding-3-small` generates 1536-dimensional embeddings. GPT-4o-mini is used for pre-computed chunk summaries. Enhanced chunk storage includes section metadata, percentage lists, and completeness scores. SHA256-based file deduplication prevents reprocessing identical documents.

### AI Integration
OpenAI GPT-4o-mini is the primary language model for chat responses, with automatic fallback and retry logic. Strict token budgeting is implemented. Retrieval uses vector similarity search with MMR reranking and a hybrid keyword fallback. Graceful degradation ensures fallback to keyword search if the vector index is unavailable. Real-time performance logging tracks retrieval and processing steps. Context assembly uses a summary-first approach. Response generation includes retry logic with exponential backoff and automatic model downgrading. Citation management de-duplicates sources with document titles and page-specific deep links.

## External Dependencies

### Core Services
- **Supabase**: PostgreSQL database hosting and management.
- **OpenAI API**: Provides GPT models (GPT-4o, GPT-4o-mini) for chat completion and `text-embedding-3-small` for embeddings.

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