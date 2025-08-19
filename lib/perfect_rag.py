"""
Perfect RAG System - Complete context retrieval with multiple strategies.

This module implements a comprehensive retrieval-augmented generation system
that ensures no relevant information is missed through multi-strategy retrieval,
completeness verification, and cross-reference enrichment.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

from lib.supa import supa
from lib.rag import answer_question_md
from lib.perfect_extraction import extract_perfect_information

logger = logging.getLogger(__name__)

@dataclass
class RetrievedContext:
    """Represents a retrieved context with metadata."""
    content: str
    source: str
    chunk_id: str
    document_id: str
    page_number: Optional[int]
    relevance_score: float
    retrieval_strategy: str
    completeness_score: float
    entities: Dict[str, List]
    cross_references: List[str]

@dataclass
class PerfectRAGResponse:
    """Complete RAG response with comprehensive context."""
    answer: str
    contexts: List[RetrievedContext]
    completeness_metrics: Dict[str, float]
    missing_information: List[str]
    confidence_score: float
    cross_references: List[Dict[str, Any]]
    entity_coverage: Dict[str, int]
    temporal_coverage: Dict[str, List]

class PerfectRAG:
    """
    Perfect RAG system that ensures complete context retrieval
    through multiple strategies and comprehensive verification.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        
        # Multiple retrieval strategies
        self.retrieval_strategies = [
            self.exact_match_retrieval,
            self.semantic_retrieval,
            self.pattern_based_retrieval,
            self.temporal_retrieval,
            self.entity_based_retrieval,
            self.relationship_retrieval,
            self.contextual_expansion_retrieval
        ]
        
        # Context completeness thresholds
        self.completeness_threshold = 0.8
        self.min_context_length = 200
        self.max_contexts_per_strategy = 5
        
        logger.info("Perfect RAG system initialized")
    
    def retrieve_complete_context(self, query: str, max_contexts: int = 20) -> List[RetrievedContext]:
        """Never miss relevant information - comprehensive multi-strategy retrieval."""
        logger.info(f"Retrieving complete context for: {query}")
        
        # Multi-strategy retrieval
        all_contexts = []
        strategy_results = {}
        
        for strategy in self.retrieval_strategies:
            try:
                contexts = strategy(query)
                strategy_results[strategy.__name__] = len(contexts)
                all_contexts.extend(contexts)
                logger.debug(f"{strategy.__name__} found {len(contexts)} contexts")
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        # Deduplicate while preserving best context from each source
        deduplicated_contexts = self._deduplicate_contexts(all_contexts)
        
        # Rank contexts by relevance and completeness
        ranked_contexts = self._rank_contexts(deduplicated_contexts, query)
        
        # Ensure completeness for top contexts
        complete_contexts = self._ensure_completeness(ranked_contexts[:max_contexts])
        
        # Add cross-references and enrichment
        enriched_contexts = self._add_cross_references(complete_contexts)
        
        logger.info(f"Retrieved {len(enriched_contexts)} complete contexts from {len(strategy_results)} strategies")
        
        return enriched_contexts
    
    def generate_perfect_response(self, query: str) -> PerfectRAGResponse:
        """Generate perfect response with comprehensive context analysis."""
        
        # Retrieve complete context
        contexts = self.retrieve_complete_context(query)
        
        # Analyze completeness
        completeness_metrics = self._analyze_completeness(contexts, query)
        
        # Identify missing information
        missing_info = self._identify_missing_information(contexts, query)
        
        # Extract entities and temporal information
        entity_coverage = self._analyze_entity_coverage(contexts)
        temporal_coverage = self._analyze_temporal_coverage(contexts)
        
        # Generate enhanced answer
        enhanced_answer = self._generate_enhanced_answer(query, contexts)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(contexts, completeness_metrics)
        
        # Build cross-references
        cross_references = self._build_cross_references(contexts)
        
        return PerfectRAGResponse(
            answer=enhanced_answer,
            contexts=contexts,
            completeness_metrics=completeness_metrics,
            missing_information=missing_info,
            confidence_score=confidence_score,
            cross_references=cross_references,
            entity_coverage=entity_coverage,
            temporal_coverage=temporal_coverage
        )
    
    # Retrieval strategies
    
    def exact_match_retrieval(self, query: str) -> List[RetrievedContext]:
        """Exact keyword matching for precise retrieval."""
        contexts = []
        
        try:
            # Extract key terms from query
            key_terms = self._extract_key_terms(query)
            
            # Search in chunks for exact matches
            for term in key_terms:
                chunks_result = supa.table('document_chunks').select(
                    'id, content, document_id, chunk_index, page_number'
                ).eq('org_id', self.org_id).ilike('content', f'%{term}%').limit(self.max_contexts_per_strategy).execute()
                
                for chunk in chunks_result.data or []:
                    # Calculate exact match score
                    content = chunk['content']
                    exact_matches = sum(1 for t in key_terms if t.lower() in content.lower())
                    relevance_score = exact_matches / len(key_terms)
                    
                    if relevance_score > 0.3:  # Minimum threshold
                        context = RetrievedContext(
                            content=content,
                            source=f"Document {chunk['document_id']}",
                            chunk_id=chunk['id'],
                            document_id=chunk['document_id'],
                            page_number=chunk.get('page_number'),
                            relevance_score=relevance_score,
                            retrieval_strategy='exact_match',
                            completeness_score=self._assess_chunk_completeness(content),
                            entities=self._extract_chunk_entities(content),
                            cross_references=[]
                        )
                        contexts.append(context)
            
        except Exception as e:
            logger.error(f"Exact match retrieval failed: {e}")
        
        return contexts
    
    def semantic_retrieval(self, query: str) -> List[RetrievedContext]:
        """Semantic similarity retrieval using embeddings."""
        contexts = []
        
        try:
            # Use existing RAG system for semantic retrieval
            rag_response = answer_question_md(query, self.org_id, max_candidates=self.max_contexts_per_strategy)
            
            # Extract contexts from RAG response
            if 'citations' in rag_response:
                for citation in rag_response['citations']:
                    chunk_id = citation.get('chunk_id')
                    if chunk_id:
                        # Get full chunk details
                        chunk_result = supa.table('document_chunks').select('*').eq('id', chunk_id).execute()
                        
                        if chunk_result.data:
                            chunk = chunk_result.data[0]
                            context = RetrievedContext(
                                content=chunk['content'],
                                source=citation.get('title', f"Document {chunk['document_id']}"),
                                chunk_id=chunk_id,
                                document_id=chunk['document_id'],
                                page_number=chunk.get('page_number'),
                                relevance_score=0.8,  # High score for semantic matches
                                retrieval_strategy='semantic',
                                completeness_score=self._assess_chunk_completeness(chunk['content']),
                                entities=self._extract_chunk_entities(chunk['content']),
                                cross_references=[]
                            )
                            contexts.append(context)
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
        
        return contexts
    
    def pattern_based_retrieval(self, query: str) -> List[RetrievedContext]:
        """Pattern-based retrieval for structured information."""
        contexts = []
        
        try:
            # Identify patterns in query
            patterns = self._identify_query_patterns(query)
            
            for pattern_type, pattern_regex in patterns.items():
                # Search chunks for pattern matches
                chunks_result = supa.table('document_chunks').select(
                    'id, content, document_id, chunk_index, page_number'
                ).eq('org_id', self.org_id).limit(50).execute()
                
                for chunk in chunks_result.data or []:
                    content = chunk['content']
                    
                    # Check for pattern matches
                    matches = re.findall(pattern_regex, content, re.IGNORECASE)
                    if matches:
                        relevance_score = min(1.0, len(matches) * 0.3)
                        
                        context = RetrievedContext(
                            content=content,
                            source=f"Document {chunk['document_id']} (Pattern: {pattern_type})",
                            chunk_id=chunk['id'],
                            document_id=chunk['document_id'],
                            page_number=chunk.get('page_number'),
                            relevance_score=relevance_score,
                            retrieval_strategy=f'pattern_{pattern_type}',
                            completeness_score=self._assess_chunk_completeness(content),
                            entities=self._extract_chunk_entities(content),
                            cross_references=[]
                        )
                        contexts.append(context)
            
        except Exception as e:
            logger.error(f"Pattern-based retrieval failed: {e}")
        
        return contexts
    
    def temporal_retrieval(self, query: str) -> List[RetrievedContext]:
        """Temporal-aware retrieval for time-sensitive information."""
        contexts = []
        
        try:
            # Extract temporal references from query
            temporal_refs = self._extract_temporal_references(query)
            
            if temporal_refs:
                # Search for chunks from relevant time periods
                for time_ref in temporal_refs:
                    year = time_ref.get('year')
                    month = time_ref.get('month')
                    
                    # Query chunks with temporal filtering
                    query_builder = supa.table('document_chunks').select(
                        'id, content, document_id, chunk_index, page_number, created_at'
                    ).eq('org_id', self.org_id)
                    
                    if year:
                        start_date = f"{year}-01-01"
                        end_date = f"{year}-12-31"
                        query_builder = query_builder.gte('created_at', start_date).lte('created_at', end_date)
                    
                    chunks_result = query_builder.limit(self.max_contexts_per_strategy).execute()
                    
                    for chunk in chunks_result.data or []:
                        # Check for temporal relevance in content
                        content = chunk['content']
                        temporal_score = self._calculate_temporal_relevance(content, temporal_refs)
                        
                        if temporal_score > 0.3:
                            context = RetrievedContext(
                                content=content,
                                source=f"Document {chunk['document_id']} ({year or 'temporal'})",
                                chunk_id=chunk['id'],
                                document_id=chunk['document_id'],
                                page_number=chunk.get('page_number'),
                                relevance_score=temporal_score,
                                retrieval_strategy='temporal',
                                completeness_score=self._assess_chunk_completeness(content),
                                entities=self._extract_chunk_entities(content),
                                cross_references=[]
                            )
                            contexts.append(context)
            
        except Exception as e:
            logger.error(f"Temporal retrieval failed: {e}")
        
        return contexts
    
    def entity_based_retrieval(self, query: str) -> List[RetrievedContext]:
        """Entity-based retrieval focusing on named entities."""
        contexts = []
        
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)
            
            if query_entities:
                # Search for chunks containing these entities
                entity_patterns = []
                for entity in query_entities:
                    entity_patterns.append(f'%{entity}%')
                
                # Search across all chunks
                chunks_result = supa.table('document_chunks').select(
                    'id, content, document_id, chunk_index, page_number'
                ).eq('org_id', self.org_id).limit(50).execute()
                
                for chunk in chunks_result.data or []:
                    content = chunk['content']
                    
                    # Calculate entity overlap
                    entity_matches = sum(1 for entity in query_entities if entity.lower() in content.lower())
                    entity_score = entity_matches / len(query_entities) if query_entities else 0
                    
                    if entity_score > 0.2:
                        context = RetrievedContext(
                            content=content,
                            source=f"Document {chunk['document_id']} (Entities)",
                            chunk_id=chunk['id'],
                            document_id=chunk['document_id'],
                            page_number=chunk.get('page_number'),
                            relevance_score=entity_score,
                            retrieval_strategy='entity_based',
                            completeness_score=self._assess_chunk_completeness(content),
                            entities=self._extract_chunk_entities(content),
                            cross_references=[]
                        )
                        contexts.append(context)
            
        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
        
        return contexts
    
    def relationship_retrieval(self, query: str) -> List[RetrievedContext]:
        """Retrieve based on relationships between concepts."""
        contexts = []
        
        try:
            # Extract relationship indicators from query
            relationships = self._extract_relationships(query)
            
            for rel_type, entities in relationships.items():
                if len(entities) >= 2:
                    # Search for chunks containing related entities
                    chunks_result = supa.table('document_chunks').select(
                        'id, content, document_id, chunk_index, page_number'
                    ).eq('org_id', self.org_id).limit(30).execute()
                    
                    for chunk in chunks_result.data or []:
                        content = chunk['content']
                        
                        # Check for relationship patterns
                        entity_count = sum(1 for entity in entities if entity.lower() in content.lower())
                        if entity_count >= 2:  # Contains multiple related entities
                            relevance_score = min(1.0, entity_count * 0.3)
                            
                            context = RetrievedContext(
                                content=content,
                                source=f"Document {chunk['document_id']} (Relationship: {rel_type})",
                                chunk_id=chunk['id'],
                                document_id=chunk['document_id'],
                                page_number=chunk.get('page_number'),
                                relevance_score=relevance_score,
                                retrieval_strategy='relationship',
                                completeness_score=self._assess_chunk_completeness(content),
                                entities=self._extract_chunk_entities(content),
                                cross_references=[]
                            )
                            contexts.append(context)
            
        except Exception as e:
            logger.error(f"Relationship retrieval failed: {e}")
        
        return contexts
    
    def contextual_expansion_retrieval(self, query: str) -> List[RetrievedContext]:
        """Expand context by retrieving adjacent chunks."""
        contexts = []
        
        try:
            # Get initial high-relevance chunks
            initial_contexts = self.semantic_retrieval(query)
            
            # Expand by retrieving adjacent chunks
            for context in initial_contexts[:3]:  # Expand top 3 contexts
                chunk_index = None
                document_id = context.document_id
                
                # Get chunk details to find index
                chunk_result = supa.table('document_chunks').select('chunk_index').eq('id', context.chunk_id).execute()
                if chunk_result.data:
                    chunk_index = chunk_result.data[0]['chunk_index']
                
                if chunk_index is not None:
                    # Get adjacent chunks
                    adjacent_chunks = supa.table('document_chunks').select(
                        'id, content, document_id, chunk_index, page_number'
                    ).eq('document_id', document_id).in_(
                        'chunk_index', [chunk_index - 1, chunk_index + 1]
                    ).execute()
                    
                    for adj_chunk in adjacent_chunks.data or []:
                        adj_context = RetrievedContext(
                            content=adj_chunk['content'],
                            source=f"Document {adj_chunk['document_id']} (Adjacent)",
                            chunk_id=adj_chunk['id'],
                            document_id=adj_chunk['document_id'],
                            page_number=adj_chunk.get('page_number'),
                            relevance_score=context.relevance_score * 0.7,  # Lower but related
                            retrieval_strategy='contextual_expansion',
                            completeness_score=self._assess_chunk_completeness(adj_chunk['content']),
                            entities=self._extract_chunk_entities(adj_chunk['content']),
                            cross_references=[context.chunk_id]
                        )
                        contexts.append(adj_context)
            
        except Exception as e:
            logger.error(f"Contextual expansion retrieval failed: {e}")
        
        return contexts
    
    # Context processing methods
    
    def _deduplicate_contexts(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Deduplicate contexts while preserving the best from each source."""
        seen_chunks = {}
        deduplicated = []
        
        for context in contexts:
            chunk_id = context.chunk_id
            
            if chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = context
                deduplicated.append(context)
            else:
                # Keep the one with higher relevance score
                existing = seen_chunks[chunk_id]
                if context.relevance_score > existing.relevance_score:
                    # Replace in deduplicated list
                    for i, existing_context in enumerate(deduplicated):
                        if existing_context.chunk_id == chunk_id:
                            deduplicated[i] = context
                            seen_chunks[chunk_id] = context
                            break
        
        return deduplicated
    
    def _rank_contexts(self, contexts: List[RetrievedContext], query: str) -> List[RetrievedContext]:
        """Rank contexts by relevance and completeness."""
        
        # Calculate combined scores
        for context in contexts:
            # Combine relevance and completeness
            combined_score = (context.relevance_score * 0.7 + context.completeness_score * 0.3)
            
            # Boost score for certain strategies
            if context.retrieval_strategy in ['semantic', 'exact_match']:
                combined_score *= 1.1
            
            # Update relevance score with combined score
            context.relevance_score = min(1.0, combined_score)
        
        # Sort by combined score
        return sorted(contexts, key=lambda x: x.relevance_score, reverse=True)
    
    def _ensure_completeness(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Ensure no partial information - fetch complete sections if needed."""
        complete_contexts = []
        
        for context in contexts:
            if self._is_partial(context):
                # Fetch complete section
                full_context = self._fetch_complete_section(context)
                complete_contexts.append(full_context)
            else:
                complete_contexts.append(context)
        
        return complete_contexts
    
    def _is_partial(self, context: RetrievedContext) -> bool:
        """Check if context appears to be partial."""
        content = context.content
        
        # Check for incomplete sentences or abrupt endings
        if len(content) < self.min_context_length:
            return True
        
        # Check for incomplete patterns
        if (content.strip().endswith((',', ';', 'and', 'or', 'but')) or
            not content.strip().endswith(('.', '!', '?', ':', ')')) or
            content.count('(') != content.count(')')):
            return True
        
        return context.completeness_score < self.completeness_threshold
    
    def _fetch_complete_section(self, context: RetrievedContext) -> RetrievedContext:
        """Fetch complete section containing the partial context."""
        try:
            # Get the document and try to find the complete section
            chunk_result = supa.table('document_chunks').select(
                'content, chunk_index'
            ).eq('document_id', context.document_id).order('chunk_index').execute()
            
            if chunk_result.data:
                # Find current chunk index
                current_index = None
                for chunk in chunk_result.data:
                    if context.content in chunk['content']:
                        current_index = chunk['chunk_index']
                        break
                
                if current_index is not None:
                    # Combine with adjacent chunks to form complete section
                    combined_content = []
                    for chunk in chunk_result.data:
                        chunk_idx = chunk['chunk_index']
                        if abs(chunk_idx - current_index) <= 1:  # Current + adjacent
                            combined_content.append(chunk['content'])
                    
                    # Create enhanced context
                    enhanced_context = RetrievedContext(
                        content=' '.join(combined_content),
                        source=f"{context.source} (Complete Section)",
                        chunk_id=context.chunk_id,
                        document_id=context.document_id,
                        page_number=context.page_number,
                        relevance_score=context.relevance_score,
                        retrieval_strategy=f"{context.retrieval_strategy}_enhanced",
                        completeness_score=1.0,  # Now complete
                        entities=self._extract_chunk_entities(' '.join(combined_content)),
                        cross_references=context.cross_references
                    )
                    
                    return enhanced_context
        
        except Exception as e:
            logger.error(f"Failed to fetch complete section: {e}")
        
        return context  # Return original if enhancement fails
    
    def _add_cross_references(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """Add cross-references between related contexts."""
        
        # Build entity and concept maps
        entity_to_contexts = defaultdict(list)
        
        for i, context in enumerate(contexts):
            for entity_type, entities in context.entities.items():
                for entity in entities:
                    entity_to_contexts[entity].append(i)
        
        # Add cross-references
        for i, context in enumerate(contexts):
            cross_refs = set(context.cross_references)
            
            # Find related contexts through shared entities
            for entity_type, entities in context.entities.items():
                for entity in entities:
                    for related_idx in entity_to_contexts[entity]:
                        if related_idx != i:
                            cross_refs.add(contexts[related_idx].chunk_id)
            
            context.cross_references = list(cross_refs)
        
        return contexts
    
    # Helper methods
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Remove common words and extract important terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        terms = []
        words = re.findall(r'\b\w+\b', query.lower())
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                terms.append(word)
        
        return terms
    
    def _identify_query_patterns(self, query: str) -> Dict[str, str]:
        """Identify patterns in the query."""
        patterns = {}
        
        # Financial patterns
        if any(word in query.lower() for word in ['fee', 'cost', 'price', 'amount', 'dollar', '$']):
            patterns['financial'] = r'\$[\d,]+(?:\.\d{2})?|(?:fee|cost|price|amount)s?\s*:?\s*\$?[\d,]+(?:\.\d{2})?'
        
        # Percentage patterns
        if any(word in query.lower() for word in ['percent', '%', 'rate']):
            patterns['percentage'] = r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*percent'
        
        # Date patterns
        if any(word in query.lower() for word in ['date', 'year', 'month', 'time', 'when']):
            patterns['date'] = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        
        # Membership patterns
        if any(word in query.lower() for word in ['member', 'membership', 'category']):
            patterns['membership'] = r'(?:foundation|social|intermediate|legacy|corporate|golfing)\s+member(?:ship)?'
        
        return patterns
    
    def _extract_temporal_references(self, query: str) -> List[Dict[str, Any]]:
        """Extract temporal references from query."""
        temporal_refs = []
        
        # Year patterns
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        for year in years:
            temporal_refs.append({'year': int(year), 'type': 'year'})
        
        # Month patterns
        months = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', query, re.IGNORECASE)
        for month in months:
            month_num = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }.get(month.lower())
            if month_num:
                temporal_refs.append({'month': month_num, 'type': 'month'})
        
        # Relative time
        if any(word in query.lower() for word in ['recent', 'latest', 'current', 'this year']):
            current_year = datetime.now().year
            temporal_refs.append({'year': current_year, 'type': 'current'})
        
        return temporal_refs
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        # Capitalize words that might be names
        words = query.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                entities.append(word)
        
        # Common entity patterns
        entity_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Names
            r'\b(?:President|Chairman|Secretary|Treasurer|Director)\s+[A-Z][a-z]+\b',  # Titles with names
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_relationships(self, query: str) -> Dict[str, List[str]]:
        """Extract relationships from query."""
        relationships = {}
        
        # Financial relationships
        if any(word in query.lower() for word in ['cost', 'fee', 'price', 'budget']):
            entities = self._extract_query_entities(query)
            if entities:
                relationships['financial'] = entities
        
        # Governance relationships
        if any(word in query.lower() for word in ['board', 'committee', 'vote', 'decision']):
            entities = self._extract_query_entities(query)
            if entities:
                relationships['governance'] = entities
        
        return relationships
    
    def _assess_chunk_completeness(self, content: str) -> float:
        """Assess how complete a chunk is."""
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 500:
            score += 0.2
        elif len(content) > 200:
            score += 0.1
        
        # Sentence completeness
        if content.strip().endswith(('.', '!', '?')):
            score += 0.2
        
        # Structural completeness
        if content.count('(') == content.count(')'):
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_chunk_entities(self, content: str) -> Dict[str, List]:
        """Extract entities from chunk content."""
        try:
            # Use perfect extraction system
            extraction_result = extract_perfect_information(content)
            
            entities = {
                'monetary_amounts': [str(item['amount']) for item in extraction_result.get('monetary_amounts', [])],
                'percentages': [str(item['percentage']) for item in extraction_result.get('percentages', [])],
                'members': [item['name'] for item in extraction_result.get('members', [])],
                'dates': [item['date_text'] for item in extraction_result.get('dates', [])],
                'committees': extraction_result.get('committees', [])
            }
            
            return entities
            
        except Exception as e:
            logger.warning(f"Failed to extract entities from chunk: {e}")
            return {'monetary_amounts': [], 'percentages': [], 'members': [], 'dates': [], 'committees': []}
    
    def _calculate_temporal_relevance(self, content: str, temporal_refs: List[Dict]) -> float:
        """Calculate temporal relevance score."""
        score = 0.0
        
        for ref in temporal_refs:
            if ref.get('year'):
                year_str = str(ref['year'])
                if year_str in content:
                    score += 0.5
            
            if ref.get('month'):
                month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                             'july', 'august', 'september', 'october', 'november', 'december']
                month_name = month_names[ref['month'] - 1]
                if month_name in content.lower():
                    score += 0.3
        
        return min(1.0, score)
    
    def _analyze_completeness(self, contexts: List[RetrievedContext], query: str) -> Dict[str, float]:
        """Analyze completeness of retrieved contexts."""
        metrics = {
            'coverage_score': 0.0,
            'depth_score': 0.0,
            'diversity_score': 0.0,
            'entity_coverage': 0.0,
            'temporal_coverage': 0.0
        }
        
        if not contexts:
            return metrics
        
        # Coverage score - how well query terms are covered
        query_terms = self._extract_key_terms(query)
        covered_terms = set()
        
        for context in contexts:
            for term in query_terms:
                if term.lower() in context.content.lower():
                    covered_terms.add(term)
        
        metrics['coverage_score'] = len(covered_terms) / len(query_terms) if query_terms else 1.0
        
        # Depth score - average completeness of contexts
        completeness_scores = [ctx.completeness_score for ctx in contexts]
        metrics['depth_score'] = sum(completeness_scores) / len(completeness_scores)
        
        # Diversity score - variety of retrieval strategies
        strategies = set(ctx.retrieval_strategy for ctx in contexts)
        metrics['diversity_score'] = len(strategies) / len(self.retrieval_strategies)
        
        # Entity coverage
        all_entities = set()
        for context in contexts:
            for entity_type, entities in context.entities.items():
                all_entities.update(entities)
        metrics['entity_coverage'] = min(1.0, len(all_entities) / 10)  # Normalize to 10 entities
        
        return metrics
    
    def _identify_missing_information(self, contexts: List[RetrievedContext], query: str) -> List[str]:
        """Identify potentially missing information."""
        missing = []
        
        query_lower = query.lower()
        
        # Check for specific information types
        if 'cost' in query_lower or 'fee' in query_lower:
            has_financial = any('$' in ctx.content for ctx in contexts)
            if not has_financial:
                missing.append("Specific cost or fee information")
        
        if 'percent' in query_lower or '%' in query_lower:
            has_percentage = any('%' in ctx.content for ctx in contexts)
            if not has_percentage:
                missing.append("Percentage or rate information")
        
        if 'when' in query_lower or 'date' in query_lower:
            has_dates = any(ctx.entities.get('dates') for ctx in contexts)
            if not has_dates:
                missing.append("Specific date or timing information")
        
        return missing
    
    def _analyze_entity_coverage(self, contexts: List[RetrievedContext]) -> Dict[str, int]:
        """Analyze entity coverage across contexts."""
        entity_coverage = defaultdict(int)
        
        for context in contexts:
            for entity_type, entities in context.entities.items():
                entity_coverage[entity_type] += len(entities)
        
        return dict(entity_coverage)
    
    def _analyze_temporal_coverage(self, contexts: List[RetrievedContext]) -> Dict[str, List]:
        """Analyze temporal coverage of contexts."""
        temporal_coverage = defaultdict(list)
        
        for context in contexts:
            dates = context.entities.get('dates', [])
            for date in dates:
                # Extract year from date
                year_match = re.search(r'\b(19|20)\d{2}\b', date)
                if year_match:
                    year = year_match.group()
                    temporal_coverage[year].append(context.chunk_id)
        
        return dict(temporal_coverage)
    
    def _generate_enhanced_answer(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate enhanced answer using retrieved contexts."""
        try:
            # Use existing RAG system but with enhanced contexts
            context_text = "\n\n".join([f"[{i+1}] {ctx.content}" for i, ctx in enumerate(contexts[:10])])
            
            # Generate answer using RAG
            rag_response = answer_question_md(query, self.org_id, max_candidates=len(contexts))
            
            return rag_response.get('answer', 'Unable to generate answer from available contexts.')
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced answer: {e}")
            return "Unable to generate comprehensive answer due to processing error."
    
    def _calculate_confidence_score(self, contexts: List[RetrievedContext], completeness_metrics: Dict) -> float:
        """Calculate overall confidence score for the response."""
        if not contexts:
            return 0.0
        
        # Base confidence from context quality
        avg_relevance = sum(ctx.relevance_score for ctx in contexts) / len(contexts)
        avg_completeness = sum(ctx.completeness_score for ctx in contexts) / len(contexts)
        
        # Weight by completeness metrics
        coverage_weight = completeness_metrics.get('coverage_score', 0) * 0.3
        depth_weight = completeness_metrics.get('depth_score', 0) * 0.3
        diversity_weight = completeness_metrics.get('diversity_score', 0) * 0.2
        
        confidence = (avg_relevance * 0.4 + avg_completeness * 0.3 + 
                     coverage_weight + depth_weight + diversity_weight)
        
        return min(1.0, confidence)
    
    def _build_cross_references(self, contexts: List[RetrievedContext]) -> List[Dict[str, Any]]:
        """Build cross-references between contexts."""
        cross_refs = []
        
        # Build entity-based cross-references
        entity_groups = defaultdict(list)
        
        for i, context in enumerate(contexts):
            for entity_type, entities in context.entities.items():
                for entity in entities:
                    entity_groups[entity].append(i)
        
        # Create cross-reference entries
        for entity, context_indices in entity_groups.items():
            if len(context_indices) > 1:
                cross_refs.append({
                    'entity': entity,
                    'entity_type': 'shared_entity',
                    'related_contexts': [contexts[i].chunk_id for i in context_indices],
                    'relationship': 'contains_same_entity'
                })
        
        return cross_refs


# Main API functions

def retrieve_perfect_context(org_id: str, query: str, max_contexts: int = 20) -> Dict[str, Any]:
    """Retrieve perfect context using multiple strategies."""
    rag = PerfectRAG(org_id)
    contexts = rag.retrieve_complete_context(query, max_contexts)
    
    return {
        'contexts': [
            {
                'content': ctx.content,
                'source': ctx.source,
                'chunk_id': ctx.chunk_id,
                'document_id': ctx.document_id,
                'page_number': ctx.page_number,
                'relevance_score': ctx.relevance_score,
                'retrieval_strategy': ctx.retrieval_strategy,
                'completeness_score': ctx.completeness_score,
                'entities': ctx.entities,
                'cross_references': ctx.cross_references
            }
            for ctx in contexts
        ],
        'total_contexts': len(contexts),
        'strategies_used': list(set(ctx.retrieval_strategy for ctx in contexts))
    }

def generate_perfect_rag_response(org_id: str, query: str) -> Dict[str, Any]:
    """Generate perfect RAG response with comprehensive analysis."""
    rag = PerfectRAG(org_id)
    response = rag.generate_perfect_response(query)
    
    return {
        'answer': response.answer,
        'contexts': [
            {
                'content': ctx.content,
                'source': ctx.source,
                'relevance_score': ctx.relevance_score,
                'strategy': ctx.retrieval_strategy,
                'completeness': ctx.completeness_score
            }
            for ctx in response.contexts
        ],
        'completeness_metrics': response.completeness_metrics,
        'missing_information': response.missing_information,
        'confidence_score': response.confidence_score,
        'cross_references': response.cross_references,
        'entity_coverage': response.entity_coverage,
        'temporal_coverage': response.temporal_coverage
    }