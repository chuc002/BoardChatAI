"""
Institutional Knowledge Graph - Complete relationship mapping and temporal analysis.

This module builds a comprehensive knowledge graph connecting all board entities,
decisions, outcomes, and contextual factors to provide deep institutional insights.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

from lib.supa import supa

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str  # person, decision, committee, vendor, policy, outcome
    name: str
    properties: Dict[str, Any]
    created_at: datetime
    importance_score: float = 0.5

@dataclass
class GraphEdge:
    """Represents a relationship between nodes."""
    source_id: str
    target_id: str
    relationship_type: str  # voted_for, led_to, influenced_by, amended, etc.
    properties: Dict[str, Any]
    strength: float  # 0.0 to 1.0
    created_at: datetime
    temporal_context: Optional[Dict[str, Any]] = None

@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    path_strength: float
    temporal_span: Tuple[datetime, datetime]
    insights: List[str]

class InstitutionalKnowledgeGraph:
    """Main knowledge graph system."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self.temporal_index: Dict[int, List[str]] = defaultdict(list)  # year -> node_ids
        
    def build_complete_graph(self) -> Dict[str, Any]:
        """Build the complete institutional knowledge graph."""
        logger.info("Building complete institutional knowledge graph")
        
        # Clear existing graph
        self._reset_graph()
        
        # Build all entity types
        self._build_member_nodes()
        self._build_decision_nodes()
        self._build_committee_nodes() 
        self._build_vendor_nodes()
        self._build_policy_nodes()
        
        # Create all relationships
        self._create_decision_relationships()
        self._create_temporal_relationships()
        self._create_influence_relationships()
        self._create_outcome_relationships()
        
        # Calculate importance scores
        self._calculate_node_importance()
        
        # Build temporal index
        self._build_temporal_index()
        
        logger.info(f"Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'node_types': self._get_node_type_counts(),
            'relationship_types': self._get_relationship_type_counts(),
            'temporal_span': self._get_temporal_span(),
            'build_timestamp': datetime.now().isoformat()
        }
    
    def find_decision_ripple_effects(self, decision_id: str, years_forward: int = 5) -> Dict[str, Any]:
        """Find all ripple effects from a specific decision."""
        if decision_id not in self.nodes:
            return {"error": f"Decision {decision_id} not found in graph"}
        
        decision_node = self.nodes[decision_id]
        decision_date = decision_node.created_at
        
        # Find all paths forward from this decision
        ripple_effects = {
            'direct_outcomes': [],
            'influenced_decisions': [],
            'member_reactions': [],
            'policy_changes': [],
            'financial_impacts': [],
            'timeline': [],
            'affected_entities': set()
        }
        
        # BFS to find connected nodes within time window
        end_date = decision_date + timedelta(days=years_forward * 365)
        visited = set()
        queue = deque([(decision_id, 0, [])])  # (node_id, depth, path)
        
        while queue:
            current_id, depth, path = queue.popleft()
            
            if current_id in visited or depth > 4:  # Limit depth to prevent explosion
                continue
            
            visited.add(current_id)
            current_node = self.nodes[current_id]
            
            # Skip nodes outside time window
            if current_node.created_at > end_date:
                continue
            
            # Find outgoing edges from current node
            for edge in self.edges:
                if (edge.source_id == current_id and 
                    edge.target_id not in visited and
                    self.nodes[edge.target_id].created_at <= end_date):
                    
                    target_node = self.nodes[edge.target_id]
                    new_path = path + [edge]
                    
                    # Categorize the effect
                    self._categorize_ripple_effect(edge, target_node, ripple_effects, new_path)
                    
                    # Continue BFS
                    queue.append((edge.target_id, depth + 1, new_path))
        
        # Build timeline
        timeline_events = []
        for effect_list in [ripple_effects['direct_outcomes'], ripple_effects['influenced_decisions'], 
                           ripple_effects['policy_changes']]:
            for effect in effect_list:
                if 'date' in effect:
                    timeline_events.append(effect)
        
        timeline_events.sort(key=lambda x: x['date'])
        ripple_effects['timeline'] = timeline_events
        ripple_effects['affected_entities'] = list(ripple_effects['affected_entities'])
        
        return ripple_effects
    
    def analyze_member_voting_patterns(self, member_name: str) -> Dict[str, Any]:
        """Analyze a board member's complete voting patterns and influences."""
        member_nodes = [n for n in self.nodes.values() 
                       if n.type == 'person' and member_name.lower() in n.name.lower()]
        
        if not member_nodes:
            return {"error": f"Member {member_name} not found"}
        
        member_node = member_nodes[0]
        
        # Find all decisions this member participated in
        member_decisions = []
        voting_alliances = defaultdict(int)
        topics_voted_on = defaultdict(list)
        
        for edge in self.edges:
            if edge.source_id == member_node.id and edge.relationship_type.startswith('voted_'):
                decision_node = self.nodes[edge.target_id]
                vote_type = edge.relationship_type.split('_')[1]  # for, against, abstain
                
                decision_data = {
                    'decision_id': decision_node.id,
                    'title': decision_node.name,
                    'date': decision_node.created_at.isoformat(),
                    'vote': vote_type,
                    'decision_type': decision_node.properties.get('decision_type', 'unknown'),
                    'amount': decision_node.properties.get('amount_involved', 0),
                    'outcome': decision_node.properties.get('outcome', 'unknown')
                }
                
                member_decisions.append(decision_data)
                topics_voted_on[decision_data['decision_type']].append(vote_type)
                
                # Find other members who voted the same way on this decision
                for other_edge in self.edges:
                    if (other_edge.target_id == decision_node.id and 
                        other_edge.relationship_type == edge.relationship_type and
                        other_edge.source_id != member_node.id):
                        
                        other_member = self.nodes[other_edge.source_id]
                        voting_alliances[other_member.name] += 1
        
        # Calculate voting statistics
        total_votes = len(member_decisions)
        votes_for = sum(1 for d in member_decisions if d['vote'] == 'for')
        votes_against = sum(1 for d in member_decisions if d['vote'] == 'against')
        votes_abstain = sum(1 for d in member_decisions if d['vote'] == 'abstain')
        
        # Find topic preferences
        topic_preferences = {}
        for topic, votes in topics_voted_on.items():
            for_count = votes.count('for')
            topic_preferences[topic] = {
                'total_votes': len(votes),
                'support_rate': for_count / len(votes) if votes else 0,
                'typical_stance': 'supportive' if for_count > len(votes)/2 else 'cautious'
            }
        
        # Find strongest voting alliances
        top_allies = sorted(voting_alliances.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'member_name': member_node.name,
            'total_decisions_participated': total_votes,
            'voting_breakdown': {
                'votes_for': votes_for,
                'votes_against': votes_against, 
                'votes_abstain': votes_abstain,
                'support_rate': votes_for / total_votes if total_votes > 0 else 0
            },
            'topic_preferences': topic_preferences,
            'voting_allies': [{'name': name, 'agreements': count} for name, count in top_allies],
            'recent_decisions': sorted(member_decisions, key=lambda x: x['date'], reverse=True)[:10],
            'member_profile': {
                'tenure_start': member_node.created_at.isoformat(),
                'roles': member_node.properties.get('roles', []),
                'committees': member_node.properties.get('committees', []),
                'expertise_areas': member_node.properties.get('expertise_areas', [])
            }
        }
    
    def find_policy_evolution_chain(self, policy_topic: str) -> Dict[str, Any]:
        """Track how a policy evolved over time with all amendments and impacts."""
        # Find all policies related to the topic
        policy_nodes = [n for n in self.nodes.values() 
                       if n.type == 'policy' and policy_topic.lower() in n.name.lower()]
        
        if not policy_nodes:
            return {"error": f"No policies found for topic: {policy_topic}"}
        
        # Build evolution chain
        evolution_chain = []
        policy_impacts = []
        
        for policy_node in policy_nodes:
            # Find decisions that amended this policy
            amendments = []
            for edge in self.edges:
                if (edge.target_id == policy_node.id and 
                    edge.relationship_type in ['amended', 'modified', 'updated']):
                    
                    decision_node = self.nodes[edge.source_id]
                    amendments.append({
                        'date': decision_node.created_at.isoformat(),
                        'decision_title': decision_node.name,
                        'amendment_type': edge.properties.get('amendment_type', 'modification'),
                        'rationale': edge.properties.get('rationale', ''),
                        'vote_outcome': decision_node.properties.get('outcome', 'unknown')
                    })
            
            # Find impacts of this policy
            impacts = []
            for edge in self.edges:
                if (edge.source_id == policy_node.id and 
                    edge.relationship_type == 'led_to'):
                    
                    outcome_node = self.nodes[edge.target_id]
                    impacts.append({
                        'outcome_type': outcome_node.type,
                        'description': outcome_node.name,
                        'impact_strength': edge.strength,
                        'date': outcome_node.created_at.isoformat()
                    })
            
            evolution_chain.append({
                'policy_name': policy_node.name,
                'created_date': policy_node.created_at.isoformat(),
                'current_status': policy_node.properties.get('status', 'active'),
                'amendments': sorted(amendments, key=lambda x: x['date']),
                'impacts': impacts
            })
        
        # Sort by creation date
        evolution_chain.sort(key=lambda x: x['created_date'])
        
        return {
            'policy_topic': policy_topic,
            'evolution_chain': evolution_chain,
            'total_policies': len(policy_nodes),
            'total_amendments': sum(len(p['amendments']) for p in evolution_chain),
            'active_policies': sum(1 for p in evolution_chain if p['current_status'] == 'active')
        }
    
    def find_cyclical_patterns(self, pattern_type: str = 'all') -> Dict[str, Any]:
        """Identify cyclical patterns in decisions and outcomes."""
        patterns = {
            'seasonal': defaultdict(list),
            'economic': defaultdict(list),
            'membership': defaultdict(list),
            'committee': defaultdict(list)
        }
        
        # Analyze decisions by month/quarter for seasonal patterns
        for node in self.nodes.values():
            if node.type == 'decision':
                month = node.created_at.month
                quarter = (month - 1) // 3 + 1
                year = node.created_at.year
                
                decision_data = {
                    'id': node.id,
                    'name': node.name,
                    'type': node.properties.get('decision_type'),
                    'outcome': node.properties.get('outcome'),
                    'amount': node.properties.get('amount_involved', 0)
                }
                
                patterns['seasonal'][f'Q{quarter}'].append(decision_data)
                patterns['seasonal'][f'Month_{month:02d}'].append(decision_data)
        
        # Identify recurring decision types by season
        cyclical_insights = {
            'seasonal_patterns': {},
            'annual_cycles': {},
            'committee_cycles': {}
        }
        
        # Analyze seasonal patterns
        for period, decisions in patterns['seasonal'].items():
            if len(decisions) >= 3:  # Need sufficient data
                decision_types = [d['type'] for d in decisions if d['type']]
                if decision_types:
                    most_common_type = max(set(decision_types), key=decision_types.count)
                    success_rate = sum(1 for d in decisions if d['outcome'] in ['approved', 'passed']) / len(decisions)
                    
                    cyclical_insights['seasonal_patterns'][period] = {
                        'most_common_decision_type': most_common_type,
                        'total_decisions': len(decisions),
                        'success_rate': success_rate,
                        'avg_amount': statistics.mean([d['amount'] for d in decisions if d['amount']]) if any(d['amount'] for d in decisions) else 0
                    }
        
        return cyclical_insights
    
    def generate_contextual_insights(self, entity_id: str, context_years: int = 3) -> List[str]:
        """Generate contextual insights about an entity."""
        if entity_id not in self.nodes:
            return [f"Entity {entity_id} not found in knowledge graph"]
        
        entity = self.nodes[entity_id]
        insights = []
        
        # Find related entities
        related_entities = self._find_related_entities(entity_id, max_distance=2)
        
        # Generate insights based on entity type
        if entity.type == 'decision':
            insights.extend(self._generate_decision_insights(entity, related_entities))
        elif entity.type == 'person':
            insights.extend(self._generate_member_insights(entity, related_entities))
        elif entity.type == 'committee':
            insights.extend(self._generate_committee_insights(entity, related_entities))
        
        # Add temporal context
        insights.extend(self._generate_temporal_insights(entity, context_years))
        
        return insights[:10]  # Return top 10 insights
    
    def query_graph(self, query: str) -> Dict[str, Any]:
        """Query the knowledge graph using natural language."""
        query_lower = query.lower()
        results = {
            'matching_nodes': [],
            'relevant_relationships': [],
            'insights': []
        }
        
        # Simple keyword matching for now
        keywords = query_lower.split()
        
        for node in self.nodes.values():
            node_text = (node.name + ' ' + str(node.properties)).lower()
            matches = sum(1 for keyword in keywords if keyword in node_text)
            
            if matches > 0:
                results['matching_nodes'].append({
                    'id': node.id,
                    'type': node.type,
                    'name': node.name,
                    'relevance_score': matches / len(keywords),
                    'properties': node.properties
                })
        
        # Sort by relevance
        results['matching_nodes'].sort(key=lambda x: x['relevance_score'], reverse=True)
        results['matching_nodes'] = results['matching_nodes'][:20]  # Top 20 matches
        
        # Find relationships between matching nodes
        matching_node_ids = set(n['id'] for n in results['matching_nodes'])
        for edge in self.edges:
            if edge.source_id in matching_node_ids and edge.target_id in matching_node_ids:
                results['relevant_relationships'].append({
                    'source': self.nodes[edge.source_id].name,
                    'target': self.nodes[edge.target_id].name,
                    'relationship': edge.relationship_type,
                    'strength': edge.strength
                })
        
        return results
    
    # Helper methods
    
    def _reset_graph(self):
        """Reset the graph to empty state."""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency_list.clear()
        self.reverse_adjacency.clear()
        self.temporal_index.clear()
    
    def _build_member_nodes(self):
        """Build nodes for all board members."""
        try:
            members = supa.table('board_member_insights').select('*').eq('org_id', self.org_id).execute()
            
            for member in members.data or []:
                node_id = f"member_{member['id']}"
                self.nodes[node_id] = GraphNode(
                    id=node_id,
                    type='person',
                    name=member['member_name'],
                    properties={
                        'roles': member.get('committee_assignments', []),
                        'expertise_areas': member.get('expertise_areas', []),
                        'tenure_months': member.get('total_tenure_months', 0),
                        'voting_pattern': member.get('voting_pattern_summary', {}),
                        'effectiveness_score': member.get('effectiveness_score', 0.5)
                    },
                    created_at=self._parse_date(member.get('start_date', datetime.now().isoformat())),
                    importance_score=member.get('effectiveness_score', 0.5)
                )
                
        except Exception as e:
            logger.warning(f"Failed to build member nodes: {e}")
    
    def _build_decision_nodes(self):
        """Build nodes for all decisions."""
        try:
            decisions = supa.table('decision_registry').select('*').eq('org_id', self.org_id).execute()
            
            for decision in decisions.data or []:
                node_id = f"decision_{decision['id']}"
                self.nodes[node_id] = GraphNode(
                    id=node_id,
                    type='decision',
                    name=decision['title'],
                    properties={
                        'decision_type': decision.get('decision_type'),
                        'outcome': decision.get('outcome'),
                        'amount_involved': decision.get('amount_involved'),
                        'vote_counts': {
                            'for': decision.get('vote_count_for', 0),
                            'against': decision.get('vote_count_against', 0),
                            'abstain': decision.get('vote_count_abstain', 0)
                        },
                        'tags': decision.get('tags', [])
                    },
                    created_at=self._parse_date(decision.get('date', decision.get('created_at'))),
                    importance_score=self._calculate_decision_importance(decision)
                )
                
        except Exception as e:
            logger.warning(f"Failed to build decision nodes: {e}")
    
    def _build_committee_nodes(self):
        """Build nodes for committees based on document analysis."""
        # Extract committees from decisions and documents
        committees_found = set()
        
        for node in self.nodes.values():
            if node.type in ['decision', 'person']:
                props = node.properties
                if 'committees' in props:
                    committees_found.update(props['committees'])
                if 'committee_assignments' in props:
                    committees_found.update(props['committee_assignments'])
        
        for committee_name in committees_found:
            if committee_name:
                node_id = f"committee_{committee_name.lower().replace(' ', '_')}"
                self.nodes[node_id] = GraphNode(
                    id=node_id,
                    type='committee',
                    name=committee_name,
                    properties={
                        'decisions_influenced': [],
                        'members': [],
                        'performance_metrics': {}
                    },
                    created_at=datetime.now() - timedelta(days=365),  # Default to 1 year ago
                    importance_score=0.6
                )
    
    def _build_vendor_nodes(self):
        """Build nodes for vendors mentioned in decisions."""
        # Extract vendors from decision descriptions
        vendor_keywords = ['contractor', 'vendor', 'supplier', 'company', 'corp', 'inc', 'llc']
        
        for node in self.nodes.values():
            if node.type == 'decision':
                description = node.properties.get('description', '').lower()
                
                # Simple vendor detection - would be enhanced with NER
                words = description.split()
                for i, word in enumerate(words):
                    if any(keyword in word for keyword in vendor_keywords):
                        # Try to extract vendor name (next few words)
                        potential_vendor = ' '.join(words[max(0, i-2):i+3])
                        if len(potential_vendor) > 5:
                            vendor_id = f"vendor_{hash(potential_vendor) % 10000}"
                            if vendor_id not in self.nodes:
                                self.nodes[vendor_id] = GraphNode(
                                    id=vendor_id,
                                    type='vendor',
                                    name=potential_vendor.title(),
                                    properties={
                                        'contracts': [],
                                        'performance_history': []
                                    },
                                    created_at=node.created_at,
                                    importance_score=0.4
                                )
    
    def _build_policy_nodes(self):
        """Build nodes for policies from institutional knowledge."""
        try:
            knowledge = supa.table('institutional_knowledge').select('*').eq('org_id', self.org_id).eq('knowledge_type', 'procedural').execute()
            
            for item in knowledge.data or []:
                node_id = f"policy_{item['id']}"
                self.nodes[node_id] = GraphNode(
                    id=node_id,
                    type='policy',
                    name=item['title'],
                    properties={
                        'category': item.get('category'),
                        'context': item.get('context'),
                        'status': 'active' if item.get('is_current') else 'inactive',
                        'confidence_score': item.get('confidence_score', 0.5)
                    },
                    created_at=self._parse_date(item.get('time_period_start', item.get('created_at'))),
                    importance_score=item.get('confidence_score', 0.5)
                )
                
        except Exception as e:
            logger.warning(f"Failed to build policy nodes: {e}")
    
    def _create_decision_relationships(self):
        """Create relationships between decisions and members."""
        try:
            # Get decision participation data
            participation = supa.table('decision_participation').select('*').eq('org_id', self.org_id).execute()
            
            for p in participation.data or []:
                decision_id = f"decision_{p['decision_id']}"
                member_id = f"member_{p['member_insight_id']}"
                
                if decision_id in self.nodes and member_id in self.nodes:
                    vote = p.get('vote', 'unknown')
                    relationship_type = f"voted_{vote}"
                    
                    edge = GraphEdge(
                        source_id=member_id,
                        target_id=decision_id,
                        relationship_type=relationship_type,
                        properties={
                            'participation_level': p.get('participation_level'),
                            'influence_level': p.get('influence_level'),
                            'was_pivotal': p.get('was_pivotal_vote', False)
                        },
                        strength=self._calculate_vote_strength(p),
                        created_at=self.nodes[decision_id].created_at
                    )
                    
                    self.edges.append(edge)
                    self.adjacency_list[member_id].append(decision_id)
                    self.reverse_adjacency[decision_id].append(member_id)
                    
        except Exception as e:
            logger.warning(f"Failed to create decision relationships: {e}")
    
    def _create_temporal_relationships(self):
        """Create temporal relationships between decisions."""
        decisions_by_date = [(n.created_at, n) for n in self.nodes.values() if n.type == 'decision']
        decisions_by_date.sort(key=lambda x: x[0])
        
        for i in range(len(decisions_by_date) - 1):
            current_date, current_decision = decisions_by_date[i]
            next_date, next_decision = decisions_by_date[i + 1]
            
            # If decisions are related and close in time, create temporal link
            time_diff = (next_date - current_date).days
            if time_diff <= 180:  # Within 6 months
                
                # Check if decisions are related by type or content
                if self._are_decisions_related(current_decision, next_decision):
                    edge = GraphEdge(
                        source_id=current_decision.id,
                        target_id=next_decision.id,
                        relationship_type='preceded',
                        properties={
                            'time_gap_days': time_diff,
                            'relationship_basis': 'temporal_proximity'
                        },
                        strength=max(0.1, 1.0 - (time_diff / 180)),
                        created_at=next_date
                    )
                    
                    self.edges.append(edge)
                    self.adjacency_list[current_decision.id].append(next_decision.id)
                    self.reverse_adjacency[next_decision.id].append(current_decision.id)
    
    def _create_influence_relationships(self):
        """Create influence relationships between entities."""
        # Committee -> Decision influences
        for node in self.nodes.values():
            if node.type == 'committee':
                committee_name = node.name.lower()
                
                for decision_node in self.nodes.values():
                    if decision_node.type == 'decision':
                        decision_text = (decision_node.name + ' ' + 
                                       str(decision_node.properties)).lower()
                        
                        if committee_name in decision_text:
                            edge = GraphEdge(
                                source_id=node.id,
                                target_id=decision_node.id,
                                relationship_type='influenced',
                                properties={'influence_type': 'committee_recommendation'},
                                strength=0.7,
                                created_at=decision_node.created_at
                            )
                            
                            self.edges.append(edge)
                            self.adjacency_list[node.id].append(decision_node.id)
                            self.reverse_adjacency[decision_node.id].append(node.id)
    
    def _create_outcome_relationships(self):
        """Create relationships between decisions and their outcomes."""
        # This would analyze subsequent events/decisions that resulted from earlier ones
        # For now, create basic outcome relationships based on decision success/failure
        
        for decision_node in self.nodes.values():
            if decision_node.type == 'decision':
                outcome = decision_node.properties.get('outcome')
                
                if outcome in ['approved', 'passed']:
                    # Look for subsequent related activities
                    decision_date = decision_node.created_at
                    end_date = decision_date + timedelta(days=365)  # Look 1 year forward
                    
                    for other_node in self.nodes.values():
                        if (other_node.type == 'decision' and 
                            other_node.created_at > decision_date and
                            other_node.created_at <= end_date and
                            self._are_decisions_related(decision_node, other_node)):
                            
                            edge = GraphEdge(
                                source_id=decision_node.id,
                                target_id=other_node.id,
                                relationship_type='led_to',
                                properties={'outcome_type': 'subsequent_decision'},
                                strength=0.6,
                                created_at=other_node.created_at
                            )
                            
                            self.edges.append(edge)
                            self.adjacency_list[decision_node.id].append(other_node.id)
                            self.reverse_adjacency[other_node.id].append(decision_node.id)
    
    def _calculate_node_importance(self):
        """Calculate importance scores for all nodes using graph metrics."""
        # Use degree centrality and other graph metrics
        for node_id, node in self.nodes.items():
            in_degree = len(self.reverse_adjacency[node_id])
            out_degree = len(self.adjacency_list[node_id])
            
            # Base importance on connectivity
            centrality_score = (in_degree + out_degree) / max(1, len(self.nodes))
            
            # Adjust based on node type
            if node.type == 'decision':
                # Important decisions affect more entities
                importance = min(1.0, centrality_score * 2)
            elif node.type == 'person':
                # Important members participate in more decisions
                importance = min(1.0, centrality_score * 1.5)
            else:
                importance = centrality_score
            
            node.importance_score = max(node.importance_score, importance)
    
    def _build_temporal_index(self):
        """Build index of nodes by time period."""
        for node in self.nodes.values():
            year = node.created_at.year
            self.temporal_index[year].append(node.id)
    
    def _categorize_ripple_effect(self, edge: GraphEdge, target_node: GraphNode, 
                                 ripple_effects: Dict, path: List[GraphEdge]):
        """Categorize a ripple effect into the appropriate bucket."""
        effect_data = {
            'target_name': target_node.name,
            'target_type': target_node.type,
            'relationship': edge.relationship_type,
            'strength': edge.strength,
            'date': target_node.created_at.isoformat(),
            'path_length': len(path)
        }
        
        if edge.relationship_type == 'led_to':
            ripple_effects['direct_outcomes'].append(effect_data)
        elif target_node.type == 'decision':
            ripple_effects['influenced_decisions'].append(effect_data)
        elif target_node.type == 'policy':
            ripple_effects['policy_changes'].append(effect_data)
        elif edge.relationship_type.startswith('voted_'):
            ripple_effects['member_reactions'].append(effect_data)
        
        ripple_effects['affected_entities'].add(target_node.name)
    
    def _find_related_entities(self, entity_id: str, max_distance: int = 2) -> List[GraphNode]:
        """Find entities related to the given entity within max_distance."""
        related = []
        visited = set()
        queue = deque([(entity_id, 0)])
        
        while queue:
            current_id, distance = queue.popleft()
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            
            if distance > 0:  # Don't include the original entity
                related.append(self.nodes[current_id])
            
            # Add neighbors
            for neighbor_id in self.adjacency_list[current_id]:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, distance + 1))
            
            for neighbor_id in self.reverse_adjacency[current_id]:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, distance + 1))
        
        return related
    
    def _generate_decision_insights(self, decision: GraphNode, related: List[GraphNode]) -> List[str]:
        """Generate insights for a decision node."""
        insights = []
        
        # Find similar decisions
        similar_decisions = [n for n in related if n.type == 'decision']
        if similar_decisions:
            outcomes = [d.properties.get('outcome') for d in similar_decisions]
            success_rate = outcomes.count('approved') / len(outcomes) if outcomes else 0
            insights.append(f"Similar decisions have a {success_rate:.0%} success rate")
        
        # Find involved members
        members = [n for n in related if n.type == 'person']
        if members:
            member_names = [m.name for m in members[:3]]
            insights.append(f"Key participants: {', '.join(member_names)}")
        
        return insights
    
    def _generate_member_insights(self, member: GraphNode, related: List[GraphNode]) -> List[str]:
        """Generate insights for a member node."""
        insights = []
        
        decisions = [n for n in related if n.type == 'decision']
        if decisions:
            decision_types = [d.properties.get('decision_type') for d in decisions]
            most_common_type = max(set(decision_types), key=decision_types.count) if decision_types else None
            if most_common_type:
                insights.append(f"Most active in {most_common_type} decisions")
        
        return insights
    
    def _generate_committee_insights(self, committee: GraphNode, related: List[GraphNode]) -> List[str]:
        """Generate insights for a committee node."""
        insights = []
        
        influenced_decisions = [n for n in related if n.type == 'decision']
        if influenced_decisions:
            success_count = sum(1 for d in influenced_decisions 
                              if d.properties.get('outcome') in ['approved', 'passed'])
            success_rate = success_count / len(influenced_decisions)
            insights.append(f"Committee recommendations have {success_rate:.0%} success rate")
        
        return insights
    
    def _generate_temporal_insights(self, entity: GraphNode, context_years: int) -> List[str]:
        """Generate temporal context insights."""
        insights = []
        
        entity_year = entity.created_at.year
        
        # Find what else was happening around the same time
        concurrent_entities = []
        for year in range(entity_year - 1, entity_year + 2):
            concurrent_entities.extend(self.temporal_index.get(year, []))
        
        # Filter out self and get interesting concurrent events
        concurrent_decisions = [self.nodes[node_id] for node_id in concurrent_entities 
                              if node_id != entity.id and self.nodes[node_id].type == 'decision']
        
        if concurrent_decisions:
            insights.append(f"Occurred during period with {len(concurrent_decisions)} other major decisions")
        
        return insights
    
    # Utility methods
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        if not date_str:
            return datetime.now()
        
        try:
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(date_str, '%Y-%m-%d')
        except Exception:
            return datetime.now()
    
    def _calculate_decision_importance(self, decision: Dict) -> float:
        """Calculate importance score for a decision."""
        score = 0.5  # Base score
        
        # Amount involved increases importance
        amount = decision.get('amount_involved', 0)
        if amount > 100000:
            score += 0.3
        elif amount > 10000:
            score += 0.2
        
        # Controversial decisions are more important
        votes_for = decision.get('vote_count_for', 0)
        votes_against = decision.get('vote_count_against', 0)
        
        if votes_for + votes_against > 0:
            margin = abs(votes_for - votes_against) / (votes_for + votes_against)
            if margin < 0.3:  # Close vote
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_vote_strength(self, participation: Dict) -> float:
        """Calculate the strength of a voting relationship."""
        base_strength = 0.5
        
        if participation.get('was_pivotal_vote'):
            base_strength += 0.3
        
        influence = participation.get('influence_level', 'none')
        if influence in ['high', 'decisive']:
            base_strength += 0.2
        elif influence == 'medium':
            base_strength += 0.1
        
        return min(1.0, base_strength)
    
    def _are_decisions_related(self, decision1: GraphNode, decision2: GraphNode) -> bool:
        """Check if two decisions are related."""
        # Check if same decision type
        if decision1.properties.get('decision_type') == decision2.properties.get('decision_type'):
            return True
        
        # Check for common tags
        tags1 = set(decision1.properties.get('tags', []))
        tags2 = set(decision2.properties.get('tags', []))
        
        if tags1.intersection(tags2):
            return True
        
        # Check for similar amounts
        amount1 = decision1.properties.get('amount_involved', 0)
        amount2 = decision2.properties.get('amount_involved', 0)
        
        if amount1 > 0 and amount2 > 0:
            ratio = min(amount1, amount2) / max(amount1, amount2)
            if ratio > 0.5:  # Similar amounts
                return True
        
        return False
    
    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get count of nodes by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.type] += 1
        return dict(counts)
    
    def _get_relationship_type_counts(self) -> Dict[str, int]:
        """Get count of relationships by type."""
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.relationship_type] += 1
        return dict(counts)
    
    def _get_temporal_span(self) -> Dict[str, str]:
        """Get the temporal span of the graph."""
        if not self.nodes:
            return {}
        
        dates = [node.created_at for node in self.nodes.values()]
        return {
            'earliest': min(dates).isoformat(),
            'latest': max(dates).isoformat(),
            'span_years': (max(dates) - min(dates)).days / 365.25
        }

# Main API functions

def build_knowledge_graph(org_id: str) -> Dict[str, Any]:
    """Build the complete institutional knowledge graph."""
    graph = InstitutionalKnowledgeGraph(org_id)
    return graph.build_complete_graph()

def analyze_decision_ripple_effects(org_id: str, decision_id: str, years_forward: int = 5) -> Dict[str, Any]:
    """Analyze the ripple effects of a specific decision."""
    graph = InstitutionalKnowledgeGraph(org_id)
    graph.build_complete_graph()
    return graph.find_decision_ripple_effects(decision_id, years_forward)

def get_member_complete_analysis(org_id: str, member_name: str) -> Dict[str, Any]:
    """Get complete analysis of a board member's patterns and influences."""
    graph = InstitutionalKnowledgeGraph(org_id)
    graph.build_complete_graph()
    return graph.analyze_member_voting_patterns(member_name)

def trace_policy_evolution(org_id: str, policy_topic: str) -> Dict[str, Any]:
    """Trace how a policy evolved over time."""
    graph = InstitutionalKnowledgeGraph(org_id)
    graph.build_complete_graph()
    return graph.find_policy_evolution_chain(policy_topic)

def find_governance_cycles(org_id: str) -> Dict[str, Any]:
    """Find cyclical patterns in governance."""
    graph = InstitutionalKnowledgeGraph(org_id)
    graph.build_complete_graph()
    return graph.find_cyclical_patterns()

def query_knowledge_graph(org_id: str, query: str) -> Dict[str, Any]:
    """Query the knowledge graph with natural language."""
    graph = InstitutionalKnowledgeGraph(org_id)
    graph.build_complete_graph()
    return graph.query_graph(query)