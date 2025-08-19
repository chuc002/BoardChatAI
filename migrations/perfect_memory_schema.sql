-- Perfect Memory Database Schema for Complete Institutional Intelligence
-- This schema captures every word, decision, and interaction for perfect recall

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Every word spoken or written - complete institutional record
CREATE TABLE complete_record (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    type VARCHAR(50) NOT NULL, -- 'meeting', 'email', 'memo', 'discussion', 'presentation', 'informal'
    source_document_id UUID REFERENCES documents(id),
    participants TEXT[] NOT NULL DEFAULT '{}',
    content TEXT NOT NULL,
    content_summary TEXT,
    key_topics TEXT[],
    decisions_made UUID[] DEFAULT '{}',
    action_items JSONB DEFAULT '[]',
    follow_ups JSONB DEFAULT '[]',
    outcomes JSONB DEFAULT '{}',
    location TEXT,
    duration_minutes INTEGER,
    importance_score FLOAT DEFAULT 0.5,
    sentiment_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    INDEX (org_id, date),
    INDEX (org_id, type),
    INDEX (participants),
    INDEX (key_topics)
);

-- Every decision with complete context and lifecycle
CREATE TABLE decision_complete (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    decision_title TEXT NOT NULL,
    proposed_date DATE NOT NULL,
    proposed_by TEXT NOT NULL,
    description TEXT NOT NULL,
    rationale TEXT,
    background_context TEXT,
    financial_implications JSONB,
    stakeholders_affected TEXT[],
    discussion_points JSONB DEFAULT '[]', -- Array of {speaker, point, timestamp, sentiment}
    concerns_raised JSONB DEFAULT '[]', -- Array of {concern, raised_by, severity, addressed}
    modifications JSONB DEFAULT '[]', -- Array of {modification, reason, impact}
    vote_date DATE,
    vote_details JSONB, -- {for: 5, against: 2, abstain: 1, details: [{member, vote, reason}]}
    vote_margin FLOAT,
    unanimous BOOLEAN DEFAULT FALSE,
    implementation_plan JSONB,
    implementation_start_date DATE,
    implementation_completion_date DATE,
    actual_implementation JSONB,
    budget_projected DECIMAL(12,2),
    budget_actual DECIMAL(12,2),
    cost_variance FLOAT,
    outcomes_measured JSONB,
    success_metrics JSONB,
    member_feedback JSONB,
    lessons_learned TEXT,
    would_repeat BOOLEAN,
    retrospective_assessment TEXT,
    related_decisions UUID[] DEFAULT '{}',
    precedent_decisions UUID[] DEFAULT '{}',
    consequence_decisions UUID[] DEFAULT '{}',
    decision_type VARCHAR(50),
    urgency_level VARCHAR(20) DEFAULT 'normal',
    complexity_score FLOAT DEFAULT 0.5,
    risk_assessment JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, proposed_date),
    INDEX (org_id, decision_type),
    INDEX (proposed_by),
    INDEX (stakeholders_affected),
    INDEX (vote_date)
);

-- Every person's complete institutional history
CREATE TABLE member_complete_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    member_name TEXT NOT NULL,
    member_id TEXT UNIQUE,
    active_status BOOLEAN DEFAULT TRUE,
    join_date DATE,
    departure_date DATE,
    membership_category TEXT,
    
    -- Position and role history
    positions_held JSONB DEFAULT '[]', -- Array of {position, start_date, end_date, responsibilities}
    committees_served JSONB DEFAULT '[]', -- Array of {committee, role, start_date, end_date, contributions}
    leadership_roles JSONB DEFAULT '[]', -- Array of {role, term, achievements, challenges}
    
    -- Voting and participation patterns
    votes_cast JSONB DEFAULT '{}', -- Detailed voting history with reasoning
    voting_patterns JSONB DEFAULT '{}', -- Analysis of voting tendencies
    proposals_made JSONB DEFAULT '[]', -- All proposals submitted
    meeting_attendance JSONB DEFAULT '{}', -- Detailed attendance tracking
    participation_metrics JSONB DEFAULT '{}', -- Speaking time, motion frequency, etc.
    
    -- Expertise and knowledge
    expertise_areas TEXT[] DEFAULT '{}',
    professional_background TEXT,
    educational_background TEXT,
    relevant_experience TEXT,
    specialized_knowledge TEXT[],
    
    -- Positions and preferences
    known_positions JSONB DEFAULT '{}', -- Documented positions on various topics
    policy_preferences JSONB DEFAULT '{}', -- Consistent policy positions
    decision_influences JSONB DEFAULT '{}', -- Factors that influence their decisions
    
    -- Relationships and dynamics
    relationships JSONB DEFAULT '{}', -- Professional relationships within organization
    influence_network JSONB DEFAULT '{}', -- Who influences them and whom they influence
    collaboration_patterns JSONB DEFAULT '{}', -- Frequent collaborators and partnerships
    conflict_history JSONB DEFAULT '[]', -- Any recorded conflicts or disagreements
    
    -- Performance and effectiveness
    effectiveness_scores JSONB DEFAULT '{}', -- Various effectiveness metrics
    contribution_assessment JSONB DEFAULT '{}', -- Assessment of contributions over time
    leadership_effectiveness JSONB DEFAULT '{}', -- Leadership skill assessment
    communication_style TEXT,
    decision_making_style TEXT,
    
    -- Meta information
    data_completeness_score FLOAT DEFAULT 0.0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    
    UNIQUE(org_id, member_name),
    INDEX (org_id, member_name),
    INDEX (org_id, active_status),
    INDEX (membership_category),
    INDEX (expertise_areas),
    INDEX (join_date)
);

-- Conversation and discussion threads for perfect context
CREATE TABLE discussion_threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    thread_title TEXT NOT NULL,
    topic TEXT NOT NULL,
    started_date TIMESTAMPTZ NOT NULL,
    ended_date TIMESTAMPTZ,
    participants TEXT[],
    discussion_type VARCHAR(50), -- 'formal_meeting', 'committee', 'informal', 'email_chain'
    parent_thread_id UUID REFERENCES discussion_threads(id),
    related_decisions UUID[],
    key_points JSONB DEFAULT '[]',
    consensus_reached BOOLEAN DEFAULT FALSE,
    outcome_summary TEXT,
    follow_up_required BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, topic),
    INDEX (org_id, started_date),
    INDEX (participants)
);

-- Individual messages/statements within discussions
CREATE TABLE discussion_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES discussion_threads(id) ON DELETE CASCADE,
    speaker TEXT NOT NULL,
    message_date TIMESTAMPTZ NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(30) DEFAULT 'statement', -- 'question', 'answer', 'motion', 'objection', 'support'
    sentiment FLOAT DEFAULT 0.0,
    importance_score FLOAT DEFAULT 0.5,
    references_previous BOOLEAN DEFAULT FALSE,
    referenced_message_id UUID REFERENCES discussion_messages(id),
    decision_influence FLOAT DEFAULT 0.0, -- How much this influenced final decision
    entities_mentioned JSONB DEFAULT '[]',
    topics_covered TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (thread_id, message_date),
    INDEX (speaker),
    INDEX (message_type)
);

-- Outcomes and follow-up tracking
CREATE TABLE outcome_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    related_decision_id UUID REFERENCES decision_complete(id),
    related_record_id UUID REFERENCES complete_record(id),
    outcome_type VARCHAR(50), -- 'financial', 'operational', 'member_satisfaction', 'strategic'
    target_date DATE,
    actual_date DATE,
    success_metrics JSONB,
    actual_results JSONB,
    variance_analysis JSONB,
    lessons_learned TEXT,
    responsible_parties TEXT[],
    status VARCHAR(30) DEFAULT 'planned', -- 'planned', 'in_progress', 'completed', 'cancelled', 'delayed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, related_decision_id),
    INDEX (target_date),
    INDEX (status)
);

-- Knowledge capture for institutional wisdom
CREATE TABLE institutional_wisdom (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    wisdom_type VARCHAR(50), -- 'best_practice', 'lesson_learned', 'cultural_norm', 'process_improvement'
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    context TEXT,
    learned_from_decisions UUID[],
    learned_from_records UUID[],
    applicability_scope TEXT,
    importance_level VARCHAR(20) DEFAULT 'medium',
    validation_count INTEGER DEFAULT 1,
    last_referenced TIMESTAMPTZ,
    times_referenced INTEGER DEFAULT 0,
    effectiveness_rating FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    INDEX (org_id, wisdom_type),
    INDEX (org_id, importance_level),
    INDEX (learned_from_decisions),
    INDEX (last_referenced)
);

-- Relationship mapping between all entities
CREATE TABLE entity_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    entity_a_type VARCHAR(50), -- 'member', 'decision', 'record', 'committee', 'topic'
    entity_a_id TEXT,
    entity_b_type VARCHAR(50),
    entity_b_id TEXT,
    relationship_type VARCHAR(50), -- 'influences', 'collaborates_with', 'opposes', 'builds_on', 'conflicts_with'
    relationship_strength FLOAT DEFAULT 0.5,
    relationship_context TEXT,
    temporal_start DATE,
    temporal_end DATE,
    evidence_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, entity_a_type, entity_a_id),
    INDEX (org_id, entity_b_type, entity_b_id),
    INDEX (relationship_type),
    INDEX (relationship_strength)
);

-- Performance and analytics views
CREATE VIEW member_performance_summary AS
SELECT 
    mch.org_id,
    mch.member_name,
    mch.active_status,
    jsonb_array_length(mch.positions_held) as positions_count,
    jsonb_array_length(mch.committees_served) as committees_count,
    jsonb_array_length(mch.proposals_made) as proposals_count,
    (mch.effectiveness_scores->>'overall')::FLOAT as overall_effectiveness,
    mch.last_updated
FROM member_complete_history mch;

CREATE VIEW decision_success_metrics AS
SELECT 
    dc.org_id,
    dc.decision_type,
    COUNT(*) as total_decisions,
    AVG(dc.vote_margin) as avg_vote_margin,
    COUNT(*) FILTER (WHERE dc.unanimous = TRUE) as unanimous_count,
    AVG(dc.cost_variance) as avg_cost_variance,
    COUNT(*) FILTER (WHERE dc.would_repeat = TRUE) as would_repeat_count,
    AVG(dc.complexity_score) as avg_complexity
FROM decision_complete dc
GROUP BY dc.org_id, dc.decision_type;

CREATE VIEW discussion_analytics AS
SELECT 
    dt.org_id,
    dt.topic,
    COUNT(DISTINCT dt.id) as thread_count,
    COUNT(dm.id) as message_count,
    AVG(dm.importance_score) as avg_importance,
    COUNT(*) FILTER (WHERE dt.consensus_reached = TRUE) as consensus_reached_count,
    array_agg(DISTINCT unnest(dt.participants)) as all_participants
FROM discussion_threads dt
LEFT JOIN discussion_messages dm ON dt.id = dm.thread_id
GROUP BY dt.org_id, dt.topic;

-- Indexing for optimal performance
CREATE INDEX idx_complete_record_content_search ON complete_record USING gin(to_tsvector('english', content));
CREATE INDEX idx_decision_complete_search ON decision_complete USING gin(to_tsvector('english', description || ' ' || rationale));
CREATE INDEX idx_member_search ON member_complete_history USING gin(to_tsvector('english', member_name || ' ' || COALESCE(professional_background, '')));
CREATE INDEX idx_discussion_content_search ON discussion_messages USING gin(to_tsvector('english', content));

-- Comments documenting the schema
COMMENT ON TABLE complete_record IS 'Captures every institutional interaction, meeting, and communication for perfect recall';
COMMENT ON TABLE decision_complete IS 'Complete lifecycle tracking of every organizational decision with full context';
COMMENT ON TABLE member_complete_history IS 'Comprehensive history and analysis of every organization member';
COMMENT ON TABLE discussion_threads IS 'Threaded conversations and discussions with complete context preservation';
COMMENT ON TABLE discussion_messages IS 'Individual messages and statements within discussion threads';
COMMENT ON TABLE outcome_tracking IS 'Tracks outcomes and results of decisions and initiatives';
COMMENT ON TABLE institutional_wisdom IS 'Captures organizational knowledge and lessons learned';
COMMENT ON TABLE entity_relationships IS 'Maps relationships between all organizational entities';

-- Sample data for testing (optional)
-- This would be populated by the intelligence systems during operation