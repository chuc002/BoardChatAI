-- Enhanced Institutional Memory System for Board Continuity MVP
-- This migration adds comprehensive tracking for decisions, patterns, and institutional knowledge
-- to achieve perfect recall of all board activities and decisions.

-- 1. DECISION REGISTRY TABLE
-- Tracks EVERY decision ever made with full context and traceability
CREATE TABLE IF NOT EXISTS decision_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id VARCHAR(100) UNIQUE NOT NULL, -- Human-readable ID (e.g., "2024-01-15-MEMBERSHIP-001")
    org_id UUID NOT NULL,
    created_by UUID,
    
    -- Decision Details
    date DATE NOT NULL,
    meeting_date DATE, -- When discussed/voted
    decision_type VARCHAR(50) NOT NULL, -- financial, membership, policy, governance, etc.
    title VARCHAR(255) NOT NULL,
    description TEXT,
    rationale TEXT, -- Why this decision was made
    
    -- Voting Information
    vote_count_for INTEGER DEFAULT 0,
    vote_count_against INTEGER DEFAULT 0,
    vote_count_abstain INTEGER DEFAULT 0,
    quorum_present BOOLEAN DEFAULT false,
    outcome VARCHAR(20) NOT NULL, -- approved, rejected, deferred, withdrawn
    
    -- Financial Information
    amount_involved DECIMAL(12,2), -- Dollar amount if applicable
    budget_category VARCHAR(100),
    fiscal_impact TEXT, -- Long-term financial implications
    
    -- Source Documentation
    source_document_id UUID, -- Links to documents table
    source_page_numbers INTEGER[], -- Specific pages where decision appears
    meeting_minutes_id UUID,
    
    -- Categorization and Search
    tags TEXT[], -- financial, membership, policy, emergency, routine, etc.
    priority_level VARCHAR(20) DEFAULT 'normal', -- low, normal, high, critical
    
    -- Context and Precedents
    precedent_decisions UUID[], -- Array of related decision IDs
    superseded_by UUID, -- If this decision was later overturned
    supersedes UUID[], -- Previous decisions this replaces
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (source_document_id) REFERENCES documents(id) ON DELETE SET NULL
);

-- Index for fast decision lookup
CREATE INDEX idx_decision_registry_org_date ON decision_registry(org_id, date);
CREATE INDEX idx_decision_registry_type ON decision_registry(org_id, decision_type);
CREATE INDEX idx_decision_registry_tags ON decision_registry USING GIN(tags);
CREATE INDEX idx_decision_registry_amount ON decision_registry(org_id, amount_involved) WHERE amount_involved IS NOT NULL;

-- 2. HISTORICAL PATTERNS TABLE  
-- Identifies and tracks recurring patterns in board decisions
CREATE TABLE IF NOT EXISTS historical_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    
    -- Pattern Identification
    pattern_type VARCHAR(100) NOT NULL, -- fee_increase, membership_transfer, emergency_vote, etc.
    pattern_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Pattern Statistics
    frequency_count INTEGER DEFAULT 0, -- How many times this pattern occurred
    success_rate DECIMAL(5,2) DEFAULT 0.00, -- Percentage of successful outcomes
    average_duration_days INTEGER, -- How long these decisions typically take
    
    -- Financial Patterns
    typical_amount DECIMAL(12,2), -- Common dollar amount for this pattern
    amount_range_min DECIMAL(12,2),
    amount_range_max DECIMAL(12,2),
    
    -- Timing Patterns
    common_months INTEGER[], -- Months when this typically occurs
    seasonal_trend VARCHAR(50), -- quarterly, annual, irregular, etc.
    
    -- Success/Failure Analysis
    common_failures TEXT[], -- Common reasons for failure
    success_factors TEXT[], -- What makes these decisions succeed
    risk_indicators TEXT[], -- Warning signs to watch for
    
    -- Instances and Examples
    decision_instances UUID[], -- All decision IDs that match this pattern
    best_example UUID, -- Most successful example
    worst_example UUID, -- Least successful example
    
    -- Context
    first_occurrence DATE, -- When this pattern first appeared
    last_occurrence DATE, -- Most recent occurrence
    trend VARCHAR(20) DEFAULT 'stable', -- increasing, decreasing, stable, volatile
    
    -- Metadata
    confidence_score DECIMAL(3,2) DEFAULT 0.50, -- How confident we are in this pattern
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (best_example) REFERENCES decision_registry(id) ON DELETE SET NULL,
    FOREIGN KEY (worst_example) REFERENCES decision_registry(id) ON DELETE SET NULL
);

CREATE INDEX idx_historical_patterns_org_type ON historical_patterns(org_id, pattern_type);
CREATE INDEX idx_historical_patterns_frequency ON historical_patterns(org_id, frequency_count DESC);
CREATE INDEX idx_historical_patterns_success ON historical_patterns(org_id, success_rate DESC);

-- 3. ENHANCED DOC_CHUNKS TABLE
-- Add new columns to existing doc_chunks for better context and analysis
ALTER TABLE doc_chunks 
ADD COLUMN IF NOT EXISTS section_type VARCHAR(100), -- reinstatement, fees, governance, membership, etc.
ADD COLUMN IF NOT EXISTS is_complete BOOLEAN DEFAULT false, -- Whether this chunk contains a complete section
ADD COLUMN IF NOT EXISTS contains_decision BOOLEAN DEFAULT false, -- Contains a formal decision
ADD COLUMN IF NOT EXISTS contains_precedent BOOLEAN DEFAULT false, -- Contains precedent or historical reference
ADD COLUMN IF NOT EXISTS decision_count INTEGER DEFAULT 0, -- Number of decisions mentioned
ADD COLUMN IF NOT EXISTS entities_mentioned JSONB DEFAULT '{}', -- Names, amounts, dates extracted
ADD COLUMN IF NOT EXISTS cross_references JSONB DEFAULT '[]', -- Links to related chunks
ADD COLUMN IF NOT EXISTS importance_score DECIMAL(3,2) DEFAULT 0.50, -- How important this chunk is
ADD COLUMN IF NOT EXISTS extracted_decisions UUID[], -- Direct links to decision_registry entries
ADD COLUMN IF NOT EXISTS temporal_context JSONB DEFAULT '{}'; -- Time-based context (before/after events)

-- Indexes for enhanced doc_chunks
CREATE INDEX IF NOT EXISTS idx_doc_chunks_section_type ON doc_chunks(section_type) WHERE section_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_doc_chunks_decisions ON doc_chunks(org_id) WHERE contains_decision = true;
CREATE INDEX IF NOT EXISTS idx_doc_chunks_precedents ON doc_chunks(org_id) WHERE contains_precedent = true;
CREATE INDEX IF NOT EXISTS idx_doc_chunks_entities ON doc_chunks USING GIN(entities_mentioned);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_importance ON doc_chunks(org_id, importance_score DESC);

-- 4. INSTITUTIONAL KNOWLEDGE TABLE
-- Captures learned insights and cultural context beyond formal decisions
CREATE TABLE IF NOT EXISTS institutional_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    
    -- Knowledge Classification
    knowledge_type VARCHAR(100) NOT NULL, -- cultural, procedural, historical, expertise, relationships
    category VARCHAR(100), -- governance, finance, membership, operations, etc.
    title VARCHAR(255) NOT NULL,
    
    -- Knowledge Content
    context TEXT NOT NULL, -- The actual knowledge/insight
    implications TEXT, -- What this means for future decisions
    examples TEXT[], -- Specific examples or cases
    
    -- Source and Validation
    learned_from VARCHAR(255), -- Where this knowledge came from
    source_documents UUID[], -- Supporting documentation
    confidence_score DECIMAL(3,2) DEFAULT 0.50, -- How reliable this knowledge is
    validation_count INTEGER DEFAULT 0, -- How many times this has been confirmed
    
    -- Temporal Context
    time_period_start DATE, -- When this knowledge became relevant
    time_period_end DATE, -- If no longer relevant
    is_current BOOLEAN DEFAULT true,
    
    -- Relationships
    related_decisions UUID[], -- Decisions that demonstrate this knowledge
    related_patterns UUID[], -- Patterns that support this knowledge
    contradictory_evidence UUID[], -- Decisions that contradict this
    
    -- Usage and Impact
    times_referenced INTEGER DEFAULT 0,
    last_referenced TIMESTAMP WITH TIME ZONE,
    impact_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    
    -- Metadata
    tags TEXT[], -- For categorization and search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
);

CREATE INDEX idx_institutional_knowledge_org_type ON institutional_knowledge(org_id, knowledge_type);
CREATE INDEX idx_institutional_knowledge_category ON institutional_knowledge(org_id, category);
CREATE INDEX idx_institutional_knowledge_confidence ON institutional_knowledge(org_id, confidence_score DESC);
CREATE INDEX idx_institutional_knowledge_current ON institutional_knowledge(org_id) WHERE is_current = true;
CREATE INDEX idx_institutional_knowledge_tags ON institutional_knowledge USING GIN(tags);

-- 5. BOARD MEMBER INSIGHTS TABLE
-- Tracks individual board member patterns, expertise, and contributions
CREATE TABLE IF NOT EXISTS board_member_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    
    -- Member Identification
    member_name VARCHAR(255) NOT NULL,
    member_id VARCHAR(100), -- If available from other systems
    member_title VARCHAR(100), -- President, Treasurer, etc.
    
    -- Tenure and Service
    start_date DATE,
    end_date DATE, -- NULL if still active
    total_tenure_months INTEGER,
    is_active BOOLEAN DEFAULT true,
    
    -- Committee and Role Information
    committee_assignments TEXT[], -- Finance, Membership, Governance, etc.
    leadership_roles TEXT[], -- Chair, Vice-Chair, Secretary, etc.
    role_history JSONB DEFAULT '[]', -- Historical role changes
    
    -- Expertise and Specializations
    expertise_areas TEXT[], -- finance, legal, operations, membership, etc.
    professional_background VARCHAR(255),
    key_contributions TEXT[], -- Notable contributions or achievements
    
    -- Voting and Decision Patterns
    total_decisions_participated INTEGER DEFAULT 0,
    voting_pattern_summary JSONB DEFAULT '{}', -- yes/no/abstain percentages
    frequently_supports TEXT[], -- Types of decisions they typically support
    frequently_opposes TEXT[], -- Types they typically oppose
    
    -- Decision Leadership
    decisions_authored UUID[], -- Decisions they proposed or championed
    decisions_influenced UUID[], -- Decisions where they had significant input
    successful_proposals INTEGER DEFAULT 0,
    failed_proposals INTEGER DEFAULT 0,
    
    -- Financial Decision Patterns
    avg_financial_decision_size DECIMAL(12,2),
    largest_decision_supported DECIMAL(12,2),
    fiscal_philosophy VARCHAR(50), -- conservative, moderate, progressive
    
    -- Collaboration Patterns
    frequent_allies TEXT[], -- Other members they often vote with
    frequent_opponents TEXT[], -- Members they often disagree with
    coalition_building_score DECIMAL(3,2) DEFAULT 0.50, -- How good at building consensus
    
    -- Meeting and Participation
    attendance_rate DECIMAL(5,2) DEFAULT 100.00,
    average_speaking_time INTEGER, -- Minutes per meeting (if available)
    questions_asked_per_meeting DECIMAL(4,1) DEFAULT 0.0,
    
    -- Historical Context and Notes
    cultural_context TEXT, -- Understanding of club culture and traditions
    institutional_memory TEXT, -- What they remember that others might not
    mentorship_relationships TEXT[], -- Who they mentor or are mentored by
    
    -- Performance Metrics
    effectiveness_score DECIMAL(3,2) DEFAULT 0.50,
    leadership_score DECIMAL(3,2) DEFAULT 0.50,
    innovation_score DECIMAL(3,2) DEFAULT 0.50, -- How often they propose new ideas
    
    -- Metadata
    data_quality_score DECIMAL(3,2) DEFAULT 0.50, -- How complete/reliable this data is
    last_analyzed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
);

CREATE INDEX idx_board_member_insights_org_active ON board_member_insights(org_id) WHERE is_active = true;
CREATE INDEX idx_board_member_insights_tenure ON board_member_insights(org_id, total_tenure_months DESC);
CREATE INDEX idx_board_member_insights_expertise ON board_member_insights USING GIN(expertise_areas);
CREATE INDEX idx_board_member_insights_effectiveness ON board_member_insights(org_id, effectiveness_score DESC);

-- 6. DECISION RELATIONSHIPS TABLE
-- Links decisions to board members who participated
CREATE TABLE IF NOT EXISTS decision_participation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    
    decision_id UUID NOT NULL,
    member_insight_id UUID NOT NULL,
    
    -- Participation Details
    vote VARCHAR(20), -- for, against, abstain, absent
    participation_level VARCHAR(50) DEFAULT 'voted', -- proposed, seconded, discussed, voted, absent
    speaking_time_minutes INTEGER DEFAULT 0,
    influence_level VARCHAR(20) DEFAULT 'none', -- none, low, medium, high, decisive
    
    -- Position and Arguments
    position_summary TEXT, -- Their stated position
    key_arguments TEXT[], -- Main points they made
    questions_asked TEXT[], -- Questions they raised
    concerns_raised TEXT[], -- Specific concerns they voiced
    
    -- Relationships
    supported_by TEXT[], -- Other members who agreed with them
    opposed_by TEXT[], -- Members who disagreed
    
    -- Context
    was_pivotal_vote BOOLEAN DEFAULT false, -- Did their vote decide the outcome?
    changed_mind_during_discussion BOOLEAN DEFAULT false,
    final_position VARCHAR(20), -- Their ultimate stance
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (decision_id) REFERENCES decision_registry(id) ON DELETE CASCADE,
    FOREIGN KEY (member_insight_id) REFERENCES board_member_insights(id) ON DELETE CASCADE,
    
    UNIQUE(decision_id, member_insight_id)
);

CREATE INDEX idx_decision_participation_decision ON decision_participation(decision_id);
CREATE INDEX idx_decision_participation_member ON decision_participation(member_insight_id);
CREATE INDEX idx_decision_participation_pivotal ON decision_participation(org_id) WHERE was_pivotal_vote = true;

-- 7. CONTEXTUAL INSIGHTS VIEW
-- Combines data from multiple tables for comprehensive analysis
CREATE OR REPLACE VIEW v_comprehensive_decisions AS
SELECT 
    dr.id,
    dr.decision_id,
    dr.org_id,
    dr.date,
    dr.decision_type,
    dr.title,
    dr.description,
    dr.outcome,
    dr.amount_involved,
    dr.tags,
    
    -- Pattern Information
    hp.pattern_name,
    hp.success_rate as pattern_success_rate,
    hp.typical_amount as pattern_typical_amount,
    
    -- Document Context
    d.filename as source_document,
    dr.source_page_numbers,
    
    -- Participation Summary
    COUNT(dp.id) as total_participants,
    COUNT(CASE WHEN dp.vote = 'for' THEN 1 END) as votes_for,
    COUNT(CASE WHEN dp.vote = 'against' THEN 1 END) as votes_against,
    COUNT(CASE WHEN dp.vote = 'abstain' THEN 1 END) as votes_abstain,
    
    -- Member Insights
    STRING_AGG(DISTINCT bmi.member_name, ', ') as key_participants,
    STRING_AGG(DISTINCT bmi.member_name, ', ') FILTER (WHERE dp.influence_level IN ('high', 'decisive')) as influential_members
    
FROM decision_registry dr
LEFT JOIN historical_patterns hp ON dr.decision_type = hp.pattern_type AND dr.org_id = hp.org_id
LEFT JOIN documents d ON dr.source_document_id = d.id
LEFT JOIN decision_participation dp ON dr.id = dp.decision_id
LEFT JOIN board_member_insights bmi ON dp.member_insight_id = bmi.id
GROUP BY dr.id, dr.decision_id, dr.org_id, dr.date, dr.decision_type, dr.title, 
         dr.description, dr.outcome, dr.amount_involved, dr.tags, hp.pattern_name, 
         hp.success_rate, hp.typical_amount, d.filename, dr.source_page_numbers;

-- 8. TRIGGER FUNCTIONS FOR AUTOMATIC UPDATES
-- Update patterns when new decisions are added
CREATE OR REPLACE FUNCTION update_historical_patterns()
RETURNS TRIGGER AS $$
BEGIN
    -- Update pattern frequency and success rates
    INSERT INTO historical_patterns (
        org_id, pattern_type, pattern_name, frequency_count, 
        last_occurrence, decision_instances
    )
    VALUES (
        NEW.org_id, 
        NEW.decision_type, 
        INITCAP(REPLACE(NEW.decision_type, '_', ' ')) || ' Decisions',
        1,
        NEW.date,
        ARRAY[NEW.id]
    )
    ON CONFLICT (org_id, pattern_type) 
    DO UPDATE SET
        frequency_count = historical_patterns.frequency_count + 1,
        last_occurrence = NEW.date,
        decision_instances = array_append(historical_patterns.decision_instances, NEW.id),
        success_rate = (
            SELECT 
                (COUNT(*) FILTER (WHERE outcome IN ('approved')) * 100.0 / COUNT(*))
            FROM decision_registry 
            WHERE org_id = NEW.org_id AND decision_type = NEW.decision_type
        ),
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_patterns
    AFTER INSERT ON decision_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_historical_patterns();

-- 9. SAMPLE DATA AND CONFIGURATION
-- Create some initial institutional knowledge entries
INSERT INTO institutional_knowledge (org_id, knowledge_type, category, title, context, confidence_score, tags)
SELECT 
    '00000000-0000-0000-0000-000000000001'::UUID,
    'procedural',
    'governance',
    'Quorum Requirements',
    'Board meetings require at least 60% of active members present to conduct official business. Emergency meetings can be called with 48-hour notice.',
    0.95,
    ARRAY['governance', 'meetings', 'quorum']
WHERE NOT EXISTS (
    SELECT 1 FROM institutional_knowledge 
    WHERE title = 'Quorum Requirements' 
    AND org_id = '00000000-0000-0000-0000-000000000001'::UUID
);

-- Add comments to tables for documentation
COMMENT ON TABLE decision_registry IS 'Comprehensive registry of all board decisions with full context and traceability';
COMMENT ON TABLE historical_patterns IS 'Analysis of recurring decision patterns and their success rates';
COMMENT ON TABLE institutional_knowledge IS 'Captured organizational wisdom and cultural context';
COMMENT ON TABLE board_member_insights IS 'Individual board member analysis including voting patterns and expertise';
COMMENT ON TABLE decision_participation IS 'Links decisions to participating board members with detailed context';

-- Final verification
SELECT 
    schemaname, 
    tablename, 
    tableowner 
FROM pg_tables 
WHERE tablename IN (
    'decision_registry', 
    'historical_patterns', 
    'institutional_knowledge', 
    'board_member_insights',
    'decision_participation'
)
ORDER BY tablename;