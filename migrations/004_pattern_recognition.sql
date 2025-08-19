-- Pattern Recognition Database Tables for Governance Intelligence
-- Creates tables to track decision patterns, governance trends, and voting behaviors

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Decision patterns tracking - identifies recurring patterns in governance decisions
CREATE TABLE decision_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(50), -- 'approval_sequence', 'budget_cycle', 'vendor_selection', 'membership_category', 'fee_adjustment'
    trigger_conditions JSONB, -- Conditions that typically trigger this pattern
    typical_timeline_days INTEGER, -- Average time from proposal to resolution
    success_rate FLOAT DEFAULT 0.0, -- Percentage of decisions following this pattern that succeed
    approval_rate FLOAT DEFAULT 0.0, -- Percentage that get approved
    average_cost DECIMAL(12,2), -- Average financial impact
    cost_variance_pct FLOAT, -- Typical variance from projected costs
    risk_factors JSONB, -- Common risk factors for this pattern
    success_factors JSONB, -- Factors that correlate with success
    mitigation_strategies JSONB, -- Recommended strategies to improve outcomes
    examples JSONB, -- Array of decision IDs that match this pattern
    occurrence_frequency VARCHAR(20) DEFAULT 'occasional', -- 'frequent', 'occasional', 'rare'
    seasonal_patterns JSONB, -- Monthly/quarterly patterns if applicable
    stakeholder_resistance_level FLOAT DEFAULT 0.5, -- Typical resistance encountered
    confidence_score FLOAT DEFAULT 0.5, -- Confidence in pattern validity
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    INDEX (org_id, pattern_type),
    INDEX (org_id, success_rate),
    INDEX (pattern_name),
    INDEX (occurrence_frequency)
);

-- Governance trends tracking - monitors long-term organizational trends
CREATE TABLE governance_trends (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    trend_name VARCHAR(100) NOT NULL,
    trend_type VARCHAR(50), -- 'fees', 'membership', 'facilities', 'events', 'governance', 'financial'
    trend_direction VARCHAR(20), -- 'increasing', 'decreasing', 'stable', 'cyclical', 'volatile'
    start_date DATE NOT NULL,
    end_date DATE, -- NULL if trend is ongoing
    magnitude FLOAT, -- Quantified measure of the trend strength
    rate_of_change FLOAT, -- Rate at which trend is progressing
    correlation_factors JSONB, -- Factors that correlate with this trend
    external_influences JSONB, -- External factors affecting the trend
    impact_assessment TEXT, -- Qualitative assessment of trend impact
    quantitative_metrics JSONB, -- Specific metrics and measurements
    supporting_decisions UUID[], -- Decision IDs that support/evidence this trend
    supporting_records UUID[], -- Record IDs that evidence this trend
    predictive_indicators JSONB, -- Leading indicators for trend continuation
    inflection_points JSONB, -- Points where trend changed direction
    cyclical_patterns JSONB, -- If cyclical, pattern details
    confidence_level FLOAT DEFAULT 0.5, -- Confidence in trend analysis
    volatility_score FLOAT DEFAULT 0.0, -- Measure of trend volatility
    trend_strength VARCHAR(20) DEFAULT 'moderate', -- 'weak', 'moderate', 'strong'
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    analyzed_by TEXT,
    INDEX (org_id, trend_type),
    INDEX (org_id, start_date),
    INDEX (trend_direction),
    INDEX (confidence_level)
);

-- Member voting patterns - comprehensive voting behavior analysis
CREATE TABLE voting_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    member_name VARCHAR(255) NOT NULL,
    member_id TEXT,
    current_position VARCHAR(50), -- 'president', 'treasurer', 'secretary', 'member', 'chairman'
    position_history JSONB, -- Historical positions held
    voting_tendencies JSONB, -- {'conservative': 0.8, 'fiscal_strict': 0.9, 'change_resistant': 0.3}
    policy_preferences JSONB, -- Preferences on various policy areas
    committee_preferences JSONB, -- Committee involvement and performance
    issue_positions JSONB, -- Known positions on specific issues
    voting_frequency FLOAT DEFAULT 1.0, -- How often they vote (1.0 = always present)
    influence_score FLOAT DEFAULT 0.5, -- Measure of influence on other members
    consistency_score FLOAT DEFAULT 0.5, -- How consistent their voting is
    predictability_score FLOAT DEFAULT 0.5, -- How predictable their votes are
    alignment_patterns JSONB, -- Which members they typically align with
    opposition_patterns JSONB, -- Which members they typically oppose
    swing_vote_frequency FLOAT DEFAULT 0.0, -- How often they are the deciding vote
    proposal_success_rate FLOAT DEFAULT 0.0, -- Success rate of their proposals
    amendment_frequency FLOAT DEFAULT 0.0, -- How often they propose amendments
    fiscal_voting_pattern VARCHAR(20), -- 'conservative', 'liberal', 'moderate', 'variable'
    change_adoption_speed VARCHAR(20), -- 'early', 'majority', 'late', 'resistant'
    collaboration_style VARCHAR(30), -- 'consensus_builder', 'independent', 'follower', 'contrarian'
    communication_effectiveness FLOAT DEFAULT 0.5, -- How effectively they communicate positions
    expertise_areas TEXT[], -- Areas where they have demonstrated expertise
    voting_blocs JSONB, -- Voting blocs they participate in
    seasonal_patterns JSONB, -- Any seasonal voting patterns
    tenure_years FLOAT, -- Years of service
    attendance_rate FLOAT DEFAULT 1.0, -- Meeting attendance rate
    preparation_level VARCHAR(20) DEFAULT 'moderate', -- 'high', 'moderate', 'low'
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    analyzed_by TEXT,
    UNIQUE(org_id, member_name),
    INDEX (org_id, member_name),
    INDEX (org_id, current_position),
    INDEX (influence_score),
    INDEX (consistency_score),
    INDEX (fiscal_voting_pattern)
);

-- Financial patterns tracking - patterns in financial decisions and outcomes
CREATE TABLE financial_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    pattern_name VARCHAR(100) NOT NULL,
    financial_category VARCHAR(50), -- 'fees', 'capital_expenditure', 'operating_costs', 'revenue'
    pattern_description TEXT,
    typical_amounts JSONB, -- Range and typical amounts for this pattern
    timing_patterns JSONB, -- When these patterns typically occur
    approval_thresholds JSONB, -- Decision thresholds for different amounts
    cost_overrun_tendency FLOAT DEFAULT 0.0, -- Tendency for cost overruns
    budget_accuracy FLOAT DEFAULT 0.8, -- Historical budget accuracy
    roi_patterns JSONB, -- Return on investment patterns
    payback_periods JSONB, -- Typical payback periods
    risk_assessment JSONB, -- Financial risk patterns
    member_resistance_by_amount JSONB, -- Resistance patterns by amount
    success_predictors JSONB, -- Factors that predict financial success
    failure_predictors JSONB, -- Factors that predict problems
    seasonal_effects JSONB, -- Seasonal financial patterns
    inflation_adjustments JSONB, -- How pattern adjusts for inflation
    comparative_benchmarks JSONB, -- How pattern compares to industry benchmarks
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, financial_category),
    INDEX (org_id, pattern_name)
);

-- Communication patterns - how decisions are communicated and discussed
CREATE TABLE communication_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    communication_type VARCHAR(50), -- 'formal_meeting', 'email_discussion', 'informal_chat', 'committee_review'
    topic_category VARCHAR(50), -- 'financial', 'membership', 'governance', 'facilities'
    discussion_length_avg INTEGER, -- Average discussion length in minutes
    participant_count_avg INTEGER, -- Average number of participants
    consensus_time_avg INTEGER, -- Average time to reach consensus
    conflict_frequency FLOAT DEFAULT 0.0, -- How often conflicts arise
    resolution_methods JSONB, -- Common conflict resolution methods
    information_quality FLOAT DEFAULT 0.5, -- Quality of information shared
    decision_clarity FLOAT DEFAULT 0.5, -- How clearly decisions are communicated
    follow_up_effectiveness FLOAT DEFAULT 0.5, -- How well follow-ups are handled
    stakeholder_satisfaction FLOAT DEFAULT 0.5, -- Stakeholder satisfaction with communication
    communication_channels JSONB, -- Channels used for this type of communication
    effectiveness_metrics JSONB, -- Metrics for communication effectiveness
    improvement_opportunities JSONB, -- Identified areas for improvement
    best_practices JSONB, -- Documented best practices
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, communication_type),
    INDEX (org_id, topic_category)
);

-- Risk patterns - patterns in risk identification and management
CREATE TABLE risk_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    risk_category VARCHAR(50), -- 'financial', 'operational', 'regulatory', 'reputational', 'strategic'
    risk_name VARCHAR(100) NOT NULL,
    probability_range JSONB, -- Historical probability ranges
    impact_range JSONB, -- Historical impact ranges
    detection_patterns JSONB, -- How risks are typically detected
    response_patterns JSONB, -- Common response strategies
    mitigation_effectiveness JSONB, -- Effectiveness of mitigation strategies
    early_warning_indicators JSONB, -- Early warning signs
    escalation_patterns JSONB, -- How risks escalate
    resolution_timeframes JSONB, -- Typical resolution times
    cost_of_mitigation JSONB, -- Typical costs to address
    stakeholder_impact JSONB, -- Impact on different stakeholders
    regulatory_implications JSONB, -- Regulatory considerations
    insurance_considerations JSONB, -- Insurance coverage implications
    lessons_learned JSONB, -- Key lessons from past occurrences
    prevention_strategies JSONB, -- Proven prevention strategies
    last_occurrence DATE, -- Last time this risk pattern occurred
    frequency_analysis JSONB, -- Analysis of occurrence frequency
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (org_id, risk_category),
    INDEX (org_id, last_occurrence)
);

-- Performance and analytics views for pattern recognition
CREATE VIEW pattern_effectiveness_summary AS
SELECT 
    dp.org_id,
    dp.pattern_type,
    dp.pattern_name,
    dp.success_rate,
    dp.approval_rate,
    dp.confidence_score,
    dp.occurrence_frequency,
    jsonb_array_length(dp.examples) as example_count,
    dp.last_updated
FROM decision_patterns dp
ORDER BY dp.success_rate DESC, dp.confidence_score DESC;

CREATE VIEW member_influence_ranking AS
SELECT 
    vp.org_id,
    vp.member_name,
    vp.current_position,
    vp.influence_score,
    vp.consistency_score,
    vp.predictability_score,
    vp.proposal_success_rate,
    vp.voting_frequency,
    vp.tenure_years
FROM voting_patterns vp
ORDER BY vp.influence_score DESC, vp.consistency_score DESC;

CREATE VIEW trend_impact_analysis AS
SELECT 
    gt.org_id,
    gt.trend_type,
    gt.trend_direction,
    gt.magnitude,
    gt.confidence_level,
    gt.trend_strength,
    CASE 
        WHEN gt.end_date IS NULL THEN 'ongoing'
        ELSE 'completed'
    END as trend_status,
    CASE 
        WHEN gt.end_date IS NULL THEN 
            EXTRACT(days FROM NOW() - gt.start_date::timestamp)
        ELSE 
            EXTRACT(days FROM gt.end_date::timestamp - gt.start_date::timestamp)
    END as duration_days
FROM governance_trends gt
ORDER BY gt.confidence_level DESC, gt.magnitude DESC;

CREATE VIEW risk_priority_matrix AS
SELECT 
    rp.org_id,
    rp.risk_category,
    rp.risk_name,
    (rp.probability_range->>'max')::FLOAT as max_probability,
    (rp.impact_range->>'max')::FLOAT as max_impact,
    ((rp.probability_range->>'max')::FLOAT * (rp.impact_range->>'max')::FLOAT) as risk_score,
    rp.last_occurrence,
    CASE 
        WHEN rp.last_occurrence IS NULL THEN 999
        ELSE EXTRACT(days FROM NOW() - rp.last_occurrence::timestamp)
    END as days_since_last_occurrence
FROM risk_patterns rp
ORDER BY risk_score DESC, days_since_last_occurrence ASC;

-- Indexing for optimal pattern recognition performance
CREATE INDEX idx_decision_patterns_search ON decision_patterns USING gin(to_tsvector('english', pattern_name || ' ' || COALESCE(trigger_conditions::text, '')));
CREATE INDEX idx_governance_trends_search ON governance_trends USING gin(to_tsvector('english', trend_name || ' ' || COALESCE(impact_assessment, '')));
CREATE INDEX idx_voting_patterns_search ON voting_patterns USING gin(to_tsvector('english', member_name || ' ' || COALESCE(expertise_areas::text, '')));

-- Pattern correlation indexes for advanced analytics
CREATE INDEX idx_pattern_correlation ON decision_patterns(org_id, pattern_type, success_rate);
CREATE INDEX idx_trend_correlation ON governance_trends(org_id, trend_type, trend_direction, magnitude);
CREATE INDEX idx_voting_correlation ON voting_patterns(org_id, influence_score, consistency_score);

-- Comments documenting the pattern recognition schema
COMMENT ON TABLE decision_patterns IS 'Tracks recurring patterns in organizational decisions with success metrics and predictive factors';
COMMENT ON TABLE governance_trends IS 'Monitors long-term organizational trends with quantitative analysis and predictive indicators';
COMMENT ON TABLE voting_patterns IS 'Comprehensive analysis of member voting behaviors, influence, and predictability patterns';
COMMENT ON TABLE financial_patterns IS 'Patterns in financial decisions, budget accuracy, and cost management behaviors';
COMMENT ON TABLE communication_patterns IS 'Analysis of communication effectiveness and discussion patterns in governance';
COMMENT ON TABLE risk_patterns IS 'Risk identification, management patterns, and mitigation strategy effectiveness tracking';

-- Sample data inserts for testing pattern recognition (optional)
-- INSERT INTO decision_patterns (org_id, pattern_name, pattern_type, success_rate, approval_rate) 
-- VALUES ('63602dc6-defe-4355-b66c-aa6b3b1273e3', 'Annual Fee Increase', 'fee_adjustment', 0.85, 0.75);

-- Performance optimization settings
-- These help with pattern matching and analytics queries
SET work_mem = '256MB';
SET random_page_cost = 1.1;
SET effective_cache_size = '2GB';