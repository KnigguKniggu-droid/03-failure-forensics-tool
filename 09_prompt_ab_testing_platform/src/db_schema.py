"""Database schema for the prompt A/B testing platform.

PostgreSQL schema for prompt version registry, experiments, and outcomes.
Includes migration SQL for initial setup.
"""

from __future__ import annotations

INITIAL_MIGRATION = """
-- Prompt version registry
CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_id    VARCHAR(255) NOT NULL,
    version      VARCHAR(20) NOT NULL,
    system_prompt TEXT NOT NULL,
    user_template TEXT NOT NULL,
    model        VARCHAR(100) NOT NULL,
    temperature  REAL DEFAULT 0.7,
    max_tokens   INTEGER DEFAULT 1000,
    changelog    TEXT DEFAULT '',
    created_by   VARCHAR(255) DEFAULT '',
    created_at   TIMESTAMP DEFAULT NOW(),
    is_active    BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (prompt_id, version)
);

-- Experiment definitions
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id    VARCHAR(255) PRIMARY KEY,
    name             VARCHAR(500) NOT NULL,
    description      TEXT DEFAULT '',
    status           VARCHAR(20) DEFAULT 'draft',
    kill_switch_threshold REAL DEFAULT 0.05,
    statistical_significance REAL DEFAULT 0.05,
    sample_size_target INTEGER DEFAULT 1000,
    started_at       TIMESTAMP,
    ended_at         TIMESTAMP,
    created_at       TIMESTAMP DEFAULT NOW()
);

-- Experiment variants
CREATE TABLE IF NOT EXISTS experiment_variants (
    variant_id       VARCHAR(255) NOT NULL,
    experiment_id    VARCHAR(255) NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    allocation       VARCHAR(20) NOT NULL,
    prompt_id        VARCHAR(255) NOT NULL,
    prompt_version   VARCHAR(20) NOT NULL,
    traffic_percentage REAL NOT NULL CHECK (traffic_percentage >= 0 AND traffic_percentage <= 1),
    PRIMARY KEY (variant_id, experiment_id),
    FOREIGN KEY (prompt_id, prompt_version) REFERENCES prompt_versions(prompt_id, version)
);

-- Experiment outcomes
CREATE TABLE IF NOT EXISTS experiment_outcomes (
    id          SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    variant_id  VARCHAR(255) NOT NULL,
    user_id     VARCHAR(255) NOT NULL,
    success     BOOLEAN NOT NULL,
    score       REAL NOT NULL CHECK (score >= 0 AND score <= 1),
    latency_ms  REAL DEFAULT 0,
    timestamp   TIMESTAMP DEFAULT NOW()
);

-- Statistical results
CREATE TABLE IF NOT EXISTS statistical_results (
    experiment_id    VARCHAR(255) PRIMARY KEY REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    control_mean     REAL NOT NULL,
    treatment_mean   REAL NOT NULL,
    control_std      REAL NOT NULL,
    treatment_std    REAL NOT NULL,
    control_n        INTEGER NOT NULL,
    treatment_n      INTEGER NOT NULL,
    t_statistic      REAL NOT NULL,
    p_value          REAL NOT NULL,
    is_significant   BOOLEAN NOT NULL,
    ci_low           REAL NOT NULL,
    ci_high          REAL NOT NULL,
    winner           VARCHAR(20) NOT NULL,
    effect_size      REAL NOT NULL,
    computed_at      TIMESTAMP DEFAULT NOW()
);

-- Kill switch log
CREATE TABLE IF NOT EXISTS kill_switch_events (
    id               SERIAL PRIMARY KEY,
    experiment_id    VARCHAR(255) NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    triggered        BOOLEAN NOT NULL,
    reason           TEXT,
    performance_delta REAL NOT NULL,
    threshold        REAL NOT NULL,
    variant_killed   VARCHAR(255),
    timestamp        TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_experiment ON experiment_outcomes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_variant ON experiment_outcomes(variant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_user ON experiment_outcomes(user_id);
CREATE INDEX IF NOT EXISTS idx_variants_experiment ON experiment_variants(experiment_id);
"""

DROP_MIGRATION = """
DROP TABLE IF EXISTS kill_switch_events CASCADE;
DROP TABLE IF EXISTS statistical_results CASCADE;
DROP TABLE IF EXISTS experiment_outcomes CASCADE;
DROP TABLE IF EXISTS experiment_variants CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;
DROP TABLE IF EXISTS prompt_versions CASCADE;
"""
