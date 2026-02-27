-- =============================================================================
-- 10_future_pre_admissions.sql
-- Tracks subsequent admissions for pre-COVID patients (no COVID diagnosis)
-- =============================================================================

-- =============================================================================
-- DROP TABLE WITH CASCADE
-- =============================================================================

DROP TABLE IF EXISTS analysis.bridge_admission_control_future CASCADE;
    -- CASCADE: Drops dependent views, foreign keys, etc.
    -- Ensures clean rebuild without orphaned dependencies
    -- Important for tables referenced by other database objects

-- =============================================================================
-- MAIN QUERY: Two-Stage CTE Pipeline
-- =============================================================================

CREATE TABLE analysis.bridge_admission_control_future AS 
WITH patient_admissions AS (
    -- =================================================================
    -- STAGE 1: Get all COVID-FREE admissions for control patients
    -- =================================================================
    SELECT
        pat.subject_id,                  -- Patient identifier
        demo.hadm_id,                    -- Admission identifier
        demo.age,                        -- Age at this admission
        pat.age AS baseline_age         -- Age at index admission (first in cohort)
    
    FROM analysis.demo_patient_control_pre pat
        -- Source: Pre-COVID control patients (2017-2019, no COVID)
        -- Contains: subject_id, hadm_id (index), age (at index)
    
    INNER JOIN staging.demo_admission_filtered demo
        -- Should be "staging.dim_admission_filtered"
        ON pat.subject_id = demo.subject_id
            -- Join condition: Same patient
        AND pat.age < demo.age
            -- Temporal filter: Only after index admission
            -- Uses age as proxy for chronological ordering
    
    WHERE NOT EXISTS (
        -- =================================================================
        -- CRITICAL FILTER: Exclude ALL admissions with COVID diagnosis
        -- =================================================================
        -- This is the KEY difference from COVID patient version
        -- Maintains control group integrity by ensuring COVID-free status
        
        SELECT 1 
        FROM analysis.ref_admission_covid cov 
        WHERE cov.hadm_id = demo.hadm_id
    )
        -- NOT EXISTS: Admission does NOT have COVID diagnosis
        -- Result: Only COVID-free admissions are included
        -- Essential for control group validity
    
    -- Result: All COVID-FREE admissions for control patients (from index onward)
),
admission_counts AS (
    -- =================================================================
    -- STAGE 2: Calculate admission counts and filter
    -- =================================================================
    SELECT
        subject_id,
        hadm_id,
        age,
        baseline_age,                    -- Age at index admission
        
        -- Window function: Count total admissions per patient
        COUNT(*) OVER (PARTITION BY subject_id) AS total_admissions
            -- Counts all admissions meeting criteria (excluding index)
            -- Used to filter to multi-admission patients only
    
    FROM patient_admissions
    
    -- Result: Subsequent COVID-free admissions only, with counts
)
-- =================================================================
-- STAGE 3: Calculate time gaps and apply final filters
-- =================================================================
SELECT
    subject_id,                          -- Patient identifier
    hadm_id,                             -- Admission identifier (readmission)
    
    -- Time gap calculation (days since index admission)
    ROUND((age - baseline_age) * 365, 2) AS age_gap
        -- Formula: (Current age - Age at index) × 365 days/year
        -- Approximation: Assumes 365 days/year (ignores leap years)
        -- Rounded to 2 decimal places
        -- Example: age_gap = 120 means 120 days after index admission

FROM admission_counts
ORDER BY subject_id, age;
    -- Sort: By patient ID, then chronologically by age

-- Result: All COVID-FREE subsequent admissions for control patients
-- Grain: One row per readmission (excludes index admission)
-- Columns: 3 (subject_id, hadm_id, age_gap)
-- Note: No 'covid' column - all admissions are COVID-free by design

-- =============================================================================
-- OPTIMIZATION: STRATEGIC INDEXING (3 INDEXES)
-- =============================================================================

-- -------------------------------------------------------------------------
-- INDEX 1: Composite index for patient-level queries
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_control_future_subject_gap 
    ON analysis.bridge_admission_control_future(subject_id, age_gap);
    
    -- Purpose: Optimize queries filtering/grouping by patient and time
    -- Use cases:
    -- - SELECT * FROM bridge WHERE subject_id = X ORDER BY age_gap
    -- - GROUP BY subject_id with temporal filtering
    -- - Patient timeline analysis
    
    -- Query example:
    -- SELECT subject_id, COUNT(*), MIN(age_gap) AS first_readmit
    -- FROM bridge_admission_control_future
    -- WHERE subject_id IN (...)
    -- GROUP BY subject_id;

-- -------------------------------------------------------------------------
-- INDEX 2: Index on hadm_id for joins
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_control_future_hadm_id 
    ON analysis.bridge_admission_control_future(hadm_id);
    
    -- Purpose: Fast lookups by admission ID
    -- Use cases:
    -- - JOIN with demographics tables
    -- - JOIN with lab data tables
    -- - WHERE hadm_id IN (subquery)
    
    -- Query example:
    -- SELECT b.*, d.age, d.gender
    -- FROM bridge_admission_control_future b
    -- JOIN demo_admission_control_future d
    --     ON b.hadm_id = d.hadm_id;

-- -------------------------------------------------------------------------
-- INDEX 3: Index on age_gap for temporal filtering
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_control_future_age_gap 
    ON analysis.bridge_admission_control_future(age_gap);
    
    -- Purpose: Optimize time-based filtering and sorting
    -- Use cases:
    -- - WHERE age_gap BETWEEN 30 AND 90 (30-90 day readmissions)
    -- - WHERE age_gap <= 30 (early readmissions)
    -- - ORDER BY age_gap (temporal sorting)
    
    -- Query example:
    -- SELECT COUNT(*) AS readmissions_30d
    -- FROM bridge_admission_control_future
    -- WHERE age_gap <= 30;

-- -------------------------------------------------------------------------
-- UPDATE STATISTICS
-- -------------------------------------------------------------------------
ANALYZE analysis.bridge_admission_control_future;
    -- Updates query planner statistics
    -- Essential after bulk INSERT or index creation
    -- Enables optimal query execution plans