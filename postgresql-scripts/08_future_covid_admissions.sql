-- =============================================================================
-- 08_future_covid_admissions.sql
-- Tracks subsequent admissions for COVID patients with time gaps
-- =============================================================================

-- =============================================================================
-- DROP TABLE WITH CASCADE
-- =============================================================================

DROP TABLE IF EXISTS analysis.bridge_admission_covid_future CASCADE;
    -- CASCADE: Drops dependent objects (views, foreign keys, etc.)
    -- Ensures clean rebuild without manual dependency cleanup
    -- Important when this table is referenced by other objects

-- =============================================================================
-- MAIN QUERY: Three-Stage CTE Pipeline
-- =============================================================================

CREATE TABLE analysis.bridge_admission_covid_future AS 
WITH patient_admissions AS (
    -- =================================================================
    -- STAGE 1: Get all admissions for COVID patients efficiently
    -- =================================================================
    SELECT
        pat.subject_id,                  -- Patient identifier
        demo.hadm_id,                    -- Admission identifier
        demo.age,                        -- Age at this admission 
		
        EXISTS (
            SELECT 1 
            FROM analysis.ref_admission_covid cov 
            WHERE cov.hadm_id = demo.hadm_id
        ) AS has_covid_diagnosis,
            -- TRUE: This admission has COVID diagnosis
            -- FALSE: No COVID diagnosis
            -- EXISTS is faster than LEFT JOIN for existence checks
        
        pat.age AS baseline_age          -- Age at first COVID admission
            -- Renamed from min_age for clarity
            -- Used to calculate time gaps
    
    FROM analysis.ref_patient_covid pat
        -- Source: COVID patients with their first COVID admission
        -- Contains: subject_id, hadm_id (first), age (at first)
    
    INNER JOIN staging.demo_admission_filtered demo
        ON pat.subject_id = demo.subject_id
            -- Join condition: Same patient
        AND pat.age < demo.age
            -- Temporal filter: Only after first COVID
            -- Uses age as proxy for chronological ordering
    
    -- Result: All admissions for COVID patients (from after COVID onward)
),
admission_counts AS (
    -- =================================================================
    -- STAGE 2: Calculate admission counts and filter
    -- =================================================================
    SELECT
        subject_id,
        hadm_id,
        age,
        baseline_age,                    -- Age at first COVID admission
        has_covid_diagnosis,             -- Does this admission have COVID?
        
        -- Window function: Count total admissions per patient
        COUNT(*) OVER (PARTITION BY subject_id) AS total_admissions
            -- Includes all admissions (first + subsequent)
            -- Used to filter to multi-admission patients only
    
    FROM patient_admissions    
    -- Result: Subsequent admissions only, with counts
)
-- =================================================================
-- STAGE 3: Calculate time gaps and apply final filters
-- =================================================================
SELECT
    subject_id,                          -- Patient identifier
    hadm_id,                             -- Admission identifier (readmission)
    
    -- Time gap calculation (days since first COVID admission)
    ROUND((age - baseline_age) * 365, 2) AS age_gap,
        -- Formula: (Current age - Age at first COVID) × 365 days/year
        -- Approximation: Assumes 365 days/year (ignores leap years)
        -- Rounded to 2 decimal places
        -- Example: age_gap = 90 means 90 days after first COVID
    
    -- COVID diagnosis flag (convert boolean to integer)
    CASE WHEN has_covid_diagnosis THEN 1 ELSE 0 END AS covid
        -- 1 = This readmission has COVID diagnosis (reinfection/relapse)
        -- 0 = No COVID diagnosis at this readmission
		
FROM admission_counts
ORDER BY subject_id, age;
    -- Sort: By patient ID, then chronologically by age

-- Result: All subsequent admissions for COVID patients with readmissions
-- Grain: One row per readmission (excludes first COVID admission)
-- Columns: 4 (subject_id, hadm_id, age_gap, covid)

-- =============================================================================
-- OPTIMIZATION: COMPREHENSIVE INDEXING STRATEGY
-- =============================================================================

-- -------------------------------------------------------------------------
-- INDEX 1: Composite index for patient-level queries
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_covid_future_subject_gap 
    ON analysis.bridge_admission_covid_future(subject_id, age_gap);
    
    -- Purpose: Optimize queries that filter/group by patient and time
    -- Use cases:
    -- - SELECT * FROM bridge WHERE subject_id = X ORDER BY age_gap
    -- - GROUP BY subject_id with temporal filtering
    -- - Joining on subject_id with age_gap filters
    
    -- Query example:
    -- SELECT subject_id, COUNT(*), MIN(age_gap)
    -- FROM bridge_admission_covid_future
    -- WHERE subject_id IN (...)
    -- GROUP BY subject_id;
    
-- -------------------------------------------------------------------------
-- INDEX 2: Index on hadm_id for joins
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_covid_future_hadm_id 
    ON analysis.bridge_admission_covid_future(hadm_id);
    
    -- Purpose: Fast lookups by admission ID
    -- Use cases:
    -- - JOIN with demographics: WHERE hadm_id = demo.hadm_id
    -- - JOIN with lab data: WHERE hadm_id = lab.hadm_id
    -- - WHERE hadm_id IN (subquery)
    
    -- Query example:
    -- SELECT * FROM bridge_admission_covid_future
    -- WHERE hadm_id IN (SELECT hadm_id FROM some_other_table);

-- -------------------------------------------------------------------------
-- INDEX 3: Index on age_gap for temporal filtering
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_covid_future_age_gap 
    ON analysis.bridge_admission_covid_future(age_gap);
    
    -- Purpose: Optimize time-based filtering and sorting
    -- Use cases:
    -- - WHERE age_gap BETWEEN 30 AND 90 (30-90 day readmissions)
    -- - WHERE age_gap <= 30 (early readmissions)
    -- - ORDER BY age_gap (temporal sorting)
    
    -- Query example:
    -- SELECT COUNT(*) FROM bridge_admission_covid_future
    -- WHERE age_gap <= 30;  -- 30-day readmission rate

-- -------------------------------------------------------------------------
-- INDEX 4: Index on covid flag for group analysis
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_covid_future_covid_flag 
    ON analysis.bridge_admission_covid_future(covid);
    
    -- Purpose: Fast filtering and aggregation by COVID status
    -- Use cases:
    -- - WHERE covid = 1 (only COVID readmissions)
    -- - GROUP BY covid (compare COVID vs non-COVID readmissions)
    -- - COUNT by covid flag
    
    -- Query example:
    -- SELECT covid, COUNT(*), AVG(age_gap)
    -- FROM bridge_admission_covid_future
    -- GROUP BY covid;

-- -------------------------------------------------------------------------
-- INDEX 5: Composite index for common analysis pattern
-- -------------------------------------------------------------------------
CREATE INDEX idx_bridge_covid_future_gap_covid 
    ON analysis.bridge_admission_covid_future(age_gap, covid);
    
    -- Purpose: Optimize queries filtering by both time and COVID status
    -- Use cases:
    -- - WHERE age_gap <= 90 AND covid = 1 (early COVID reinfections)
    -- - Temporal analysis stratified by COVID status
    -- - Complex filtering on both dimensions
    
    -- Query example:
    -- SELECT COUNT(*) FROM bridge_admission_covid_future
    -- WHERE age_gap BETWEEN 30 AND 90 AND covid = 1;
    -- -- COVID reinfections within 30-90 days

-- -------------------------------------------------------------------------
-- UPDATE STATISTICS
-- -------------------------------------------------------------------------
ANALYZE analysis.bridge_admission_covid_future;
    -- Updates query planner statistics for optimal performance
    -- Essential after bulk INSERT or index creation
    -- Helps PostgreSQL choose the best execution plan