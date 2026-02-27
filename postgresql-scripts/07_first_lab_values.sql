-- =============================================================================
-- 07_first_lab_values.sql
-- Extracts first occurrence of each lab item per admission for all groups
-- =============================================================================

-- =============================================================================
-- FIRST LAB VALUES 1: COVID PATIENTS
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.fact_lab_covid_first CASCADE;

-- Create table with first lab measurement per admission-lab pair
CREATE TABLE analysis.fact_lab_covid_first AS
WITH ranked_rows AS (
    -- =================================================================
    -- STEP 1: Rank all lab measurements
    -- =================================================================
    SELECT
        *,                               -- All columns from fact_lab_covid
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id, lab_itemid
                -- Group by: Each unique (admission, lab test) combination
                -- Creates separate ranking windows for:
                --   - Patient A, Admission 1, Creatinine
                --   - Patient A, Admission 1, Glucose
                --   - Patient A, Admission 2, Creatinine
                --   - etc.
            
            ORDER BY lab_charttime ASC
                -- Sort by: Earliest timestamp first
                -- rn = 1 will be the first measurement
                -- rn = 2 will be the second measurement, etc.
        ) AS rn
            -- Row number: 1, 2, 3, ... within each partition
    
    FROM analysis.fact_lab_covid
        -- Source: All lab measurements for COVID patients
)
-- =================================================================
-- STEP 2: Filter to keep only first measurement
-- =================================================================
SELECT *
FROM ranked_rows
WHERE rn = 1;
    -- Filter: Keep only rows where row_number = 1
    -- Result: Earliest measurement for each (admission, lab) pair
    -- Drops the 'rn' column is NOT included in final table

-- Output: One row per unique (hadm_id, lab_itemid) combination
-- Grain: First measurement of each lab test per admission

-- Create indexes for downstream queries
CREATE INDEX idx_fact_lab_covid_first_hadm_id 
    ON analysis.fact_lab_covid_first(hadm_id);

CREATE INDEX idx_fact_lab_covid_first_itemid 
    ON analysis.fact_lab_covid_first(lab_itemid);

-- Composite index for pivot operations
CREATE INDEX idx_fact_lab_covid_first_hadm_item 
    ON analysis.fact_lab_covid_first(hadm_id, lab_itemid);

-- Index for numeric value analysis
CREATE INDEX idx_fact_lab_covid_first_valuenum 
    ON analysis.fact_lab_covid_first(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_covid_first;


-- =============================================================================
-- FIRST LAB VALUES 2: POST-COVID CONTROL GROUP
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_post_first;

-- Create table with first lab measurement per admission-lab pair
CREATE TABLE analysis.fact_lab_control_post_first AS
WITH ranked_rows AS (
    -- =================================================================
    -- IDENTICAL LOGIC TO COVID TABLE
    -- =================================================================
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id, lab_itemid
            ORDER BY lab_charttime ASC
        ) AS rn
    FROM analysis.fact_lab_control_post
        -- Source: All lab measurements for post-COVID controls
)
SELECT *
FROM ranked_rows
WHERE rn = 1;

-- Output: First measurement of each lab test per admission (post-COVID controls)

-- Create indexes
CREATE INDEX idx_fact_lab_control_post_first_hadm_id 
    ON analysis.fact_lab_control_post_first(hadm_id);

CREATE INDEX idx_fact_lab_control_post_first_itemid 
    ON analysis.fact_lab_control_post_first(lab_itemid);

CREATE INDEX idx_fact_lab_control_post_first_hadm_item 
    ON analysis.fact_lab_control_post_first(hadm_id, lab_itemid);

CREATE INDEX idx_fact_lab_control_post_first_valuenum 
    ON analysis.fact_lab_control_post_first(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_post_first;

-- =============================================================================
-- FIRST LAB VALUES 3: PRE-COVID CONTROL GROUP
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_pre_first;

-- Create table with first lab measurement per admission-lab pair
CREATE TABLE analysis.fact_lab_control_pre_first AS
WITH ranked_rows AS (
    -- =================================================================
    -- IDENTICAL LOGIC TO COVID TABLE
    -- =================================================================
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id, lab_itemid
            ORDER BY lab_charttime ASC
        ) AS rn
    FROM analysis.fact_lab_control_pre
        -- Source: All lab measurements for pre-COVID controls
)
SELECT *
FROM ranked_rows
WHERE rn = 1;

-- Output: First measurement of each lab test per admission (pre-COVID controls)

-- Create indexes
CREATE INDEX idx_fact_lab_control_pre_first_hadm_id 
    ON analysis.fact_lab_control_pre_first(hadm_id);

CREATE INDEX idx_fact_lab_control_pre_first_itemid 
    ON analysis.fact_lab_control_pre_first(lab_itemid);

CREATE INDEX idx_fact_lab_control_pre_first_hadm_item 
    ON analysis.fact_lab_control_pre_first(hadm_id, lab_itemid);

CREATE INDEX idx_fact_lab_control_pre_first_valuenum 
    ON analysis.fact_lab_control_pre_first(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_pre_first;