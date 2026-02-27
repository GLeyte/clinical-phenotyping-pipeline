-- =============================================================================
-- 06_lab_data.sql
-- Extracts selected lab events for COVID, post-COVID, and pre-COVID groups
-- =============================================================================

-- =============================================================================
-- LAB DATA 1: COVID PATIENTS
-- =============================================================================

-- Drop existing COVID lab events table if present
DROP TABLE IF EXISTS analysis.fact_lab_covid CASCADE;

-- Create lab events table for COVID patient admissions
CREATE TABLE analysis.fact_lab_covid AS
SELECT
    -- =================================================================
    -- CORE IDENTIFIERS
    -- =================================================================
    lab.hadm_id,                         -- Hospital admission ID
    
    -- =================================================================
    -- LAB EVENT DETAILS (prefixed with "lab_" for clarity)
    -- =================================================================
    lab.charttime AS lab_charttime,     -- Timestamp of lab measurement
    lab.itemid AS lab_itemid,           -- Lab test identifier (links to ref_lab_item)
    lab.value AS lab_value,             -- Text representation of result
    lab.valuenum AS lab_valuenum,       -- Numeric value (NULL if non-numeric)
    lab.valueuom AS lab_unit,           -- Unit of measurement (e.g., mg/dL, mmol/L)
    
    -- =================================================================
    -- REFERENCE RANGES & FLAGS
    -- =================================================================
    lab.ref_range_lower AS lab_ref_range_lower,  -- Lower bound of normal range
    lab.ref_range_upper AS lab_ref_range_upper,  -- Upper bound of normal range
    lab.flag AS lab_flag                         -- Abnormal flag (e.g., 'abnormal', 'delta')

FROM staging.fact_lab_filtered AS lab
    -- Source: Pre-filtered lab events (2017-2022, from 02_filter_2017_2022.sql)

-- =================================================================
-- FILTER 1: Admission-level filtering (COVID patients only)
-- =================================================================
INNER JOIN analysis.demo_patient_covid demo
    ON lab.hadm_id = demo.hadm_id
    -- Reference: First COVID admission per patient


-- =================================================================
-- FILTER 2: Lab item filtering (top 61 items only)
-- =================================================================   
INNER JOIN analysis.ref_lab_item ref
    ON lab.itemid = ref.itemid;
    -- Reference: 61 most common lab tests in COVID cohort

-- Result: All lab measurements for COVID patients, limited to top 61 tests
-- Grain: One row per lab measurement (multiple per patient/admission)

-- Create strategic indexes
CREATE INDEX idx_fact_lab_covid_hadm_id 
    ON analysis.fact_lab_covid(hadm_id);

CREATE INDEX idx_fact_lab_covid_itemid 
    ON analysis.fact_lab_covid(lab_itemid);

-- Composite index for window function queries (used in script 07)
CREATE INDEX idx_fact_lab_covid_hadm_item_time 
    ON analysis.fact_lab_covid(hadm_id, lab_itemid, lab_charttime);

-- Index for numeric value queries
CREATE INDEX idx_fact_lab_covid_valuenum 
    ON analysis.fact_lab_covid(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_covid;

-- =============================================================================
-- LAB DATA 2: POST-COVID CONTROL GROUP
-- =============================================================================

-- Drop existing post-COVID control lab events table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_post CASCADE;

-- Create lab events table for post-COVID control admissions (2020-2022, no COVID)
CREATE TABLE analysis.fact_lab_control_post AS
SELECT
    -- =================================================================
    -- CORE IDENTIFIERS
    -- =================================================================
    lab.hadm_id,                         -- Hospital admission ID
    
    -- =================================================================
    -- LAB EVENT DETAILS (prefixed with "lab_" for clarity)
    -- =================================================================
    lab.charttime AS lab_charttime,     -- Timestamp of lab measurement
    lab.itemid AS lab_itemid,           -- Lab test identifier (links to ref_lab_item)
    lab.value AS lab_value,             -- Text representation of result
    lab.valuenum AS lab_valuenum,       -- Numeric value (NULL if non-numeric)
    lab.valueuom AS lab_unit,           -- Unit of measurement (e.g., mg/dL, mmol/L)
    
    -- =================================================================
    -- REFERENCE RANGES & FLAGS
    -- =================================================================
    lab.ref_range_lower AS lab_ref_range_lower,  -- Lower bound of normal range
    lab.ref_range_upper AS lab_ref_range_upper,  -- Upper bound of normal range
    lab.flag AS lab_flag                         -- Abnormal flag (e.g., 'abnormal', 'delta')

FROM staging.fact_lab_filtered AS lab
    -- Source: Pre-filtered lab events (2017-2022)

-- =================================================================
-- FILTER 1: Admission-level filtering (post-COVID control patients)
-- =================================================================
INNER JOIN analysis.demo_patient_control_post demo
    ON lab.hadm_id = demo.hadm_id
    -- Reference: Post-COVID control admissions (2020-2022, no COVID diagnosis)

-- =================================================================
-- FILTER 2: Lab item filtering (same top 61 items)
-- =================================================================
INNER JOIN analysis.ref_lab_item ref
    ON lab.itemid = ref.itemid;
    -- Reference: Same 61 lab tests used for COVID cohort

-- Result: All lab measurements for post-COVID controls, limited to top 61 tests
-- Grain: One row per lab measurement

-- Create strategic indexes
CREATE INDEX idx_fact_lab_control_post_hadm_id 
    ON analysis.fact_lab_control_post(hadm_id);

CREATE INDEX idx_fact_lab_control_post_itemid 
    ON analysis.fact_lab_control_post(lab_itemid);

CREATE INDEX idx_fact_lab_control_post_hadm_item_time 
    ON analysis.fact_lab_control_post(hadm_id, lab_itemid, lab_charttime);

CREATE INDEX idx_fact_lab_control_post_valuenum 
    ON analysis.fact_lab_control_post(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_post;

-- =============================================================================
-- LAB DATA 3: PRE-COVID CONTROL GROUP
-- =============================================================================

-- Drop existing pre-COVID control lab events table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_pre CASCADE;

-- Create lab events table for pre-COVID control admissions (2017-2019)
CREATE TABLE analysis.fact_lab_control_pre AS
SELECT
    -- =================================================================
    -- CORE IDENTIFIERS
    -- =================================================================
    lab.hadm_id,                         -- Hospital admission ID
    
    -- =================================================================
    -- LAB EVENT DETAILS (prefixed with "lab_" for clarity)
    -- =================================================================
    lab.charttime AS lab_charttime,     -- Timestamp of lab measurement
    lab.itemid AS lab_itemid,           -- Lab test identifier (links to ref_lab_item)
    lab.value AS lab_value,             -- Text representation of result
    lab.valuenum AS lab_valuenum,       -- Numeric value (NULL if non-numeric)
    lab.valueuom AS lab_unit,           -- Unit of measurement (e.g., mg/dL, mmol/L)
    
    -- =================================================================
    -- REFERENCE RANGES & FLAGS
    -- =================================================================
    lab.ref_range_lower AS lab_ref_range_lower,  -- Lower bound of normal range
    lab.ref_range_upper AS lab_ref_range_upper,  -- Upper bound of normal range
    lab.flag AS lab_flag                         -- Abnormal flag (e.g., 'abnormal', 'delta')

FROM staging.fact_lab_filtered AS lab
    -- Source: Pre-filtered lab events (2017-2022)

-- =================================================================
-- FILTER 1: Admission-level filtering (pre-COVID control patients)
-- =================================================================
INNER JOIN analysis.demo_patient_control_pre demo
    ON lab.hadm_id = demo.hadm_id
    -- Reference: Pre-COVID control admissions (2017-2019)

-- =================================================================
-- FILTER 2: Lab item filtering (same top 61 items)
-- =================================================================
INNER JOIN analysis.ref_lab_item ref
    ON lab.itemid = ref.itemid;
    -- Reference: Same 61 lab tests used for COVID cohort

-- Result: All lab measurements for pre-COVID controls, limited to top 61 tests
-- Grain: One row per lab measurement

-- Create strategic indexes
CREATE INDEX idx_fact_lab_control_pre_hadm_id 
    ON analysis.fact_lab_control_pre(hadm_id);

CREATE INDEX idx_fact_lab_control_pre_itemid 
    ON analysis.fact_lab_control_pre(lab_itemid);

CREATE INDEX idx_fact_lab_control_pre_hadm_item_time 
    ON analysis.fact_lab_control_pre(hadm_id, lab_itemid, lab_charttime);

CREATE INDEX idx_fact_lab_control_pre_valuenum 
    ON analysis.fact_lab_control_pre(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_pre;