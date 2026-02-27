-- =============================================================================
-- 11_future_pre_data.sql
-- Creates demographics and lab data for future admissions of pre-COVID patients
-- =============================================================================

-- =============================================================================
-- DEMOGRAPHICS FOR PRE-COVID CONTROL FUTURE ADMISSIONS
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.demo_admission_control_future CASCADE;

-- Create demographics table for pre-COVID control patient readmissions
CREATE TABLE analysis.demo_admission_control_future AS
SELECT
    -- =================================================================
    -- ALL DEMOGRAPHIC COLUMNS (28 from demo_admission_filtered)
    -- =================================================================
    demo.*,
        -- Includes: subject_id, hadm_id, age, gender, race, 
        --          admission_type, length_of_stay_days,
        --          comorbidities (17 flags), charlson_comorbidity_index
    
    -- =================================================================
    -- TEMPORAL TRACKING COLUMN
    -- =================================================================
    future.age_gap AS days_gap
        -- Time elapsed since index admission (in days)
        -- Renamed from age_gap for clarity
        -- Example: 120 = readmitted 120 days after index admission

FROM staging.demo_admission_filtered AS demo
    -- Source: Complete demographics for all 2017-2022 admissions
    
JOIN analysis.bridge_admission_control_future future
    ON future.hadm_id = demo.hadm_id;
    -- Inner join: Only keeps COVID-free readmissions from bridge table
    -- Bridge contains: Subsequent admissions for pre-COVID controls

-- Output: Complete patient profile for each COVID-free readmission
-- Grain: One row per readmission (excludes index admission)
-- Columns: 29 (28 from demographics + days_gap)
-- Note: No 'covid' flag needed - all admissions are COVID-free by design

-- Create indexes for common queries
CREATE INDEX idx_demo_control_future_subject_id 
    ON analysis.demo_admission_control_future(subject_id);

CREATE INDEX idx_demo_control_future_days_gap 
    ON analysis.demo_admission_control_future(days_gap);

-- Composite index for stratified temporal analysis
CREATE INDEX idx_demo_control_future_subject_gap 
    ON analysis.demo_admission_control_future(subject_id, days_gap);

-- Index for demographic filtering
CREATE INDEX idx_demo_control_future_age 
    ON analysis.demo_admission_control_future(age);

ANALYZE analysis.demo_admission_control_future;

-- =============================================================================
-- LAB DATA FOR PRE-COVID CONTROL FUTURE ADMISSIONS
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_future CASCADE;

-- Create lab events table for pre-COVID control patient readmissions
CREATE TABLE analysis.fact_lab_control_future AS
SELECT
    -- =================================================================
    -- STANDARD LAB EVENT COLUMNS (9 total)
    -- =================================================================
    lab.hadm_id,                         -- Admission identifier (readmission)
    lab.charttime AS lab_charttime,     -- Lab measurement timestamp
    lab.itemid AS lab_itemid,           -- Lab test identifier
    lab.value AS lab_value,             -- Text result
    lab.valuenum AS lab_valuenum,       -- Numeric result
    lab.valueuom AS lab_unit,           -- Unit of measurement
    lab.ref_range_lower AS lab_ref_range_lower,  -- Normal range lower bound
    lab.ref_range_upper AS lab_ref_range_upper,  -- Normal range upper bound
    lab.flag AS lab_flag                -- Abnormality flag

FROM staging.fact_lab_filtered AS lab
    -- Source: Pre-filtered lab events (2017-2022)
    -- CORRECTED: Was incorrectly referencing mimiciv_updated.filtered_lab_after_2016

-- =================================================================
-- FILTER 1: Admission-level filtering (control readmissions only)
-- =================================================================
INNER JOIN analysis.demo_admission_control_future demo
    ON lab.hadm_id = demo.hadm_id
    -- Reference: COVID-free readmission IDs from demographics table

-- =================================================================
-- FILTER 2: Lab item filtering (top 61 items only)
-- =================================================================
INNER JOIN analysis.ref_lab_item ref
    ON lab.itemid = ref.itemid;
    -- Reference: Same 61 most common lab tests

-- Output: All lab measurements during COVID-free readmissions
-- Grain: One row per lab measurement
-- Purpose: Enable comparison of lab trajectories in control group

-- Create strategic indexes
CREATE INDEX idx_fact_lab_control_future_hadm_id 
    ON analysis.fact_lab_control_future(hadm_id);

CREATE INDEX idx_fact_lab_control_future_itemid 
    ON analysis.fact_lab_control_future(lab_itemid);

-- Critical composite index for first-value extraction
CREATE INDEX idx_fact_lab_control_future_hadm_item_time 
    ON analysis.fact_lab_control_future(hadm_id, lab_itemid, lab_charttime);

-- Partial index for numeric values
CREATE INDEX idx_fact_lab_control_future_valuenum 
    ON analysis.fact_lab_control_future(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_future;

-- =============================================================================
-- FIRST LAB VALUES FOR PRE-COVID CONTROL FUTURE ADMISSIONS
-- =============================================================================

-- Drop existing table if present
DROP TABLE IF EXISTS analysis.fact_lab_control_future_first CASCADE;

-- Create table with first lab measurement per readmission-lab pair
CREATE TABLE analysis.fact_lab_control_future_first AS
WITH ranked_rows AS (
    -- =================================================================
    -- STEP 1: Rank lab measurements within each readmission
    -- =================================================================
    SELECT
        *,                               -- All columns from fact_lab_control_future
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id, lab_itemid
                -- Group by: Each unique (readmission, lab test) combination
            
            ORDER BY lab_charttime ASC
                -- Sort by: Earliest timestamp first
                -- rn = 1 will be the first measurement at readmission
        ) AS rn
    
    FROM analysis.fact_lab_control_future
        -- Source: All lab measurements during control group readmissions
)
-- =================================================================
-- STEP 2: Filter to keep only first measurement
-- =================================================================
SELECT *
FROM ranked_rows
WHERE rn = 1;
    -- Filter: Keep only earliest measurement per (readmission, lab) pair
    -- Result: Baseline labs at each readmission

-- Output: First lab value of each test at each readmission
-- Grain: One row per (readmission, lab test) combination
-- Purpose: Enable comparison of baseline labs across admissions

-- Create indexes
CREATE INDEX idx_fact_lab_control_future_first_hadm_id 
    ON analysis.fact_lab_control_future_first(hadm_id);

CREATE INDEX idx_fact_lab_control_future_first_itemid 
    ON analysis.fact_lab_control_future_first(lab_itemid);

-- Composite index for wide-format pivoting
CREATE INDEX idx_fact_lab_control_future_first_hadm_item 
    ON analysis.fact_lab_control_future_first(hadm_id, lab_itemid);

-- Partial index for numeric analysis
CREATE INDEX idx_fact_lab_control_future_first_valuenum 
    ON analysis.fact_lab_control_future_first(lab_valuenum) 
    WHERE lab_valuenum IS NOT NULL;

ANALYZE analysis.fact_lab_control_future_first;