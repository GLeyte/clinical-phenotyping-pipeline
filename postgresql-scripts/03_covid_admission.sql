-- =============================================================================
-- 03_covid_admissions.sql
-- Identifies COVID-19 admissions (ICD code U071) and creates patient cohort
-- =============================================================================

-- =============================================================================
-- STEP 1: IDENTIFY COVID-19 ADMISSIONS REFERENCE TABLE
-- =============================================================================

-- Create table for COVID-19 admissions
DROP TABLE IF EXISTS analysis.ref_admission_covid CASCADE;
CREATE TABLE analysis.ref_admission_covid AS
SELECT DISTINCT
    hadm_id                          -- Hospital admission ID with COVID diagnosis
FROM mimiciv_hosp.diagnoses_icd diag
    -- Base table: ICD diagnosis codes
    -- Note: One admission can have multiple diagnosis codes
WHERE diag.icd_code ILIKE '%U071%';
    -- Filter: COVID-19 diagnosis code
    -- U07.1 = COVID-19, virus identified (ICD-10-CM)
    -- ILIKE = case-insensitive pattern matching
    -- % wildcards capture variations (U071, U07.1, etc.)

-- DISTINCT ON ensures one row per admission (deduplicates if multiple U071 codes)

-- Add primary key and index
ALTER TABLE analysis.ref_admission_covid 
    ADD CONSTRAINT pk_ref_admission_covid PRIMARY KEY (hadm_id);

ANALYZE analysis.ref_admission_covid;

-- =============================================================================
-- STEP 2: CREATE COVID PATIENTS REFERENCE TABLE (Using ROW_NUMBER)
-- =============================================================================

-- Drop existing patient reference table if present
DROP TABLE IF EXISTS analysis.ref_patient_covid CASCADE;

-- Create patient-level cohort with their FIRST COVID admission
CREATE TABLE analysis.ref_patient_covid AS
WITH ranked_covid_admissions AS (
    -- Assign row numbers to each patient's COVID admissions, ordered by age
    SELECT 
        demo.subject_id,                     -- Unique patient identifier
        demo.hadm_id,                        -- Hospital admission ID
        demo.age,                            -- Age at admission
        ROW_NUMBER() OVER (
            PARTITION BY demo.subject_id     -- Separate ranking per patient
            ORDER BY demo.age ASC            -- Rank by age (youngest first)
        ) AS rn                              -- Row number: 1 = first admission
    FROM staging.demo_admission_filtered AS demo
        -- Source: Filtered demographics (2017-2022 admissions)
    
    INNER JOIN analysis.ref_admission_covid AS adm
        ON demo.hadm_id = adm.hadm_id
        -- Inner join: Only keeps admissions that are COVID-positive
        -- Links demographics to identified COVID admissions
)
-- Select only the first COVID admission per patient
SELECT 
    subject_id,
    hadm_id,
    age
FROM ranked_covid_admissions
WHERE rn = 1;
    -- Filter: Keep only rows where row_number = 1
    -- Result: One row per COVID patient with their first admission details

-- Add primary key and index
ALTER TABLE analysis.ref_patient_covid 
    ADD CONSTRAINT pk_ref_patient_covid PRIMARY KEY (subject_id);

CREATE INDEX idx_ref_patient_covid_hadm_id 
    ON analysis.ref_patient_covid(hadm_id);

CREATE INDEX idx_ref_patient_covid_age 
    ON analysis.ref_patient_covid(age);

ANALYZE analysis.ref_patient_covid;

-- =============================================================================
-- STEP 4: Creates demographics table for COVID patients (first admission only)
-- =============================================================================

-- Drop existing COVID demographics table if present
DROP TABLE IF EXISTS analysis.demo_patient_covid CASCADE;

-- Create comprehensive demographics table for COVID patients
CREATE TABLE analysis.demo_patient_covid AS
SELECT
    demo.*                                    -- All demographic and comorbidity columns
FROM staging.demo_admission_filtered demo
    -- Source: Filtered demographics with age, gender, race, admission type,
    --         length of stay, comorbidities, and Charlson index (28 columns)

-- Subquery: Filter to first COVID admission per patient
INNER JOIN analysis.ref_patient_covid ref
    ON demo.hadm_id = ref.hadm_id;
    -- Result: Complete demographics for COVID patients' first admissions
    -- Grain: One row per COVID patient (deduplicated at patient level)

-- Add primary key
ALTER TABLE analysis.demo_patient_covid 
    ADD CONSTRAINT pk_demo_patient_covid PRIMARY KEY (hadm_id);

-- Add indexes for common queries
CREATE INDEX idx_demo_patient_covid_subject_id 
    ON analysis.demo_patient_covid(subject_id);

CREATE INDEX idx_demo_patient_covid_age 
    ON analysis.demo_patient_covid(age);

CREATE INDEX idx_demo_patient_covid_start_year 
    ON analysis.demo_patient_covid(start_year);

CREATE INDEX idx_demo_patient_covid_end_year
    ON analysis.demo_patient_covid(end_year);

ANALYZE analysis.demo_patient_covid;