-- =============================================================================
-- 05_control_groups.sql
-- Creates control groups: post-COVID (2020-2022, no COVID) and pre-COVID (2017-2019)
-- =============================================================================

-- =============================================================================
-- CONTROL GROUP 1: POST-COVID ERA (2020-2022, NO COVID DIAGNOSIS)
-- =============================================================================

-- Drop existing post-COVID control table if present
DROP TABLE IF EXISTS analysis.demo_patient_control_post CASCADE;

-- Create post-COVID control group (pandemic era, non-COVID patients)
CREATE TABLE analysis.demo_patient_control_post AS
WITH eligible_patients AS (
    SELECT
        -- =================================================================
        -- ALL DEMOGRAPHIC COLUMNS (28 total)
        -- =================================================================
        -- Identifiers: subject_id, hadm_id
        -- Demographics: age, gender, race, anchor_year_group
        -- Admission: admission_type, length_of_stay_days
        -- Comorbidities: 17 binary flags + charlson_comorbidity_index
        -- =================================================================
        demo.*,
        -- Add hash for deterministic sampling (reproducible results)
        hashtext(demo.subject_id::text) AS subject_hash,
        ROW_NUMBER() OVER (
            PARTITION BY demo.subject_id     -- Separate ranking per patient
            ORDER BY demo.age ASC            -- Rank by age (youngest first)
        ) AS rn
    FROM staging.demo_admission_filtered AS demo
        -- Source: Filtered demographics (2017-2022 admissions)

    WHERE 
        -- Exclusion criteria: Remove all patients with COVID-positive admissions
        NOT EXISTS (
                SELECT 1 
                FROM analysis.ref_patient_covid cov
                -- Reference table: All patients with admissions with COVID diagnosis
                WHERE cov.subject_id = demo.subject_id
            )
        -- Temporal filter: Pandemic era (2020-2022)
        AND demo.start_year >= 2020
)
SELECT 
    subject_id, hadm_id, age, start_year, end_year, gender, race,
    admission_type, length_of_stay_days, died_in_stay, died, time_died_after,
    myocardial_infarct, congestive_heart_failure, peripheral_vascular_disease,
    cerebrovascular_disease, dementia, chronic_pulmonary_disease,
    rheumatic_disease, peptic_ulcer_disease, mild_liver_disease,
    diabetes_without_cc, diabetes_with_cc, paraplegia, renal_disease,
    malignant_cancer, severe_liver_disease, metastatic_solid_tumor,
    aids, charlson_comorbidity_index
FROM eligible_patients
WHERE rn = 1
ORDER BY subject_hash  -- Deterministic ordering based on hash
LIMIT 20000;
    -- Sample size cap: Limit to 20,000 patients
    -- Purpose: (1) Computational efficiency, (2) Balanced comparison
    -- Note: First 20,000 patients after sorting by subject_id

-- Result: 20,000 unique patients from 2020-2022 without COVID diagnosis
-- Grain: One row per patient (their first non-COVID admission in this period)

-- Add constraints and indexes
ALTER TABLE analysis.demo_patient_control_post 
    ADD CONSTRAINT pk_demo_patient_control_post PRIMARY KEY (subject_id);

CREATE UNIQUE INDEX idx_demo_control_post_hadm_id 
    ON analysis.demo_patient_control_post(hadm_id);

CREATE INDEX idx_demo_control_post_age 
    ON analysis.demo_patient_control_post(age);

CREATE INDEX idx_demo_control_post_start_year 
    ON analysis.demo_patient_control_post(start_year);

CREATE INDEX idx_demo_control_post_end_year
    ON analysis.demo_patient_control_post(end_year);

-- Composite index for common filter patterns
CREATE INDEX idx_demo_control_post_age_charlson 
    ON analysis.demo_patient_control_post(age, charlson_comorbidity_index);

ANALYZE analysis.demo_patient_control_post;

-- =============================================================================
-- CONTROL GROUP 2: PRE-COVID ERA (2017-2019, NO COVID DIAGNOSIS)
-- =============================================================================

-- Drop existing pre-COVID control table if present
DROP TABLE IF EXISTS analysis.demo_patient_control_pre;

-- Create pre-COVID control group (baseline, pre-pandemic patients)
CREATE TABLE analysis.demo_patient_control_pre AS
WITH eligible_patients AS (
    SELECT
        -- =================================================================
        -- ALL DEMOGRAPHIC COLUMNS (28 total)
        -- =================================================================
        demo.*,
        -- Add hash for deterministic sampling (reproducible results)
        hashtext(demo.subject_id::text) AS subject_hash,
        ROW_NUMBER() OVER (
            PARTITION BY demo.subject_id     -- Separate ranking per patient
            ORDER BY demo.age ASC            -- Rank by age (youngest first)
        ) AS rn
    FROM staging.demo_admission_filtered AS demo
        -- Source: Filtered demographics (2017-2022 admissions)

    WHERE 
        -- Temporal filter: Pre-pandemic era (2017-2019)
        demo.end_year <= 2019
        -- Exclusion criteria: Remove all patients with COVID-positive admissions
        AND NOT EXISTS (
                SELECT 1 
                FROM analysis.ref_patient_covid cov
                -- Reference table: All patients with admissions with COVID diagnosis
                WHERE cov.subject_id = demo.subject_id
                -- Note: Should be empty for 2017-2019 (COVID emerged late 2019)
            )
)
SELECT 
    subject_id, hadm_id, age, start_year, end_year, gender, race,
    admission_type, length_of_stay_days, died_in_stay, died, time_died_after,
    myocardial_infarct, congestive_heart_failure, peripheral_vascular_disease,
    cerebrovascular_disease, dementia, chronic_pulmonary_disease,
    rheumatic_disease, peptic_ulcer_disease, mild_liver_disease,
    diabetes_without_cc, diabetes_with_cc, paraplegia, renal_disease,
    malignant_cancer, severe_liver_disease, metastatic_solid_tumor,
    aids, charlson_comorbidity_index
FROM eligible_patients
WHERE rn = 1
ORDER BY subject_hash  -- Deterministic ordering based on hash
LIMIT 20000;
    -- Sample size cap: Limit to 20,000 patients
    -- Purpose: Match sample size with post-COVID control for balance

-- Result: 20,000 unique patients from 2017-2019 without COVID diagnosis
-- Grain: One row per patient (their first admission in this period)

-- Add constraints and indexes
ALTER TABLE analysis.demo_patient_control_pre 
    ADD CONSTRAINT pk_demo_patient_control_pre PRIMARY KEY (subject_id);

CREATE UNIQUE INDEX idx_demo_control_pre_hadm_id 
    ON analysis.demo_patient_control_pre(hadm_id);

CREATE INDEX idx_demo_control_pre_age 
    ON analysis.demo_patient_control_pre(age);

CREATE INDEX idx_demo_control_pre_start_year
    ON analysis.demo_patient_control_pre(start_year);

CREATE INDEX idx_demo_control_pre_end_year 
    ON analysis.demo_patient_control_pre(end_year);

-- Composite index for common filter patterns
CREATE INDEX idx_demo_control_pre_age_charlson 
    ON analysis.demo_patient_control_pre(age, charlson_comorbidity_index);

ANALYZE analysis.demo_patient_control_pre;