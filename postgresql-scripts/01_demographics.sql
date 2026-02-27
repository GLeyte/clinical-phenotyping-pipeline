-- =============================================================================
-- 01_demographics.sql
-- Creates base demographics table with age, comorbidities, and admission info
-- =============================================================================

-- Drop existing table if present to ensure clean rebuild
DROP TABLE IF EXISTS staging.demo_admission_base;

-- Create demographics dimension table by joining multiple MIMIC-IV sources
CREATE TABLE staging.demo_admission_base AS

SELECT
    -- =========================================================================
    -- PRIMARY IDENTIFIERS
    -- =========================================================================
    d_age.subject_id,                    -- Unique patient identifier
    d_age.hadm_id,                       -- Hospital admission identifier
    
    -- =========================================================================
    -- PATIENT DEMOGRAPHICS
    -- =========================================================================
    d_age.age,                           -- Patient age at admission (years)

	-- De-identified year group (3-year range)
	(CAST(SPLIT_PART(pat.anchor_year_group, '-', 1) AS INTEGER) + FLOOR(d_age.age) - pat.anchor_age) as start_year,
	(CAST(SPLIT_PART(pat.anchor_year_group, '-', 2) AS INTEGER) + FLOOR(d_age.age) - pat.anchor_age) as end_year,
	--
    pat.gender,                          -- Patient gender (M/F)
    adm.race,                            -- Patient race/ethnicity
    
    -- =========================================================================
    -- ADMISSION CHARACTERISTICS
    -- =========================================================================
    adm.admission_type,                  -- Type: ELECTIVE, EMERGENCY, URGENT, etc.
    
    -- Calculate length of stay in days (rounded to 2 decimals)
    -- Formula: (discharge_time - admission_time) converted to days
    ROUND(EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime)) / 86400, 2) AS length_of_stay_days,
    
    -- Flag for death of patient in stay
    adm.hospital_expire_flag AS died_in_stay,

    -- Flag for death of patient up to one year post-hospital discharge
    CASE
        WHEN pat.dod IS NOT NULL THEN 1
        ELSE 0
    END AS died,
    
    -- Interval of time from admission until death
    CASE
        WHEN pat.dod IS NOT NULL AND pat.dod - adm.admittime > INTERVAL '1 day' THEN pat.dod - adm.admittime
        WHEN pat.dod IS NOT NULL AND pat.dod - adm.admittime <= INTERVAL '1 day' THEN adm.deathtime - adm.admittime
        ELSE NULL
    END AS time_died_after,
    -- =========================================================================
    -- COMORBIDITY FLAGS (Binary indicators)
    -- =========================================================================
    comor.myocardial_infarct,            -- History of heart attack
    comor.congestive_heart_failure,      -- CHF diagnosis
    comor.peripheral_vascular_disease,   -- PVD diagnosis
    comor.cerebrovascular_disease,       -- Stroke/TIA history
    comor.dementia,                      -- Dementia diagnosis
    comor.chronic_pulmonary_disease,     -- COPD/chronic lung disease
    comor.rheumatic_disease,             -- Rheumatologic disorders
    comor.peptic_ulcer_disease,          -- Peptic ulcer history
    comor.mild_liver_disease,            -- Mild liver disease
    comor.diabetes_without_cc,           -- Diabetes without complications
    comor.diabetes_with_cc,              -- Diabetes with complications
    comor.paraplegia,                    -- Paraplegia/hemiplegia
    comor.renal_disease,                 -- Moderate/severe renal disease
    comor.malignant_cancer,              -- Cancer diagnosis (non-metastatic)
    comor.severe_liver_disease,          -- Severe liver disease
    comor.metastatic_solid_tumor,        -- Metastatic cancer
    comor.aids,                          -- AIDS diagnosis
    
    -- =========================================================================
    -- COMORBIDITY SCORE
    -- =========================================================================
    comor.charlson_comorbidity_index     -- Weighted comorbidity score (0-24+)

-- =============================================================================
-- TABLE JOINS
-- =============================================================================
FROM mimiciv_derived.age AS d_age
    -- Base table: Provides age calculations for each admission

INNER JOIN mimiciv_hosp.patients AS pat
    ON d_age.subject_id = pat.subject_id
    -- Links to patient master table for gender and anchor year

INNER JOIN mimiciv_hosp.admissions AS adm
    ON d_age.hadm_id = adm.hadm_id
    -- Links to admission details for race, type, and timestamps

INNER JOIN mimiciv_derived.charlson AS comor
    ON d_age.hadm_id = comor.hadm_id;
    -- Links to Charlson comorbidity data for medical history

-- =========================================================================
-- OPTIMIZATION 2: Create Indexes on Frequently Queried Columns
-- =========================================================================

-- Index on subject_id (used for patient-level joins)
CREATE INDEX idx_demo_admission_base_subject_id 
    ON staging.demo_admission_base(subject_id);

-- Index on start_year (used for temporal filtering)
CREATE INDEX idx_demo_admission_base_start_year
    ON staging.demo_admission_base(start_year);

-- Index on end_year (used for temporal filtering)
CREATE INDEX idx_demo_admission_base_end_year
    ON staging.demo_admission_base(end_year);

-- Composite index for common filter combinations
CREATE INDEX idx_demo_admission_base_start_year_age 
    ON staging.demo_admission_base(start_year, age);

-- Composite index for common filter combinations
CREATE INDEX idx_demo_admission_base_end_year_age 
    ON staging.demo_admission_base(end_year, age);

-- Index on age (frequently used in WHERE clauses and ORDER BY)
CREATE INDEX idx_demo_admission_base_age 
    ON staging.demo_admission_base(age);

-- =========================================================================
-- OPTIMIZATION 3: Update Table Statistics
-- =========================================================================
ANALYZE staging.demo_admission_base;