-- =============================================================================
-- 12_download.sql
-- Exports analysis tables to CSV files for external use
-- =============================================================================

-- =============================================================================
-- SECTION 1: BASELINE DEMOGRAPHICS (3 COHORTS)
-- =============================================================================

-- -------------------------------------------------------------------------
-- Export 1: COVID Patient Demographics
-- -------------------------------------------------------------------------
COPY analysis.demo_patient_covid 
TO '/home/impexp/output/demo_patient_covid.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.demo_patient_covid
-- Cohort: COVID-19 patients (first COVID admission only)
-- Time Period: 2020 >
-- Grain: One row per unique COVID patient
-- Columns: 28 (demographics + comorbidities + Charlson index)
-- Use Case: Baseline characteristics of COVID patients

-- -------------------------------------------------------------------------
-- Export 2: Post-COVID Control Demographics
-- -------------------------------------------------------------------------
COPY analysis.demo_patient_control_post
TO '/home/impexp/output/demo_patient_control_post.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.demo_patient_control_post
-- Cohort: Post-COVID control (2020-2022, no COVID diagnosis)
-- Time Period: 2020 > (pandemic era, COVID-free)
-- Grain: One row per unique control patient
-- Columns: 28 (identical structure to COVID demographics)
-- Use Case: Contemporary controls for COVID comparison

-- -------------------------------------------------------------------------
-- Export 3: Pre-COVID Control Demographics
-- -------------------------------------------------------------------------
COPY analysis.demo_patient_control_pre 
TO '/home/impexp/output/demo_patient_control_pre.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.demo_patient_control_pre
-- Cohort: Pre-COVID control (2017-2019, no COVID diagnosis)
-- Time Period: 2017-2019 (pre-pandemic baseline)
-- Grain: One row per unique control patient
-- Columns: 28 (identical structure to COVID demographics)
-- Use Case: Historical baseline for temporal comparisons

-- =============================================================================
-- SECTION 2: READMISSION DEMOGRAPHICS (2 COHORTS)
-- =============================================================================

-- -------------------------------------------------------------------------
-- Export 4: COVID Patient Readmissions
-- -------------------------------------------------------------------------
COPY analysis.demo_admission_covid_future 
TO '/home/impexp/output/demo_admission_covid_future.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.demo_admission_covid_future
-- Cohort: COVID patients' subsequent admissions (readmissions)
-- Time Period: After initial COVID admission
-- Grain: One row per readmission
-- Columns: 30 (28 demographics + days_gap + covid flag)
--   - days_gap: Days since first COVID admission
--   - covid: 1 if readmission has COVID, 0 otherwise
-- Use Case: Longitudinal outcomes after COVID

-- -------------------------------------------------------------------------
-- Export 5: Pre-COVID Control Readmissions
-- -------------------------------------------------------------------------
COPY analysis.demo_admission_control_future 
TO '/home/impexp/output/demo_admission_control_future.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.demo_admission_control_future
-- Cohort: Pre-COVID control patients' COVID-free readmissions
-- Time Period: After index admission
-- Grain: One row per COVID-free readmission
-- Columns: 29 (28 demographics + days_gap)
--   - days_gap: Days since index admission
--   - No covid flag (all admissions COVID-free by design)
-- Typical Rows: Variable (depends on readmission rate)
-- Use Case: Control group longitudinal outcomes

-- =============================================================================
-- SECTION 3: BASELINE LAB DATA (3 COHORTS)
-- =============================================================================

-- -------------------------------------------------------------------------
-- Export 6: COVID Patient First Lab Values
-- -------------------------------------------------------------------------
COPY analysis.fact_lab_covid_first 
TO '/home/impexp/output/fact_lab_covid_first.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.fact_lab_covid_first
-- Cohort: COVID patients' first (baseline) lab values
-- Time Period: At first COVID admission
-- Grain: One row per (patient admission, lab test) combination
-- Columns: 9 (hadm_id, lab_charttime, lab_itemid, lab_value, 
--              lab_valuenum, lab_unit, ref_range_lower, 
--              ref_range_upper, lab_flag)
-- Lab Items: Up to 61 tests per admission (top 61 most common)
-- Use Case: Baseline lab features for modeling

-- -------------------------------------------------------------------------
-- Export 7: Post-COVID Control First Lab Values
-- -------------------------------------------------------------------------
COPY analysis.fact_lab_control_post_first 
TO '/home/impexp/output/fact_lab_control_post_first.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.fact_lab_control_post_first
-- Cohort: Post-COVID controls' first (baseline) lab values
-- Time Period: At first admission (2020-2022)
-- Grain: One row per (patient admission, lab test) combination
-- Columns: 9 (identical structure to COVID labs)
-- Lab Items: Same 61 tests for consistency
-- Use Case: Contemporary control lab features

-- -------------------------------------------------------------------------
-- Export 8: Pre-COVID Control First Lab Values
-- -------------------------------------------------------------------------
COPY analysis.fact_lab_control_pre_first 
TO '/home/impexp/output/fact_lab_control_pre_first.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.fact_lab_control_pre_first
-- Cohort: Pre-COVID controls' first (baseline) lab values
-- Time Period: At first admission (2017-2019)
-- Grain: One row per (patient admission, lab test) combination
-- Columns: 9 (identical structure to COVID labs)
-- Lab Items: Same 61 tests for consistency
-- Use Case: Historical baseline lab features

-- =============================================================================
-- SECTION 4: READMISSION LAB DATA (2 COHORTS)
-- =============================================================================

-- -------------------------------------------------------------------------
-- Export 9: COVID Readmission First Lab Values
-- -------------------------------------------------------------------------
COPY analysis.fact_lab_covid_future_first 
TO '/home/impexp/output/fact_lab_covid_future_first.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.fact_lab_covid_future_first
-- Cohort: COVID patients' first lab values at each readmission
-- Time Period: At each subsequent admission after first COVID
-- Grain: One row per (readmission, lab test) combination
-- Columns: 9 (identical structure to baseline labs)
-- Lab Items: Same 61 tests for consistency
-- Use Case: Longitudinal lab trajectories for COVID patients

-- -------------------------------------------------------------------------
-- Export 10: Pre-COVID Control Readmission First Lab Values
-- -------------------------------------------------------------------------
COPY analysis.fact_lab_control_future_first 
TO '/home/impexp/output/fact_lab_control_future_first.csv' 
WITH (FORMAT CSV, HEADER);

-- Source: analysis.fact_lab_control_future_first
-- Cohort: Pre-COVID controls' first lab values at each readmission
-- Time Period: At each COVID-free subsequent admission
-- Grain: One row per (readmission, lab test) combination
-- Columns: 9 (identical structure to baseline labs)
-- Lab Items: Same 61 tests for consistency
-- Use Case: Control group longitudinal lab trajectories