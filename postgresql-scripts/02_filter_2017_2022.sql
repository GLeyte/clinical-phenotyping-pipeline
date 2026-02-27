-- =============================================================================
-- 02_filter_2017_2022.sql
-- Filters demographics and lab events to include only data after 2016
-- =============================================================================

-- =============================================================================
-- STEP 1: FILTER DEMOGRAPHICS TABLE
-- =============================================================================

-- Drop existing filtered demographics table if present
DROP TABLE IF EXISTS staging.demo_admission_filtered;

-- Create filtered demographics table with only post-2016 admissions
CREATE TABLE staging.demo_admission_filtered AS
SELECT * 
FROM staging.demo_admission_base
WHERE start_year > 2016;
    -- anchor_year_group: De-identified 3-year ranges that approximate admission dates
    -- Includes: 2017-2022+ (6 years of data)
    -- Excludes: year <= 2016 

-- Add primary key
ALTER TABLE staging.demo_admission_filtered 
    ADD CONSTRAINT pk_demo_admission_filtered PRIMARY KEY (hadm_id);

-- Create indexes for downstream queries
CREATE INDEX idx_demo_filtered_subject_id 
    ON staging.demo_admission_filtered(subject_id);

CREATE INDEX idx_demo_filtered_start_year 
    ON staging.demo_admission_filtered(start_year);

CREATE INDEX idx_demo_filtered_end_year
    ON staging.demo_admission_filtered(end_year);

CREATE INDEX idx_demo_filtered_age 
    ON staging.demo_admission_filtered(age);

-- Composite index for patient + age queries (used in future admission scripts)
CREATE INDEX idx_demo_filtered_subject_age 
    ON staging.demo_admission_filtered(subject_id, age);

-- Update statistics
ANALYZE staging.demo_admission_filtered;

-- =============================================================================
-- STEP 2: FILTER LAB EVENTS TABLE
-- =============================================================================

-- Drop existing filtered lab events table if present
DROP TABLE IF EXISTS staging.fact_lab_filtered;

-- Create filtered lab events table matching the filtered admissions
CREATE TABLE staging.fact_lab_filtered AS
SELECT lab.* 
FROM mimiciv_hosp.labevents lab

-- Subquery: Get all admission IDs from filtered demographics
WHERE EXISTS (
    SELECT 1 
    FROM staging.demo_admission_filtered d
    WHERE d.hadm_id = lab.hadm_id
);
    -- Only includes lab events for admissions in the 2017-2022 timeframe
    -- Maintains referential integrity between demographics and lab data

-- Composite index on (hadm_id, itemid) - most common query pattern
CREATE INDEX idx_fact_lab_filtered_hadm_item 
    ON staging.fact_lab_filtered(hadm_id, itemid);

-- Index on itemid alone (for lab item filtering)
CREATE INDEX idx_fact_lab_filtered_itemid 
    ON staging.fact_lab_filtered(itemid);

-- Composite index for temporal queries
CREATE INDEX idx_fact_lab_filtered_hadm_charttime 
    ON staging.fact_lab_filtered(hadm_id, charttime);

-- Composite index for first-value queries (used in script 07)
CREATE INDEX idx_fact_lab_filtered_hadm_item_time 
    ON staging.fact_lab_filtered(hadm_id, itemid, charttime);

-- Update statistics
ANALYZE staging.fact_lab_filtered;