-- =============================================================================
-- 04_select_lab_items.sql
-- Identifies and selects top 61 most common lab items in COVID patients
-- =============================================================================

-- =============================================================================
-- DROP TABLE WITH CASCADE
-- =============================================================================

DROP TABLE IF EXISTS analysis.ref_lab_item CASCADE;
    -- CASCADE: Automatically drops dependent objects (views, foreign keys, etc.)
    -- Safer than manual cleanup when rebuilding reference tables
    -- Example dependents: Views that reference ref_lab_item

-- =============================================================================
-- SINGLE OPTIMIZED QUERY (No Temporary Tables)
-- =============================================================================

CREATE TABLE analysis.ref_lab_item AS
WITH distinct_items AS (
    -- =================================================================
    -- STEP 1: Get unique (admission, lab item) pairs efficiently
    -- =================================================================
    -- Deduplicates multiple measurements of same test in same admission
    -- Counts "how many admissions had this test" not "total measurements"
    
    SELECT DISTINCT hadm_id, itemid
    FROM staging.fact_lab_filtered
        -- Source: Pre-filtered lab events (2017-2022)
    
    WHERE EXISTS (
        -- OPTIMIZATION: EXISTS vs. IN
        -- EXISTS stops as soon as a match is found (short-circuit)
        -- More efficient than IN for large datasets
        -- Also handles NULL values better
        
        SELECT 1 
        FROM analysis.demo_patient_covid d
        WHERE d.hadm_id = staging.fact_lab_filtered.hadm_id
    )
    -- Filter: Only COVID patient admissions
    -- Result: One row per unique (admission, lab test) pair
),
item_counts AS (
    -- =================================================================
    -- STEP 2: Count frequency of each lab item
    -- =================================================================
    -- Aggregates across all COVID admissions
    
    SELECT 
        itemid,                          -- Lab test identifier
        COUNT(*) AS item_count           -- Number of admissions with this test
    FROM distinct_items
    GROUP BY itemid
    -- Result: Each lab item with its frequency count
)
-- =================================================================
-- STEP 3: Enrich with metadata and select top 61
-- =================================================================
SELECT 
    -- Frequency metric
    ic.item_count,                       -- Number of admissions with this test
    
    -- Lab item identifiers
    lab.itemid,                          -- Unique lab test ID (e.g., 50912)
    lab.label,                           -- Human-readable name (e.g., "Creatinine")
    
    -- ENHANCED METADATA (new additions)
    lab.fluid,                           -- Specimen type (Blood, Urine, etc.)
    lab.category,                        -- Test category (Chemistry, Hematology, etc.)

	-- Got from mapping 'd_labitems_to_loinc.csv'
	lab.valueuom,
	lab.omop_concept_id,
	lab.omop_concept_name,
	lab.omop_domain_id,
	lab.omop_concept_class_id

FROM mimiciv_updated.d_labitems lab
    -- Source: Lab item dictionary with complete metadata
	-- Has mapping 'd_labitems_to_loinc.csv'

INNER JOIN item_counts ic
    ON lab.itemid = ic.itemid
    -- Join: Link frequency counts to lab metadata
    -- INNER ensures only items that exist in both tables

ORDER BY ic.item_count DESC              -- Sort by frequency (most common first)
LIMIT 61;                                -- Select top 61 most common tests

-- Result: Reference table with 61 most frequently ordered labs + metadata
-- Grain: One row per selected lab item
-- Columns: 6 (item_count, itemid, label, fluid, category, loinc_code)

-- =============================================================================
-- OPTIMIZATION 1: PRIMARY KEY CONSTRAINT
-- =============================================================================

-- Add primary key on itemid for data integrity and index benefits
ALTER TABLE analysis.ref_lab_item 
    ADD CONSTRAINT pk_ref_lab_item PRIMARY KEY (itemid);
    -- Benefits:
    -- 1. Enforces uniqueness (no duplicate itemid values)
    -- 2. Automatically creates unique B-tree index
    -- 3. Enables foreign key references from other tables
    -- 4. Improves JOIN performance when joining on itemid

-- =============================================================================
-- OPTIMIZATION 2: DESCENDING INDEX ON ITEM_COUNT
-- =============================================================================

-- Index on item_count for sorting and filtering operations
CREATE INDEX idx_ref_lab_item_count 
    ON analysis.ref_lab_item(item_count DESC);
    -- DESC: Optimizes descending sorts (ORDER BY item_count DESC)
    -- Use cases:
    -- - SELECT * FROM ref_lab_item ORDER BY item_count DESC LIMIT 10
    -- - WHERE item_count > 5000
    -- Performance: O(log n) instead of O(n) for sorted retrieval

-- =============================================================================
-- OPTIMIZATION 3: INDEX ON LABEL FOR TEXT SEARCHES
-- =============================================================================

-- Index on label for name-based lookups
CREATE INDEX idx_ref_lab_item_label 
    ON analysis.ref_lab_item(label);
    -- Use cases:
    -- - WHERE label = 'Creatinine'
    -- - WHERE label LIKE 'Glucose%'
    -- - JOIN other_table ON ref.label = other.test_name
    -- Performance: Enables index scans for text searches

-- =============================================================================
-- OPTIMIZATION 4: UPDATE TABLE STATISTICS
-- =============================================================================

ANALYZE analysis.ref_lab_item;
    -- Updates PostgreSQL query planner statistics
    -- Collects info on:
    -- - Row count
    -- - Column data distributions
    -- - NULL value frequencies
    -- - Most common values
    -- Benefits:
    -- - Better query plans
    -- - More accurate JOIN cost estimates
    -- - Optimal index selection
    -- Should run after bulk loads/updates