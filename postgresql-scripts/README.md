# COVID-19 Lab Values Analysis Pipeline

## Overview

This data pipeline extracts and analyzes laboratory measurements from the MIMIC-IV database to study COVID-19 patients and compare them with control groups. The pipeline creates a structured dataset for analyzing lab value patterns, temporal trends, and readmission characteristics.

### Key Features
- **COVID Cohort**: Patients with ICD-10 code U07.1 (COVID-19 diagnosis)
- **Two Control Groups**: 
  - Pre-COVID era (2017-2019)
  - Post-COVID era (2020 >, non-COVID patients)
- **Temporal Analysis**: Tracks subsequent admissions and lab changes over time
- **Top 61 Lab Tests**: Most frequently ordered tests in COVID cohort
- **Optimized Structure**: Indexed tables for efficient querying

---

## Table of Contents

1. [Script Execution Order](#script-execution-order)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Naming Convention](#naming-convention)
4. [Table Catalog](#table-catalog)
5. [Data Lineage](#data-lineage)
6. [Prerequisites](#prerequisites)
7. [Usage Guide](#usage-guide)

---

## Setup

All the scripts assume the MIMIC-IV database is already loaded into a PostgreSQL database.

Two scripts are used to update the MIMIC-IV database, which are not part of the pipeline:
1. age.sql
2. charlson.sql

The source is [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres).

First, you need to create the database and the schemas.

```sql
CREATE DATABASE mimic;

CREATE SCHEMA staging;

CREATE SCHEMA analysis;
```

## Script Execution Order

Execute scripts in numerical order. Each script depends on tables created by previous scripts.

| Script | Purpose | Dependencies | Output Tables |
|--------|---------|--------------|---------------|
| **01_demographics.sql** | Create base demographics with comorbidities | MIMIC-IV source tables | staging.demo_admission_base |
| **02_filter_2017_2022.sql** | Filter data to 2017-2022 timeframe | 01 | staging.demo_admission_filtered<br>staging.fact_lab_filtered |
| **03_covid_admission.sql** | Identify COVID patients and admissions | 02 | analysis.ref_admission_covid<br>analysis.ref_patient_covid<br>analysis.demo_patient_covid |
| **04_select_lab_items.sql** | Select top 61 most common lab tests | 03 | analysis.ref_lab_item |
| **05_control_groups.sql** | Create pre/post-COVID control cohorts | 02, 03 | analysis.demo_patient_control_post<br>analysis.demo_patient_control_pre |
| **06_lab_data.sql** | Extract lab events for all cohorts | 03, 04, 05 | analysis.fact_lab_covid<br>analysis.fact_lab_control_post<br>analysis.fact_lab_control_pre |
| **07_first_lab_values.sql** | Get first lab measurement per admission | 06 | analysis.fact_lab_covid_first<br>analysis.fact_lab_control_post_first<br>analysis.fact_lab_control_pre_first |
| **08_future_covid_admissions.sql** | Track COVID patient readmissions | 03 | analysis.bridge_admission_covid_future |
| **09_future_covid_data.sql** | Demographics and labs for COVID readmissions | 08 | analysis.demo_admission_covid_future<br>analysis.fact_lab_covid_future<br>analysis.fact_lab_covid_future_first |
| **10_future_pre_admissions.sql** | Track pre-COVID control readmissions | 05 | analysis.bridge_admission_control_future |
| **11_future_pre_data.sql** | Demographics and labs for control readmissions | 10 | analysis.demo_admission_control_future<br>analysis.fact_lab_control_future<br>analysis.fact_lab_control_future_first |

### Quick Execution
```bash
# Execute all scripts in order
psql -d mimic -f 01_demographics.sql
psql -d mimic -f 02_filter_2017_2022.sql
psql -d mimic -f 03_covid_admission.sql
psql -d mimic -f 04_select_lab_items.sql
psql -d mimic -f 05_control_groups.sql
psql -d mimic -f 06_lab_data.sql
psql -d mimic -f 07_first_lab_values.sql
psql -d mimic -f 08_future_covid_admissions.sql
psql -d mimic -f 09_future_covid_data.sql
psql -d mimic -f 10_future_pre_admissions.sql
psql -d mimic -f 11_future_pre_data.sql
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MIMIC-IV Source Tables                     │
│  (patients, admissions, diagnoses_icd, labevents, d_labitems)  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Base Demographics                   │
│                  staging.demo_admission_base                    │
│  (All admissions with demographics + Charlson comorbidities)    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 2: Temporal Filtering (2017-2022+)         │
│         staging.demo_admission_filtered (admissions)            │
│         staging.fact_lab_filtered (lab events)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌──────────────────┐        ┌─────────────────────┐
│  COVID Cohort    │        │  Control Groups     │
│  Identification  │        │  Creation           │
└────────┬─────────┘        └──────────┬──────────┘
         │                             │
         │    ┌────────────────────────┘
         │    │             │
         ▼    ▼             ▼
    ┌────────────┐  ┌──────────────┐
    │   COVID    │  │ Post-COVID   │  Pre-COVID
    │  Patients  │  │   Controls   │   Controls
    └─────┬──────┘  └──────┬───────┘  (2017-2019)
          │                │
          │    Lab Item    │
          │   Selection    │
          │  (Top 61)      │
          │                │
          ▼                ▼
    ┌─────────────────────────────┐
    │   Lab Data Extraction       │
    │  (All + First Values)       │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Future Admissions Tracking │
    │  (Readmissions Analysis)    │
    └─────────────────────────────┘
```

---

## Naming Convention

### Pattern Structure

```
[schema].[prefix]_[entity]_[group]_[modifier]

Where:
- schema: staging, analysis
- prefix: demo (demographics), fact (events), ref (reference), bridge (relationships)
- entity: patient, admission, lab
- group: covid, control_post, control_pre, future
- modifier: first, base, filtered
```

### Table Type Definitions

#### Demographics Tables (demo_*)
- **Purpose**: Descriptive attributes about patients and admissions
- **Contains**: Age, gender, race, comorbidities, Charlson index, admission details
- **Characteristics**: Relatively static, one row per patient/admission
- **Examples**: `demo_patient_covid`, `demo_admission_base`

#### Fact Tables (fact_*)
- **Purpose**: Measurable events and observations (time-series data)
- **Contains**: Lab results, measurements with timestamps
- **Characteristics**: Many records per patient, temporal ordering
- **Examples**: `fact_lab_covid`, `fact_lab_control_post_first`

#### Reference Tables (ref_*)
- **Purpose**: Master lists and lookup tables
- **Contains**: Lists of IDs, selected items, metadata
- **Characteristics**: Small, used for filtering and joining
- **Examples**: `ref_admission_covid`, `ref_lab_item`

#### Bridge Tables (bridge_*)
- **Purpose**: Many-to-many relationships and temporal tracking
- **Contains**: Admission linkages, time gaps between events
- **Characteristics**: Connect dimension tables, track sequences
- **Examples**: `bridge_admission_covid_future`

---

## Table Catalog

### 1. STAGING SCHEMA

| Table | Type | Rows | Purpose |
|-------|------|-------------|---------|
| **staging.demo_admission_base** | Demographics | ~546K | Base admission demographics with comorbidities |
| **staging.demo_admission_filtered** | Demographics | ~187K | Filtered admissions (2017-2022+) |
| **staging.fact_lab_filtered** | Fact | ~39M | Filtered lab events (2017-2022+) |

### 2. ANALYSIS SCHEMA

#### 2.1 Reference Tables (Master Lists)

| Table | Type | Rows | Purpose |
|-------|------|-------------|---------|
| **analysis.ref_admission_covid** | Reference | ~4K | All admissions with COVID-19 diagnosis (U07.1) |
| **analysis.ref_patient_covid** | Reference | ~3.6K | Unique COVID patients (first admission only) |
| **analysis.ref_lab_item** | Reference | 61 | Top 61 most common lab tests in COVID cohort |

#### 2.2 COVID Cohort Tables

| Table | Type | Rows | Grain |
|-------|------|-------------|-------|
| **analysis.demo_patient_covid** | Demographics | ~3.6K | One row per COVID patient (first admission) |
| **analysis.fact_lab_covid** | Fact | ~1.5M | All lab events for COVID patients |
| **analysis.fact_lab_covid_first** | Fact | ~156K | First lab value per admission-test pair |

#### 2.3 Control Group - Post-COVID (2020-2022)

| Table | Type | Rows | Grain |
|-------|------|-------------|-------|
| **analysis.demo_patient_control_post** | Demographics | 20k | One row per patient (sampled, no COVID) |
| **analysis.fact_lab_control_post** | Fact | ~3.5M | All lab events for post-COVID controls |
| **analysis.fact_lab_control_post_first** | Fact | ~618K | First lab value per admission-test pair |

#### 2.4 Control Group - Pre-COVID (2017-2019)

| Table | Type | Rows | Grain |
|-------|------|-------------|-------|
| **analysis.demo_patient_control_pre** | Demographics | 20k | One row per patient (sampled, pre-pandemic) |
| **analysis.fact_lab_control_pre** | Fact | ~3.2M | All lab events for pre-COVID controls |
| **analysis.fact_lab_control_pre_first** | Fact | ~618K | First lab value per admission-test pair |

#### 2.5 Future Tracking - COVID Patients (Readmissions)

| Table | Type | Rows | Grain |
|-------|------|-------------|-------|
| **analysis.bridge_admission_covid_future** | Bridge | ~2K | One row per subsequent admission |
| **analysis.demo_admission_covid_future** | Demographics | ~2K | Demographics for COVID readmissions |
| **analysis.fact_lab_covid_future** | Fact | ~532K | All labs during COVID readmissions |
| **analysis.fact_lab_covid_future_first** | Fact | ~80K | First lab per readmission-test pair |

#### 2.6 Future Tracking - Pre-COVID Controls (Readmissions)

| Table | Type | Rows | Grain |
|-------|------|-------------|-------|
| **analysis.bridge_admission_control_future** | Bridge | ~18K | One row per subsequent admission |
| **analysis.demo_admission_control_future** | Demographics | ~18K | Demographics for control readmissions |
| **analysis.fact_lab_control_future** | Fact | ~3.4M | All labs during control readmissions |
| **analysis.fact_lab_control_future_first** | Fact | ~605K | First lab per readmission-test pair |

---

## Data Lineage

### COVID Patient Flow
```
MIMIC-IV diagnoses_icd (U07.1)
  ↓
analysis.ref_admission_covid (all COVID admissions)
  ↓
analysis.ref_patient_covid (first COVID admission per patient)
  ↓
analysis.demo_patient_covid (full demographics)
  ↓
├─→ analysis.fact_lab_covid (all labs)
│     ↓
│   analysis.fact_lab_covid_first (first values)
│
└─→ analysis.bridge_admission_covid_future (readmissions)
      ↓
    analysis.demo_admission_covid_future
      ↓
    analysis.fact_lab_covid_future
      ↓
    analysis.fact_lab_covid_future_first
```

### Control Group Flow
```
staging.demo_admission_filtered
  ↓
[Filter: year range + exclude COVID]
  ↓
├─→ analysis.demo_patient_control_post (2020-2022, no COVID)
│     ↓
│   analysis.fact_lab_control_post
│     ↓
│   analysis.fact_lab_control_post_first
│
└─→ analysis.demo_patient_control_pre (2017-2019, no COVID)
      ↓
    analysis.fact_lab_control_pre
      ↓
    analysis.fact_lab_control_pre_first
      ↓
    analysis.bridge_admission_control_future (readmissions)
      ↓
    analysis.demo_admission_control_future
      ↓
    analysis.fact_lab_control_future
      ↓
    analysis.fact_lab_control_future_first
```

---

## Prerequisites

### Required MIMIC-IV Tables
- `mimiciv_hosp.patients`
- `mimiciv_hosp.admissions`
- `mimiciv_hosp.diagnoses_icd`
- `mimiciv_hosp.labevents`
- `mimiciv_derived.age`
- `mimiciv_derived.charlson`
- `mimiciv_updated.d_labitems` (with LOINC mappings) ([source](https://github.com/MIT-LCP/mimic-code/blob/c34baed99d326d438f7b9a74eea68463925063dd/mimic-iv/mapping/d_labitems_to_loinc.csv))

### Required Schemas
```sql
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analysis;
```

### PostgreSQL Version
- PostgreSQL 12+ (for window functions and CTEs)
- Recommended: 13+ for better query optimization

### System Requirements
- **Storage**: ~10GB free space for staging tables
- **Time**: Full pipeline takes ~2-4 hours on standard hardware

#### Configuration Recommendations

Add to your PostgreSQL configuration or set per session:

```sql
-- Increase memory for complex queries
SET work_mem = '256MB';              -- Default: 4MB

-- Increase memory for index creation
SET maintenance_work_mem = '512MB';  -- Default: 64MB

-- Enable parallel execution
SET max_parallel_workers_per_gather = 4;  -- Adjust based on CPU cores

-- Optimize for SSD storage
SET random_page_cost = 1.1;          -- Default: 4.0 (HDD)
```


---

## Usage Guide

### Basic Analysis Queries

#### 1. Compare Lab Values Across Groups
```sql
SELECT 
    'COVID' as cohort,
    lab.label,
    COUNT(*) as measurement_count,
    AVG(f.lab_valuenum) as mean_value,
    STDDEV(f.lab_valuenum) as std_value
FROM analysis.fact_lab_covid_first f
JOIN analysis.ref_lab_item lab ON f.lab_itemid = lab.itemid
WHERE f.lab_valuenum IS NOT NULL
GROUP BY lab.label

UNION ALL

SELECT 
    'Post-Control' as cohort,
    lab.label,
    COUNT(*) as measurement_count,
    AVG(f.lab_valuenum) as mean_value,
    STDDEV(f.lab_valuenum) as std_value
FROM analysis.fact_lab_control_post_first f
JOIN analysis.ref_lab_item lab ON f.lab_itemid = lab.itemid
WHERE f.lab_valuenum IS NOT NULL
GROUP BY lab.label;
```

#### 2. Analyze Readmission Patterns
```sql
SELECT 
    CASE 
        WHEN days_gap <= 30 THEN '0-30 days'
        WHEN days_gap <= 90 THEN '31-90 days'
        WHEN days_gap <= 180 THEN '91-180 days'
        ELSE '180+ days'
    END as readmission_window,
    COUNT(*) as readmissions,
    SUM(covid) as covid_reinfections,
    ROUND(100.0 * SUM(covid) / COUNT(*), 2) as reinfection_rate_pct
FROM analysis.demo_admission_covid_future
GROUP BY 1
ORDER BY 1;
```

#### 3. Lab Value Trajectories
```sql
-- Compare first admission vs readmission for COVID patients
SELECT 
    lab.label,
    AVG(first.lab_valuenum) as first_admission_mean,
    AVG(future.lab_valuenum) as readmission_mean,
    AVG(future.lab_valuenum) - AVG(first.lab_valuenum) as mean_change
FROM analysis.fact_lab_covid_first first
JOIN analysis.fact_lab_covid_future_first future
    ON first.lab_itemid = future.lab_itemid
JOIN analysis.ref_lab_item lab
    ON first.lab_itemid = lab.itemid
WHERE first.lab_valuenum IS NOT NULL
  AND future.lab_valuenum IS NOT NULL
GROUP BY lab.label
ORDER BY ABS(mean_change) DESC
LIMIT 20;
```

### Data Export

#### Export to CSV
```sql
-- Export COVID demographics
\copy (SELECT * FROM analysis.demo_patient_covid) TO '/tmp/covid_demographics.csv' CSV HEADER;

-- Export lab comparison
\copy (SELECT * FROM analysis.fact_lab_covid_first) TO '/tmp/covid_labs_first.csv' CSV HEADER;
```

### Performance Tips

1. **Use indexed columns in WHERE clauses**:
   - `subject_id`, `hadm_id`, `lab_itemid`, `days_gap`

2. **Leverage first-value tables**:
   - Use `*_first` tables for baseline comparisons
   - Reduces data volume by ~80%

3. **Filter early**:
   - Apply temporal filters before joins
   - Use `EXISTS` instead of `IN` for large subqueries

4. **Analyze tables after bulk operations**:
   ```sql
   ANALYZE analysis.fact_lab_covid;
   ```

---

## Key Design Decisions

### 1. Age-Based Temporal Ordering
- Uses patient age as proxy for chronological time
- Formula: `days_gap = (current_age - baseline_age) × 365`
- Preserves de-identification while enabling temporal analysis

### 2. Control Group Sampling
- Limited to 20,000 patients per group for computational efficiency
- Deterministic sampling using `hashtext(subject_id)` for reproducibility
- Balanced comparison with COVID cohort size

### 3. Top 61 Lab Items
- Selected based on frequency in COVID cohort
- Ensures adequate sample size for statistical comparisons
- Includes standard CBC, CMP, coagulation, and inflammatory markers

### 4. First Value Strategy
- Reduces data volume while capturing admission baseline
- Enables consistent comparison across admissions
- Separate tables (`*_first`) for efficient querying

---

## Benefits of This Architecture

### 1. Clear Hierarchy
```
analysis.
├── ref_*              (Reference/Lookup tables)
├── demo_patient_*     (Patient demographics)
├── demo_admission_*   (Admission demographics)
├── fact_lab_*         (Lab measurements)
└── bridge_*           (Relationship tracking)
```

### 2. Self-Documenting Names
- `demo_patient_covid` → Demographics table, patient-level, COVID group
- `fact_lab_control_post_first` → Fact table, lab data, post-2020 control, first values

### 3. Efficient Querying
```sql
-- Get all patient demographics tables
SELECT * FROM information_schema.tables
WHERE table_name LIKE 'demo_patient%';

-- Get all COVID group tables
SELECT * FROM information_schema.tables
WHERE table_name LIKE '%_covid%';

-- Get all first-value fact tables
SELECT * FROM information_schema.tables
WHERE table_name LIKE 'fact_lab%_first';
```

### 4. Scalability
Easy to extend with new cohorts:
- `analysis.demo_patient_vaccinated` (new study group)
- `analysis.fact_lab_severe_covid` (severity stratification)
- `analysis.demo_patient_long_covid` (long-term follow-up)

### 5. Standard Data Warehouse Pattern
Follows Kimball dimensional modeling:
- **Staging** = Source data preparation
- **Dimensions** = Who, what, where, when (demographics)
- **Facts** = Measurements, events (lab values)
- **Bridge** = Complex relationships (readmissions)

---

## Troubleshooting

### Common Issues

1. **Out of memory errors**:
   - Increase PostgreSQL `work_mem` setting
   - Process data in smaller batches
   - Ensure adequate system RAM

2. **Slow query performance**:
   - Run `ANALYZE` on all tables
   - Check index usage with `EXPLAIN ANALYZE`
   - Verify table statistics are up-to-date

3. **Missing tables**:
   - Verify script execution order
   - Check for error messages in previous scripts
   - Ensure MIMIC-IV tables are accessible

4. **Index creation failures**:
   - Check disk space
   - Verify column names match schema
   - Ensure no duplicate index names

---

## License and Citation

This pipeline is designed for use with the MIMIC-IV database. Users must:
1. Complete required training (CITI Data or Specimens Only Research)
2. Sign the data use agreement
3. Cite MIMIC-IV in any publications

**MIMIC-IV Citation**:
```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). 
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
```

---

## Contact and Support

For questions about this pipeline:
- Review script comments for detailed logic
- Check MIMIC-IV documentation: https://mimic.mit.edu/
- Verify table relationships in this README

---

**Last Updated**: 2025  
**Pipeline Version**: 1.0  
**Compatible with**: MIMIC-IV v3.1+