CONFIG = {
    "modules_directory": "../../../Modules/",
    "datapath": "../../data/final-data/",
    "complete_datapath": "../../data/mimic_data/complete_data/",
    "image_save_path": "../../export/",
    "covid_data_file": "patient_covid.csv",
    "covid_train_data_file": "patient_covid_train.csv",
    "covid_test_data_file": "patient_covid_test.csv",
    "future_data_file": "patient_covid_future.csv",
    "figsize_cluster_heatmap": (12, 8),
    "figsize_future_heatmap": (12, 8),
    "control_group_train": "patient_control_pre_train.csv",
    "control_group_test": "patient_control_pre_test.csv",
    "control_group_readmission": "patient_control_pre_future.csv",
}

CATEGORICAL = {
    "aids": "AIDS",
    "cerebrovascular_disease": "Doença Cerebrovascular",
    "chronic_pulmonary_disease": "Doença Pulmonar Crônica",
    "congestive_heart_failure": "Insuficiência Cardíaca Congestiva",
    "dementia": "Demência",
    "diabetes_with_cc": "Diabetes com Complicações",
    "diabetes_without_cc": "Diabetes sem Complicações",
    "malignant_cancer": "Câncer Maligno",
    "metastatic_solid_tumor": "Tumor Sólido Metastático",
    "mild_liver_disease": "Doença Hepática Leve",
    "myocardial_infarct": "Infarto do Miocárdio",
    "paraplegia": "Paraplegia",
    "peptic_ulcer_disease": "Doença Ulcerosa Péptica",
    "peripheral_vascular_disease": "Doença Vascular Periférica",
    "renal_disease": "Doença Renal",
    "rheumatic_disease": "Doença Reumática",
    "severe_liver_disease": "Doença Hepática Grave",
    "died_in_stay": "Óbito na Internação",
    "died_after": "Óbito Pós-Alta",
    "died": "Óbito",
    "gender_M": "Gênero Masculino",
}

DATAPATH = "../../../../Mestrado-dados/data/final-data/"
COVID_TRAIN_FILE = "covid_train.csv"
COVID_TEST_FILE = "covid_test.csv"

"""
    Fonte dos valores normais: https://emedicine.medscape.com/article/2172316
"""

NORMAL_VALUES = {
    "Alanine Aminotransferase (ALT)_Chemistry_Blood_IU/L": (4, 36),
    "Albumin_Chemistry_Blood_g/dL": (3.5, 5.5),
    "Alkaline Phosphatase_Chemistry_Blood_IU/L": (30, 120),
    "Anion Gap_Chemistry_Blood_mEq/L": (
        8,
        16,
    ),  # https://emedicine.medscape.com/article/2087291
    "Asparate Aminotransferase (AST)_Chemistry_Blood_IU/L": (0, 35),
    "Bicarbonate_Chemistry_Blood_mEq/L": (21, 28),
    "Bilirubin, Total_Chemistry_Blood_mg/dL": (0.3, 1.0),
    "C-Reactive Protein_Chemistry_Blood_mg/L": (0, 10),
    "Calcium, Total_Chemistry_Blood_mg/dL": (9, 10.5),
    "Chloride_Chemistry_Blood_mEq/L": (98, 106),
    "Creatinine_Chemistry_Blood_mg/dL": (0.5, 1.2),
    "Ferritin_Chemistry_Blood_ng/mL": (10, 300),
    "Glucose_Chemistry_Blood_mg/dL": (74, 106),
    "Lactate Dehydrogenase (LD)_Chemistry_Blood_IU/L": (100, 190),
    "Magnesium_Chemistry_Blood_mg/dL": (1.3, 2.1),
    "Phosphate_Chemistry_Blood_mg/dL": (3, 4.5),
    "Potassium_Chemistry_Blood_mEq/L": (3.5, 5),
    "Sodium_Chemistry_Blood_mEq/L": (136, 145),
    "Urea Nitrogen_Chemistry_Blood_mg/dL": (10, 20),
    "Absolute Lymphocyte Count_Hematology_Blood_K/uL": (1.0, 4.0),
    "Basophils_Hematology_Blood_%": (
        0.5,
        1.0,
    ),  # https://my.clevelandclinic.org/health/body/23256-basophils
    "D-Dimer_Hematology_Blood_ng/mL|ng/mL FEU": (0, 500),
    "Eosinophils_Hematology_Blood_%": (1, 4),
    "Hematocrit_Hematology_Blood_%": (37, 50),
    "Hemoglobin_Hematology_Blood_g/dL": (12, 18),
    "INR(PT)_Hematology_Blood_nan": (
        0.8,
        1.1,
    ),  # https://www.mayoclinic.org/tests-procedures/prothrombin-time/about/pac-20384661
    "Lymphocytes_Hematology_Blood_%": (20, 40),
    "MCH_Hematology_Blood_pg": (
        27,
        31,
    ),  # https://emedicine.medscape.com/article/2054497-overview
    "MCHC_Hematology_Blood_%|g/dL": (
        32,
        36,
    ),  # https://emedicine.medscape.com/article/2054497-overview
    "MCV_Hematology_Blood_fL": (
        80,
        95,
    ),  # https://emedicine.medscape.com/article/2085770
    "Monocytes_Hematology_Blood_%": (2, 8),
    "Neutrophils_Hematology_Blood_%": (55, 70),
    "Platelet Count_Hematology_Blood_K/uL": (150, 400),
    "PT_Hematology_Blood_sec": (
        10,
        13,
    ),  # https://www.mayoclinic.org/tests-procedures/prothrombin-time/about/pac-20384661
    "PTT_Hematology_Blood_sec": (
        25,
        35,
    ),  # https://my.clevelandclinic.org/health/diagnostics/25101-partial-thromboplastin-time
    "RDW_Hematology_Blood_%": (
        11,
        14.5,
    ),  # https://emedicine.medscape.com/article/2098635-overview
    "Red Blood Cells_Hematology_Blood_m/uL": (
        3.95,
        5.65,
    ),  # https://www.mayoclinic.org/symptoms/high-red-blood-cell-count/basics/definition/sym-20050858
    "White Blood Cells_Hematology_Blood_K/uL": (5, 10),
    "pH_Hematology_Urine_units": (
        4.6,
        8.0,
    ),  # https://emedicine.medscape.com/article/2074001-overview
    "Specific Gravity_Hematology_Urine_nan": (
        1.005,
        1.030,
    ),  # https://emedicine.medscape.com/article/2074001-overview
    "Absolute Basophil Count_Hematology_Blood_K/uL": (
        0.0,
        0.3,
    ),  # https://my.clevelandclinic.org/health/body/23256-basophils
    "Absolute Eosinophil Count_Hematology_Blood_K/uL": (0.05, 0.5),
    "Absolute Monocyte Count_Hematology_Blood_K/uL": (0.1, 0.7),
    "Absolute Neutrophil Count_Hematology_Blood_K/uL": (2.5, 8.0),
    "Immature Granulocytes_Hematology_Blood_%": (
        0.0,
        2.0,
    ),  # https://my.clevelandclinic.org/health/body/22016-granulocytes
    "RDW-SD_Hematology_Blood_fL": (
        39,
        46,
    ),  # https://emedicine.medscape.com/article/2098635-overview
}

NUMERICAL = {
    "age": "Idade (anos)",
    "charlson_comorbidity_index": "Índice de Comorbidade de Charlson",
    "length_of_stay_days": "Duração da Estadia (dias)",
    "Absolute Basophil Count (K/uL)": "Contagem Absoluta de Basófilos (K/uL)",
    "Absolute Eosinophil Count (K/uL)": "Contagem Absoluta de Eosinófilos (K/uL)",
    "Absolute Lymphocyte Count (K/uL)": "Contagem Absoluta de Linfócitos (K/uL)",
    "Absolute Monocyte Count (K/uL)": "Contagem Absoluta de Monócitos (K/uL)",
    "Absolute Neutrophil Count (K/uL)": "Contagem Absoluta de Neutrófilos (K/uL)",
    "Basophils (%)": "Basófilos (%)",
    "D-Dimer (ng/mL|ng/mL FEU)": "D-Dímero (ng/mL)",
    "Eosinophils (%)": "Eosinófilos (%)",
    "Hematocrit (%)": "Hematócrito (%)",
    "Hemoglobin (g/dL)": "Hemoglobina (g/dL)",
    "INR(PT) (nan)": "INR(PT) (nan)",
    "Immature Granulocytes (%)": "Granulócitos Imaturos (%)",
    "Lymphocytes (%)": "Linfócitos (%)",
    "MCHC (%|g/dL)": "MCHC (%|g/dL)",
    "MCH (pg)": "MCH (pg)",
    "MCV (fL)": "MCV (fL)",
    "Monocytes (%)": "Monócitos (%)",
    "Neutrophils (%)": "Neutrófilos (%)",
    "PTT (sec)": "PTT (seg)",
    "PT (sec)": "PT (seg)",
    "Platelet Count (K/uL)": "Contagem de Plaquetas (K/uL)",
    "RDW-SD (fL)": "RDW-SD (fL)",
    "RDW (%)": "RDW (%)",
    "Red Blood Cells (m/uL)": "Glóbulos Vermelhos (m/uL)",
    "Specific Gravity (nan)": "Gravidade Específica (nan)",
    "White Blood Cells (K/uL)": "Glóbulos Brancos (K/uL)",
    "pH (units)": "pH (unidades)",
    "Alanine Aminotransferase (ALT) (IU/L)": "Alanina Aminotransferase (ALT) (IU/L)",
    "Albumin (g/dL)": "Albumina (g/dL)",
    "Alkaline Phosphatase (IU/L)": "Fosfatase Alcalina (IU/L)",
    "Anion Gap (mEq/L)": "Lacuna Aniônica (mEq/L)",
    "Asparate Aminotransferase (AST) (IU/L)": "Aspartato Aminotransferase (AST) (IU/L)",
    "Bicarbonate (mEq/L)": "Bicarbonato (mEq/L)",
    "Bilirubin, Total (mg/dL)": "Bilirrubina Total (mg/dL)",
    "C-Reactive Protein (mg/L)": "Proteína C-Reativa (mg/L)",
    "Calcium, Total (mg/dL)": "Cálcio Total (mg/dL)",
    "Chloride (mEq/L)": "Cloreto (mEq/L)",
    "Creatinine (mg/dL)": "Creatinina (mg/dL)",
    "Ferritin (ng/mL)": "Ferritina (ng/mL)",
    "Glucose (mg/dL)": "Glicose (mg/dL)",
    "Lactate Dehydrogenase (LD) (IU/L)": "Desidrogenase Lática (IU/L)",
    "Magnesium (mg/dL)": "Magnésio (mg/dL)",
    "Phosphate (mg/dL)": "Fosfato (mg/dL)",
    "Potassium (mEq/L)": "Potássio (mEq/L)",
    "Sodium (mEq/L)": "Sódio (mEq/L)",
    "Urea Nitrogen (mg/dL)": "Nitrogênio da Ureia (mg/dL)",
}
