import pandas as pd

CLINICAL_DATA_PATH: str = 'data/clinical_data.csv'
GENE_EXPRESSION_PATH: str = 'data/expression_data.csv'

CLINICAL_DATA_TYPES: dict = {'Age' : pd.Int8Dtype(),
  'Age category' : 'string',
  'Biopsy subsite' : 'string',
  'BRAF mutation' : 'string',
  'Biopsy site' : 'string',
  'Biopsy timepoint' : 'string',
  'CMS' : 'string',
  'Cancer type' : 'string',
  'Distal vs proximal' : 'string',
  'Donor type' : 'string',
  'EGFR mutated' : 'string',
  'Grade' : 'string',
  'HER2 mutation' : 'string',
  'Histological subtype' : 'string',
  'Histological type' : 'string',
  'KRAS mutation' : 'string',
  'Line of treatment' : 'string',
  'MSI status' : 'string',
  'Molecular type' : 'string',
  'OS status' : 'string',
  'OS time' : pd.Int32Dtype(),
  'Primary site' : 'string',
  'Prior treatment' : 'string',
  'PFS status' : 'string',
  'PFS time' : pd.Int32Dtype(),
  'Sample type' : 'string',
  'Gender' : 'string',
  'Stage' : 'string',
  'TMB' : pd.Float32Dtype(),
  'TMB group' : 'string',
  'TP53 mutation' : 'string',
  'Therapy response' : 'string',
  'Therapy type' : 'string',
  'Tumor type' : 'string'}


# I DECIDED TO DIVIDE COLUMNS BY THE KIND OF METRICS THEY REPRESENT. THE GOAL IS TO CHECK THESE METRICS INTO SEPARATE GROUPS TO MAKE IT EASIER TO EXPLORE THE CLINICAL DATA

DEMOGRAPHIC_METRICS = ['Age', 'Gender']

TUMOR_METRICS = ['Cancer type', 'Tumor type' , 'Donor type', 'Grade', 'Histological subtype', 'Histological type', 'Stage']

MUTATION_METRICS = ['BRAF mutation', 'CMS', 'EGFR mutated', 'HER2 mutation', 'KRAS mutation','MSI status', 'Molecular type', 'TP53 mutation', 'TMB', 'TMB group']

CLINICAL_METRICS = ['Line of treatment','OS status',  'OS time', 'Prior treatment', 'Therapy response', 'Therapy type', 'PFS status', 'PFS time']

INDEX_PREFIXES = ["GSM", "GTEX", "TCGA"]

# AGE_LABELS = ["0-49","50-64","65+"]


BIOPSY_SITE_COLUMNS = ['Biopsy subsite', 'Biopsy site', 'Primary site', 'Distal vs proximal']

BIOPSY_SITES = ['Colon','Rectum','Colorectum','Rectosigmoid junction']

SELECTED_COLUMS: list = ['Age', 'Biopsy site', 'Donor type', 'Histological type', 'Sample type', 'Gender', 'Stage']

DROP_SELECTED_COLUMS: list = ['Tumor type','Cancer type','Age category','HER2 mutation', "Line of treatment" ] #'Grade'


# 'Tumor type', 'Cancer type' and 'Age category' are not selected because they does not provide any information (only 1 label possible)
# 'Grade' is not selected as it contains < 3% of the labels of the
# 'HER2 mutation' does not contain any valid value
# 'Line of treatment; has only 'First line'. No information


# NOTE: CHECK ON GTEX WICH SAMPLES ARE THE SAME AND WHICH NOT
# NOTE: CHECK IN TCGA IF THERE ARE SAMPLES THAT ARE TUMOR OR NOT