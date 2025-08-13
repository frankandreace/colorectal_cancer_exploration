import settings

import pandas as pd
import numpy as np

# LOADING CLINICAL DATA
def load_clinical_data() -> pd.DataFrame:
  """
  Loads the clinical data into a pandas DataFrame.
  """
  return pd.read_csv(settings.CLINICAL_DATA_PATH, header=0, index_col=0, dtype= settings.CLINICAL_DATA_TYPES)


# LOADING EXPRESSION DATA
def load_expression_data() -> pd.DataFrame:
  """
  Loads the clinical data into a pandas DataFrame.
  """
  return pd.read_csv(settings.GENE_EXPRESSION_PATH, header=0, index_col=0)
