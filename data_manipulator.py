import settings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##### CLINICAL DATA MANIPULATION ######

def exploreNaNs(df: pd.DataFrame, figure_size: tuple = (12, 8), colormap: str = 'viridis') -> None:
  """
  Plots a heatmap to show in the whole dataframe which elements are NaNs or  normal values.
  It helps understanding NaNs pattern in the data.

  Keyword arguments:
  df -- Pandas DataFrame
  figure_size -- tuple defining figure size (default (12, 8))
  colormap -- colormap for the heatmap (default viridis)
  """
  plt.figure(figsize = figure_size)
  sns.heatmap(df.isnull(), cmap = colormap)
  plt.title('Missing Data Heatmap')
  plt.show()

def exploreValues(df: pd.DataFrame, figure_size: tuple = (12, 8), colormap: str = 'viridis') -> None:
  """
  Plots a heatmap to show in the whole dataframe which elements are NaNs or  normal values.
  It helps understanding NaNs pattern in the data.

  Keyword arguments:
  df -- Pandas DataFrame
  figure_size -- tuple defining figure size (default (12, 8))
  colormap -- colormap for the heatmap (default viridis)
  """
  plt.figure(figsize = figure_size)
  sns.heatmap(df, cmap = colormap)
  plt.title('Missing Data Heatmap')
  plt.show()


def verify_index_uniqueness(df: pd.DataFrame) -> bool:
  """
  Verifies if the index of the dataframe is unique. Returns true if so, else false.

  Keyword arguments:
  df -- Pandas DataFrame
  """
  dataframe_uniqueness = df.index.is_unique
  if (not dataframe_uniqueness):
    print("The dataframe does not have unique indexes")
    return False
  else:
    return True

def verify_clinical_df_index_prefixes(df: pd.DataFrame, prefix_list = settings.INDEX_PREFIXES) -> None:
  """
  Verifies that all the index labels have as prefix one of the string contained in a given list.

  Keyword arguments:
  df -- Pandas DataFrame
  prefix_list -- list containing the prefixes (default settings.INDEX_PREFIXES)
  """
  total_rows: int = df.shape[0]
  count_rows: int = 0
  for prefix in prefix_list:
    prefix_count = df.index.str.startswith(prefix).sum()
    count_rows += prefix_count
    print(f"{prefix} has {prefix_count}")

  if (total_rows != count_rows):
    print(f"{total_rows - count_rows} rows are not in {", ".join(settings.INDEX_PREFIXES)}")
  else:
    print(f"All rows are in {", ".join(settings.INDEX_PREFIXES)}")

def consolidate_columns(rows:pd.DataFrame) -> list:
  """
  Returns the index to drop when there are 2 indexes from same patients.
  It first checks if they agree on columns where they are both not nan and returns the
  index of the row with less not nan columns.
  If they do not agree, it returns the same with the columns non concodants to set to NaN

  Keyword arguments:
  rows -- Pandas DataFrame of selected rows
  """
  print(f"rows agree? {rows_agree_notna(rows)}")
  if rows_agree_notna(rows):
    return [rows.notna().sum(axis=1).idxmax()]
  else:
    non_concordant_cols = rows.columns[rows.notna().all(axis=0) & (rows.nunique() > 1)].tolist()

    #taking one and putting NaN in its place
    return [rows.notna().sum(axis=1).idxmax(), non_concordant_cols]




def rows_agree_notna(rows: pd.DataFrame) -> bool:
  """
  Helper function that checks if multiple rows have same labels in columns where they are not na.

  Keyword arguments:
  rows -- Pandas DataFrame of selected rows
  """
  # print(f">{rows.loc[:,rows.notna().all()].apply(lambda col: len(col.unique()) == 1, axis=0)}")
  return bool(rows.loc[:,rows.notna().all()].apply(lambda col: len(col.unique()) == 1, axis=0).all())

def get_double_donor_rows(df:pd.DataFrame, id_pos: int, dataset: str)-> bool:
  """
  Helper function that checks if a subset of the dataset extracted using 'dataset' at the beginning
  of the index and the donor id at position in the DATASET-ID1-ID2-... 'id_pos') does have unique donor ids.

  Keyword arguments:
  df -- Pandas DataFrame
  id_pos -- int place in the index 'barcode' where to find patient id.
  dataset -- str name of the sub-dataset, i.e. start of the index string
  """

  def get_participant_id(s: str) -> str:
    nonlocal id_pos
    return [s.split("-")[id_pos], s]

  patients_list = df[df.index.str.startswith(dataset)].index.map(get_participant_id).tolist()

  patients_dict = defaultdict(list)
  for pid, idx in patients_list:
    patients_dict[pid].append(idx)

  all_rows = []
  for el in patients_dict:
    if len(patients_dict[el]) > 1:
      all_rows.append(patients_dict[el])

  return all_rows


def verify_double_donor(df:pd.DataFrame, id_pos: int, dataset: str)-> bool:
  """
  Helper function that checks if a subset of the dataset extracted using 'dataset' at the beginning
  of the index and the donor id at position in the DATASET-ID1-ID2-... 'id_pos') does have unique donor ids.

  Keyword arguments:
  df -- Pandas DataFrame
  id_pos -- int place in the index 'barcode' where to find patient id.
  dataset -- str name of the sub-dataset, i.e. start of the index string
  """

  def get_participant_id(s: str) -> str:
    nonlocal id_pos
    return [s.split("-")[id_pos], s]

  patients_list = df[df.index.str.startswith(dataset)].index.map(get_participant_id).tolist()

  cols = set()
  patients_dict = defaultdict(list)
  for pid, idx in patients_list:
    patients_dict[pid].append(idx)

  for el in patients_dict:
    if len(patients_dict[el]) > 1:
      rows = df.loc[patients_dict[el]]
      mask = rows.notna().all(axis=0) & (rows.nunique() > 1)
      cols.add(tuple(rows.columns[mask].tolist()))

  unique_patients_size = len(patients_dict)
  total_patients_size = len(patients_list)

  print(f"COLUMNS IN {dataset} WHERE THERE IS DIFFERENCE BETWEEN ROWS WITH SAME DONOR: {cols}\n")
  print(f"{dataset} set is {unique_patients_size}; list is {total_patients_size}.\n")

  if(unique_patients_size != total_patients_size): return False
  else: return True

def verify_all_index_participant_unique(df: pd.DataFrame) -> bool:
  """
  Verifies that the donor id for both GTEX and TCGA are unique.

  Keyword arguments:
  df -- Pandas DataFrame
  col1 -- str name of the first column
  col2 -- str name of the second column
  """
  # first get gtex
  is_gtex_unique = verify_double_donor(df, 1, "GTEX")

  # then get tcga
  is_tcga_unique = verify_double_donor(df, 2, "TCGA")

  if (is_gtex_unique and is_tcga_unique): return True
  else: return False


def compare_two_columns(df: pd.DataFrame, col1: str, col2: str) -> int:
  """
  Compares how many rows have the same label in two columns.
  Returns the number of rows with such property.

  Keyword arguments:
  df -- Pandas DataFrame
  col1 -- str name of the first column
  col2 -- str name of the second column
  """
  # rows_count = df.shape[0]
  comparison_mask = df[col1] == df[col2]
  comparison_sum = comparison_mask.sum()
  return comparison_sum
  # print(f"{col1} and {col2} have {comparison_sum} common elements out of {rows_count} ({comparison_sum/rows_count}).")

def verify_concordance_subsite_dist_prox(df: pd.DataFrame) -> bool:
  """
  Checks that value 'Right colon' in column 'Biopsy subsite' is concordant with value 'Proximal' in column 'Distal vs proximal';
  Checks that value 'Left colon' in column 'Biopsy subsite' is concordant with value 'Distal' in column 'Distal vs proximal';

  Keyword arguments:
  df -- Pandas DataFrame
  """
  # PROXIMAL == RIGHT SIDE
  # DISTAL == LEFT SIDE
  discordant_distal = (df['Biopsy subsite'] == 'Right colon') & (df['Distal vs proximal'] == 'Distal')
  discordant_proximal = (df['Biopsy subsite'] == 'Left colon') & (df['Distal vs proximal'] == 'Proximal')

  concordant_distal = (df['Biopsy subsite'] == 'Right colon') & (df['Distal vs proximal'] == 'Proximal')
  concordant_proximal = (df['Biopsy subsite'] == 'Left colon') & (df['Distal vs proximal'] == 'Distal')

  print(f"THESE VALUES SHOULD BE 0: {discordant_distal.sum()}\t{discordant_proximal.sum()}")
  print(f"THESE VALUES SHOULD BE > 0: {concordant_distal.sum()}\t{concordant_proximal.sum()}")

  if(discordant_distal.sum() != 0 or discordant_proximal.sum() != 0): return False
  else: return True


def verify_column2_subset_comun1(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
  """
  Returns a boolean mask of the rows for which col2 has value where col1 is NaN.

  Keyword arguments:
  df -- Pandas DataFrame
  col1 -- str name of the first column
  col2 -- str name of the second column
  """
  # col1_not_nan_mask = df[col1].isna()
  # col2_not_nan_mask = df[col2].isna()

  in_col2_not_col1 = df[col1].isna() & ~df[col2].isna()

  return in_col2_not_col1


def verify_concordance_columns_on_value(df: pd.DataFrame, col1: str, col2: str, value: str) -> pd.Series:
  """
  Returns a boolean mask of the rows for which columns col1 and col2 do not have both value in the same rows

  Keyword arguments:
  df -- Pandas DataFrame
  col1 -- str name of the first column
  col2 -- str name of the second column
  value -- str value to check is in the same rows
  """
  col1_has_value_mask = df[col1] == value
  col2_has_value_mask = df[col2] == value
  col1_not_nan_mask = df[col1].notna()
  col2_not_nan_mask = df[col2].notna()

  discordant_mask = (col1_has_value_mask & col2_not_nan_mask & ~col2_has_value_mask) | (col2_has_value_mask & col1_not_nan_mask & ~col1_has_value_mask)

  # print(f"FOUND {discordant_mask.sum()} discordant rows for value '{value}' between '{col1}' and '{col2}'")
  return discordant_mask


def verify_all_colon_are_right_or_left(df: pd.DataFrame) -> pd.Series:
  """
  Verifies that all the rows that have either biopsy site or primary site as colon have also the right or left specification

  Keyword arguments:
  df -- Pandas DataFrame
  """
  colon_mask = (df['Biopsy site'] == 'Colon') | (df['Primary site'] == 'Colon')
  df_colon = df.loc[colon_mask]
  row_number = df_colon.shape[0]
  print(f"There are {row_number} rows for colon samples.")
  non_valid_rows = (df_colon['Biopsy subsite'].isna() & df_colon['Distal vs proximal'].isna())
  count_non_valid_rows = non_valid_rows.sum()
  print(f"Found {count_non_valid_rows} rows in which primary site is colon but no subsite is specified.({count_non_valid_rows * 100 / row_number :.1f}%)")
  return non_valid_rows


def merge_site_columns(df: pd.DataFrame) -> None:
  """
  Merges the information of the biopsy site from different samples

  Keyword arguments:
  df -- Pandas DataFrame
  """
  pass

def infer_dataset_from_index(df: pd.DataFrame) -> pd.Series:
  """
  Returns a pd.Series that contain the source dataset based on index prefix

  Keyword arguments:
  df -- Pandas DataFrame
  """
  possible_prefixes = '|'.join(settings.INDEX_PREFIXES)
  source = df.index.str.extract(f'({possible_prefixes})', expand=False)
  return source

def print_counts(df: pd.DataFrame, col: str) -> None:
  counts = df[col].value_counts()
  print(f"----------\n{col} counts: {counts}\n")


##### EXPRESSION DATA MANIPULATION #####

def normalize_expression_data(df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
  """
  normalize the expression data using log2 and optionally scikitlearn scaler

  Keyword arguments:
  df -- Pandas DataFrame
  scale -- bool if to use sckitlearn scaler
  """
  expr_log = np.log2(df + 1)
  if scale:
    scaler = StandardScaler()
    return scaler.fit_transform(expr_log.T).T  # Scale across genes
  else:
    return expr_log


def get_pca_dataframe(normalized_values: np.ndarray, columns: pd.Series, num_components: int = 2) -> pd.DataFrame:
  pca = PCA(n_components=num_components)
  pca_result = pca.fit_transform(normalized_values.T)
  return (pd.DataFrame(
    pca_result[:, :num_components],
    columns=['PC' + str(x) for x in range(1, num_components+1) ],
    index=columns
  ) , pca)