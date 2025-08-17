import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter
from matplotlib_venn import venn2
import numpy as np

def violin_xy(df: pd.DataFrame, x_lab: str, y_lab:str, title: str, figure_size: tuple = (10, 6), pltlim: tuple = None) -> None:
  """
  Plots a violin plot of x_lab and y_lab elements

  Keyword arguments:
  df -- Pandas DataFrame
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  """
  plt.figure(figsize=figure_size)
  sns.violinplot(data=df, x=x_lab, y=y_lab, inner='quartile')
  plt.title(title)
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)

  if pltlim:
    plt.ylim(pltlim[0], pltlim[1])
  plt.show()

def violin_y(df: pd.DataFrame, y_lab:str, title: str, figure_size: tuple = (10, 6), pltlim: tuple = None) -> None:
  """
  Plots a violin plot of y_lab elements

  Keyword arguments:
  df -- Pandas DataFrame
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  """
  plt.figure(figsize=figure_size)
  sns.violinplot(y=y_lab, data=df, inner='quartile')
  plt.title(title)
  plt.ylabel(y_lab)
  if pltlim:
    plt.ylim(pltlim[0], pltlim[1])
  plt.show()

def venn(set1: set, set2: set, label1: str, label2: str, title: str, figure_size: tuple = (10, 6)) -> None:
  """
  Plots a venn diagram of term1 and term2 elements

  Keyword arguments:
  df -- Pandas DataFrame
  set1 -- set to compare
  set2 -- set to compare
  label1 -- str label of set1
  label2 -- str label of set2
  title -- str title of the plot
  figure_size -- tuple figure size
  """

  plt.figure(figsize=figure_size)
  venn2([set1, set2], set_labels=(label1, label2))
  plt.title(title)
  plt.show()


def heatmap(df: pd.DataFrame, x_lab: str, y_lab: str, title: str, figure_size: tuple = (12, 8), colormap: str = 'viridis') -> None:
  """
  Plots a heatmap plot of x_lab and y_lab elements

  Keyword arguments:
  df -- Pandas DataFrame
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  colormap -- str colormap to use
  """
  plt.figure(figsize=figure_size)
  sns.heatmap(df, annot=True, cmap=colormap)
  plt.title(title)
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.show()

def boxplot(df: pd.DataFrame, x_lab: str, y_lab: str, title: str, figure_size: tuple = (12, 8), colormap: str = 'viridis') -> None:
  """
  Plots a box plot of x_lab and y_lab elements

  Keyword arguments:
  df -- Pandas DataFrame
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  colormap -- str colormap to use
  """
  plt.figure(figsize=figure_size)
  sns.boxplot(data=df, x=x_lab, y=y_lab)
  plt.title(title)
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.show()

def histogram(df: pd.DataFrame, label: str, x_lab: str, y_lab: str, title: str, figure_size: tuple = (12, 8), xrange: tuple = None):
  """
  Histogram of a dataframe column

  Keyword arguments:
  df -- Pandas DataFrame
  label -- str the clumn name
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  """
  plt.figure(figsize=figure_size)
  sns.histplot(df[label].dropna(), bins=100)
  if xrange != None:
    plt.xlim(xrange[0],xrange[1])
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.title(title)

def pca_components(df: pd.DataFrame, pca: PCA, label: str, num_components: int, colormap='viridis') -> None:
  """
  Plots the PCA components

  Keyword arguments:
  df -- Pandas DataFrame
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size

  """
  _, axes = plt.subplots(num_components - 1, figsize=(8, 12))
  s=60
  alpha=0.7

  components = ["PC" + str(x) for x in range(1, num_components + 1)]
  component_loop = 0

  for ax in axes.flat:
      component_1 = components[component_loop]
      component_2 = components[component_loop + 1]


      if df[label].dtype in ['object', 'category','string']:
          # Categorical variable
          # print('categorical')
          sns.scatterplot(
              data=df, x=component_1, y=component_2,
              hue=label, ax=ax, s=s, alpha=alpha
          )
      else:
          # Continuous variable
          # print('continous')
          scatter = ax.scatter(
              df[component_1], df[component_2],
              c=df[label], cmap=colormap, s=s, alpha=alpha
          )
          plt.colorbar(scatter, ax=ax)
      ax.set_title(f'{component_1} vs {component_2} labeled by {label}')
      ax.set_xlabel(f'{component_1} ({pca.explained_variance_ratio_[component_loop]:.1%})')
      ax.set_ylabel(f'{component_2} ({pca.explained_variance_ratio_[component_loop+1]:.1%})')
      component_loop += 1



  plt.tight_layout()
  plt.show()


def volcano(df: pd.DataFrame, label1: str, label2: str, x_lab: str = 'log2 Fold Change', y_lab: str = '-log10(p-adjusted value)', title: str = 'Volcano plot of DEGs', figure_size: tuple = (12, 8)) -> None:
  """
  Volcano plot of two dataframe columns.

  Keyword arguments:
  df -- Pandas DataFrame
  label1 -- str label of adjusted p-value
  label2 -- str label of log2fold
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  """

  p_value_mask = (df[label1] < 0.01)
  significance_mask = p_value_mask & (abs(df[label2]) > 2)
  up_regulated_mask = p_value_mask & (df[label2] > 2)
  down_regulated_mask = p_value_mask & (df[label2] < -2)

  df['significance'] = 'Not Significant'
  df.loc[significance_mask, 'significance'] = 'Significant'
  df.loc[up_regulated_mask, 'significance'] = 'Up-regulated'
  df.loc[down_regulated_mask, 'significance'] = 'Down-regulated'

  df['log_padj'] = -np.log10(df['padj'])

  plt.figure(figsize=figure_size)

  sns.scatterplot(
      data=df,
      x=label2,
      y='log_padj',
      hue='significance',
      palette={'Up-regulated': 'red', 'Down-regulated': 'blue', 'Not Significant': 'gray'},
      alpha=0.6,
      s=20
  )

  points_to_label = df[df['significance'].isin(['Significant', 'Up-regulated', 'Down-regulated'])].iterrows()

  for index, row in points_to_label:
    plt.text(row[label2] + 0.01,
             row['log_padj'] + 0.01,
             str(index),
              ha='left',
              va='bottom')

  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.title(title)
  plt.legend()

def survival(df, x_lab: str, y_lab: str, title: str, figure_size: tuple = (12, 8)) -> None:
  """
  Survival plot of a DataFrame

  Keyword arguments:
  df -- Pandas DataFrame
  x_lab -- str column name to plot on x axis
  y_lab -- str column name to plot on y axis
  title -- str title of the plot
  figure_size -- tuple figure size
  """

  kmf = KaplanMeierFitter()

  plt.figure(figsize=figure_size)
  for grp in ['High','Low']:
      mask = df['TMB group'] == grp
      kmf.fit(durations=df.loc[mask, 'OS time'],
              event_observed=df.loc[mask, 'OS status'],
              label=f"TMB {grp} (n={mask.sum()})")
      kmf.plot()  # Do not set explicit colors per instructions
  plt.title("Overall Survival by TMB Group")
  plt.xlabel("Time (days)")
  plt.ylabel("Survival probability")
  plt.tight_layout()
  plt.show()
