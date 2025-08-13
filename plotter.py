import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def violin_xy(df: pd.DataFrame, x_lab: str, y_lab:str, title: str, figure_size: tuple = (10, 6)) -> None:
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
  plt.show()

def violin_y(df: pd.DataFrame, y_lab:str, title: str, figure_size: tuple = (10, 6)) -> None:
  """
  Plots a violin plot of x_lab and y_lab elements

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
  sns.heatmap(df, annot=True, fmt='d', cmap=colormap)
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


def pca_components(df: pd.DataFrame, pca: PCA, label: str, num_components: int, colormap='viridis') -> None:

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