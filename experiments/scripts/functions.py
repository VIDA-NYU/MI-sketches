import io
import struct
import pandas as pd
import numpy as np
import scipy as sp
from IPython.display import Image, display
from typing import List

pd.set_option('display.precision', 3)

####

import altair as alt

alt.data_transformers.enable('data_server')
# alt.data_transformers.disable_max_rows()
# alt.renderers.enable('html')

SCALE_FACTOR = 2
OPACITY = 0.3

# Enabling altair_saver requires node and the following npm packages installed and chromedriver:
#   npm install -g vega-lite vega-cli canvas
#   sudo apt-get install chromium-chromedriver

alt.renderers.enable(
  'altair_saver',
  fmts=['pdf', 'html', 'png'],
  embed_options={
    'scaleFactor': f'{SCALE_FACTOR}',
  },
)

####

# 
# Configurations for seaborn/matplotlib
# 
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

px, py = 600, 400 # width/height in pixels
sns_dpi = int(py/4.8) # w, h = (6.4, 4.8) is the default width/height
sns_fig_width  = px/sns_dpi #sns_dpi*px
sns_fig_height = py/sns_dpi

sns.set(rc={
  'figure.figsize':(sns_fig_width, sns_fig_height),
  'figure.dpi': sns_dpi,
  'axes.labelweight': 'bold',
  'savefig.dpi': 4*sns_dpi,
})

# custom properties to make matplotlib/seaborn plots look like altair/vega plots
sns_axes_color = '.25'
alt_axes_stype = sns.axes_style('whitegrid')
alt_axes_stype['xtick.color'] = sns_axes_color
alt_axes_stype['ytick.color'] = sns_axes_color
alt_axes_stype['axes.edgecolor'] = sns_axes_color
alt_axes_stype['xtick.bottom'] = True
alt_axes_stype['ytick.left'] = True
alt_axes_stype['axes.grid'] = True
alt_axes_stype['axes.spines.right'] = False
alt_axes_stype['axes.spines.top'] = False
sns.set_style(alt_axes_stype)
# plt.rc("axes", labelweight="bold")

# sns.set_context("paper")
# sns_scale_factor = 1
# sns.set_context("notebook", font_scale=sns_scale_factor*1.5, rc={"lines.linewidth": sns_scale_factor*2.5})

####
DEFAULT_BACKEND = 'alt' # 'alt' for Altair or 'sns' for 'Seaborn'


def loess_with_error(df, x, y, color, chart_title=None, x_title=None, y_title=None,
                     error_type='stdev', bandwidth=0.1, step=0.1, save_img_at=None):
  chart = loess_with_error_chart(df, x, y, color, chart_title, x_title, y_title, error_type, bandwidth, step)
  return save_alt(chart, save_img_at)


def loess_with_error_chart(df, x, y, color: alt.Color, chart_title=None, x_title=None, y_title=None,
                     error_type='stdev', bandwidth=0.1, step=0.1):
  if y_title is None:
    y_title = y
  if x_title is None:
    x_title = x

  if type(color) is not str:
    color_groupby = color[1]
    color = color[0]
  else:
    color_groupby = color
  
  band = alt.Chart(df).mark_errorband(extent=error_type).encode(
    x=alt.X(x, title=x_title, bin=alt.BinParams(step=step)),
    y=alt.Y(y, title=y_title),
    color=color
  )
  
  band_base = alt.Chart(df).mark_errorband(extent=error_type).encode(
    x=alt.X(x, title=x_title),
    y=alt.Y(y, title=y_title),
    color=color
  )
  
  loess = band_base.transform_loess(x, y, bandwidth=bandwidth, groupby=[color_groupby]).mark_line(size=2)

  if chart_title is None:
    chart = alt.layer(band, loess)
  else:
    chart = alt.layer(band, loess, title = alt.TitleParams(chart_title, anchor='middle'))
    chart = chart.properties(
      title=chart_title
    )
  # chart.properties(width=width, height=height)
  # display(chart)
  return chart

def scatter(df, x, y, opacity=OPACITY, backend=DEFAULT_BACKEND, save_img_at=None,
            regression_line=False, color=None):
  if backend == 'alt':
    chart = scatter_alt(df, x=x, y=y, opacity=opacity, color=color)
    if regression_line:
      chart = chart + chart.transform_regression(x, y).mark_line()
    display(save_alt(chart, save_img_at))
  elif backend == 'sns' or backend == 'seaborn':
    plot = scatter_sns(df, x=x, y=y, opacity=opacity)
    save_sns(plot, save_img_at)
  else:
    raise ValueError(f'Unknown backend: {backend}')

def scatter_alt(df, x, y, x_title=None, y_title=None, chart_title=None,
                opacity=OPACITY, color=None, width=300, height=200):
  if y_title is None:
    y_title = y
  if x_title is None:
    x_title = x
  if chart_title != None:
    chart = alt.Chart(df, title=chart_title).mark_circle(size=30)
  else:
    chart = alt.Chart(df).mark_circle(size=30)
  if color != None:
    chart = chart.encode(
      x=alt.X(x, title=x_title),
      y=alt.Y(y, title=y_title),
      color=color,
      opacity=alt.value(opacity)
    )
  else:
    chart = chart.encode(
      x=alt.X(x, title=x_title),
      y=alt.Y(y, title=y_title),
      opacity=alt.value(opacity)
    )
  return chart.properties(width=width, height=height)

def scatter_sns(df, x, y, opacity=OPACITY):
  ax = sns.scatterplot(data=df, x=x, y=y, alpha=opacity, linewidth=0)
  ax.set(ylim=(0, None), xlim=(0, None))
  ax.get_figure()
  return ax

def histogram(df: pd.DataFrame, x: str, save_img_at: str = None, backend = DEFAULT_BACKEND):
  if backend == 'alt':
    chart = barplot(df, x=alt.X(x, bin=True), y='count()')
    display(chart)
    save_alt(chart, save_img_at)
  elif backend=='sns':
    plot = histogram_sns(df, x)
    save_sns(plot, save_img_at)

def histogram_sns(df, x, bins=10):
  ax = sns.histplot(data=df, x=x, bins=bins, lw=0, edgecolor="1")
  xlim = (np.min(df[x]), np.max(df[x]))
  ax.set(xlim = xlim)
  # ax.set(ylim=(0, None), xlim=(0, None))
  ax.get_figure()
  return ax

def barplot(df, x, y, save_img_at=None):
  chart = alt.Chart(df).mark_bar().encode(
    x=x,
    y=y,
  )
  save_alt(chart, save_img_at)
  return chart

def lineplot(df, x, y, color=None, save_img_at=None):
  chart = alt.Chart(df).mark_line().encode(
    x=x,
    y=y,
    color=color,
  )
  save_alt(chart, save_img_at)
  return chart

def save_sns(plot, save_img_at=None):
  fig = plot.get_figure()
  if save_img_at is not None:
    fig.savefig(save_img_at)

def save_alt(chart, filename: str, scale_factor=5):
  if filename is not None:
    chart.save(filename, scale_factor=scale_factor)
  return render_as_image(chart)

def scatter_hconcat(plots: List[alt.HConcatChart], legendX: 200, legendY=-80, save_img_at=None):    
  chart = scatter_hconcat_chart(plots, legendX=legendX, legendY=legendY)
  if save_img_at != None:
    save_alt(chart, filename=save_img_at)
  return render_as_image(chart, unconfined=False)

def scatter_hconcat_chart(plots: List[alt.HConcatChart], legendX: int, legendY: int):
  return alt.hconcat(*plots).configure_legend(
    orient='none',
    titleFontSize=18,
    labelFontSize=18,
    titleAnchor='middle',
    direction='horizontal',
    legendX=legendX,
    legendY=legendY,
    symbolOpacity=1,
    title=None,
    labelLimit=0,
  ).resolve_scale(
    y='shared',
    x='shared'
  ).configure_axis(
    labelFontSize=18,
    titleFontSize=18,
  ).configure_title(
    fontSize=20,
  )


def vconcat_hplots(plot_sets,
                   hconcat_shared_x='shared', hconcat_shared_y='shared',
                   vconcat_shared_x='shared', vconcat_shared_y='shared'):
  hcharts = [
    alt.hconcat(*plots).resolve_scale(y=hconcat_shared_x, x=hconcat_shared_y)
    for plots in plot_sets
  ]
  return alt.vconcat(*hcharts).configure_legend(
      orient='none',
      titleFontSize=18,
      labelFontSize=18,
      titleAnchor='middle',
      direction='horizontal',
      legendX=520,
      legendY=-80,
      symbolOpacity=1,
      title=None,
      labelLimit=0,
    ).resolve_scale(
      y=vconcat_shared_x,
      x=vconcat_shared_y,
    ).configure_axis(
      labelFontSize=18,
      titleFontSize=18,
    ).configure_title(
      fontSize=16,
    )


def render_as_image(chart, scale_factor=SCALE_FACTOR, unconfined=False):
  image_bytes = io.BytesIO()
  chart.save(image_bytes, format='png', scale_factor=scale_factor)
  
  image_bytes.seek(0)
  width, height = get_png_dimensions(image_bytes)
  width = width / scale_factor
  height = height / scale_factor
  
  image_bytes.seek(0)
  return Image(data=image_bytes.read(), 
               format='png',
               width=width,
               height=height,
               unconfined=unconfined)


def get_png_dimensions(image_bytes: io.BytesIO):
  '''
  Determine the dimensions of the PNG image stored in the input bytesIO
  '''
  head = image_bytes.read(24)
  if len(head) != 24:
    raise ValueError('Failed to read bytes from image_bytes.')
  check = struct.unpack('>i', head[4:8])[0]
  if check != 0x0d0a1a0a:
    raise ValueError('Failed to check magic number. The data does not look like a PNG file.')
  width, height = struct.unpack('>ii', head[16:24])
  return width, height

# def render_as_image(chart, width=None, height=None, scale_factor=1, unconfined=False, embed=True):
#   image_bytes = io.BytesIO()
  
#   chart.save(image_bytes, format='png', embed_options={
#     'scaleFactor': '1', 
#   })
#   image_bytes.seek(0)
#   if width != None:
#     width = width / scale_factor
#   if height != None:
#     height = height / scale_factor
#   print(width, height)
#   return Image(data=image_bytes.read(), embed=embed, width=width, height=height, unconfined=unconfined)

####

from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error

def compute_additional_columns(df: pd.DataFrame):
  # Replace NaN estimates by zero
  mi_ests = [
      'mi_est', 'nmi_max_est', 'nmi_sqrt_est', 'nmi_min_est',
      'mi_actual', 'nmi_max_actual', 'nmi_sqrt_actual', 'nmi_min_actual'
  ]
  df[mi_ests] = df[mi_ests].fillna(value=0)

  # Compute additional columns
  df['mi_delta'] = df['mi_actual'] - df['mi_est']
  df['mi_delta_abs'] = np.abs(df['mi_delta'])

  df['nmi_sqrt_delta'] = df['nmi_sqrt_actual'] - df['nmi_sqrt_est']
  df['nmi_sqrt_delta_abs'] = np.abs(df['nmi_sqrt_delta'])

  df['nmi_max_delta'] = df['nmi_max_actual'] - df['nmi_max_est']
  df['nmi_max_delta_abs'] = np.abs(df['nmi_max_delta'])

  df['nmi_min_delta'] = df['nmi_min_actual'] - df['nmi_min_est']
  df['nmi_min_delta_abs'] = np.abs(df['nmi_min_delta'])

  df['igr_x_est'] = df['mi_est']/df['ex_est']
  df['igr_x_actual'] = df['mi_actual']/df['ex_actual']
  df['igr_x_delta'] = df['igr_x_actual']/df['igr_x_est']
  df['igr_x_delta_abs'] = np.abs(df['igr_x_delta'])

  df['igr_y_est'] = df['mi_est']/df['ey_est']
  df['igr_y_actual'] = df['mi_actual']/df['ex_actual']
  df['igr_y_delta'] = df['igr_y_actual']/df['igr_y_est']
  df['igr_y_delta_abs'] = np.abs(df['igr_y_delta'])

  df['jcx_actual']  = df['interxy_actual'] / df['cardx_actual']
  df['jcy_actual']  = df['interxy_actual'] / df['cardy_actual']
  df['join_size_ratio']  = df['join_size_sketch'] / df['join_size_actual']


  # Rename join key distribution names
  if 'key_dist' in df.columns and df['key_dist'].dtype == 'O': # must exist and be of type object (str)
    df['key_dist'] = df['key_dist'].str.replace('SAME_AS_X', 'KeyDep')
    df['key_dist'] = df['key_dist'].str.replace('UNIQUE',    'KeyInd')

  df['pair_type'] = np.where(df.xtype != df.ytype, 'MIXED', df.xtype)
  df['pair_type_dist'] = df['pair_type'] + ' ' + df['key_dist']

  if 'estimator' not in df.columns:
    conditions = [
        df.pair_type == 'CATEGORICAL', 
        df.pair_type == 'NUMERICAL',
        df.pair_type == 'MIXED',
    ]
    choices= [
        'MLE',       # for Categorical
        'Mixed-KSG', # Numerical
        'DC-KSG',    # Mixed Pairs
    ]
    df['estimator'] = np.select(conditions, choices, default=None)


  if 'sketch_type' not in df.columns:
    df[['sketch_type','sketch_size']] = df.parameters.str.extract('(?P<sketch_type>.+)\:(?P<sketch_size>.+)\.0')
    df['sketch_type'] = df['sketch_type'].str.replace('KMV',   'LV2SK')
    df['sketch_type'] = df['sketch_type'].str.replace('SPPKF', 'TUPSK')
    df['sketch_type'] = df['sketch_type'].str.replace('PRISK', 'PRISK')
    df['sketch_n'] = df['sketch_size']

  # df['jsxy_actual'] = df['interxy_actual'] / df['unionxy_actual']
  return df

def correlations(a, b):
  sp = spearmanr(a, b)
  kt = kendalltau(a, b)
  pr = pearsonr(a, b)
  r2 = r2_score(a, b)
  rmse = mean_squared_error(a, b, squared=False)
  var_ab = np.var(np.abs(a-b))
  corrs = [
    ['Kendall\'s Tau', kt[0], kt[1]],
    ['Pearson\'s R', pr[0], pr[1]],
    ['Spearman\'s R', sp[0], sp[1]],
    ['R2', r2, None],
    ['RMSE', rmse, None],
    ['Var(a-b)', var_ab, None]
  ]
  return pd.DataFrame(corrs, columns=['correlation', 'value', 'p-value'])

def metrics_table(df: pd.DataFrame, a_name: str, b_name: str, df_filters=None, filter_invalid=False):
  metrics = []
  for df_filter, filter_name in df_filters:
    if filter_invalid:
      with pd.option_context('mode.use_inf_as_null', True):
        finite_values = (df[a_name].notnull())&(df[b_name].notnull())
        df_filter = (df_filter & finite_values) if df_filter is not None else finite_values
    df_filtered = df[df_filter] if df_filter is not None else df
    a = df_filtered[a_name]
    b = df_filtered[b_name]
    
    if len(a) < 2 or len(b) < 2:
      print(f'WARNING: vector has less than 2 entries after applying filters. Skipping.')
      continue
    
    non_zero = b > 0

    metrics.append([
      filter_name,
      # kendalltau(a, b)[0],
      pearsonr(a, b)[0],
      spearmanr(a, b)[0],
      r2_score(a, b),
      mean_squared_error(a, b),
      mean_squared_error(a, b, squared=False),
      np.mean(a-b),
      np.var(a-b),
      np.var(a/b),
      np.mean(a[non_zero]/b[non_zero]),
      a.shape[0]
    ])
  columns = [
    'Metric',
    # 'Kendall\'s Tau',
    'Pearson\'s R',
    'Spearman\'s R',
    'R2',
    'MSE',
    'RMSE',
    'Diff',
    'Var(a/b)',
    'Var(a-b)',
    'avg(a/b)',
    '# records'
  ]
  return pd.DataFrame(metrics, columns=columns)

####
