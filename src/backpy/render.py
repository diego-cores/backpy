"""
Render module

Contains functions responsible for orchestrating the price chart.

Variables:
    logger (Logger): Logger variable.

Functions:
    gen_price_axes: Generate the axes to draw the price, volume and indicators.
    draw_indicators: Draw the indicators using 'plot_indicators'.
    correct_index: Function to correct index by converting it to float.
    get_width: Calculate the width of `index` if it has not been calculated already.
"""

from typing import Sequence
import logging

from matplotlib.legend_handler import HandlerTuple
from matplotlib.axes._axes import Axes
from matplotlib.dates import date2num
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from . import custom_plt as cpl
from . import _commons as _cm
from . import exception
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

def gen_price_axes(fig:plt.Figure|None = None, draw_price:bool = True, 
                draw_vol:bool = True, idc_names:Sequence[str] = []) -> dict[str, Axes]|None:
    """
    Gen price axes

    Generate the axes to draw the price, volume and indicators.
    Create name and axes pairs for each axes.

    Args:
        fig (Figure|None, optional): Matplotlib figure.
        draw_price (bool, optional): Add axes for price.
        draw_vol (bool, optional): Add axes for volume.
        idc_names (Sequence[str], optional): List of indicators keys to include.

    Returns:
        dict[str,Axes]: Dictionary with the name of the panel and its axis.
    """

    # Global
    axes_order = _cm.graph_panel_order
    diff_cf = _cm.graph_first_size

    if diff_cf < 1:
        raise exception.PlotError("'diff_cf' cannot be less than 1.")

    needed_axes = []
    if draw_vol: needed_axes = ['volume']

    for i,v in enumerate(idc_names):
        if not v in _cm.__plot_indicators.keys():
            continue

        panel = _cm.__plot_indicators[v]['panel'] or v
        if not panel in needed_axes:
            needed_axes.append(panel)

    axes_num = len(needed_axes)
    ax1 = None
    diff = 0

    if draw_price:
        if not 'price' in needed_axes: 
            needed_axes.append('price')
        else:
            axes_num -= 1

        diff = axes_num+diff_cf

        fig_kw = {'fig': fig} if fig is not None else {}
        ax1 = plt.subplot2grid((axes_num+diff,1), (0,0), rowspan=diff, 
                            colspan=1, **fig_kw)

    needed_axes.sort(key=lambda x: axes_order.get(x, 99))

    axes = cpl.gen_axes(axes_num, pad=diff, sharex=ax1, 
        autoget_sharex=True, fig=fig)
    if ax1: axes.insert(0, ax1)

    if not axes:
        logger.warning('Nothing to draw')
        return

    # Drawn indicators
    named_axes = dict(zip(needed_axes, axes))
    return named_axes

def draw_indicators(indicators:Sequence[str], named_axes:dict[str, Axes], x_index:Sequence, width:float) -> None:
    """
    Draw indicators

    Draw the indicators using 'plot_indicators'.

    Args:
        indicators (Sequence[str]): Indicators to be plot must be in 'plot_indicators'.
        named_axes (dict[str, Axes]): Dictionary with panel:axis where each indicator will be drawn.
        x_index (Sequence): Index x, used for drawing decoration.
        width (float): Width of each point, used for drawing decoration.
    """

    if not indicators:
        return
    if not all(map(lambda idc: idc in _cm.__plot_indicators, indicators)):
        raise exception.PlotError("All indicators must be in '__plot_indicators'.")

    colorin_axes = {i:[] for i in named_axes.keys()}

    # Global
    names_tolegend = _cm.graph_legend_pname
    legend_rname = _cm.graph_legend_rname
    legend_args = _cm.graph_legend_args
    fontsize = _cm.graph_legend_fontsize

    mathform = (lambda title, description: 
        rf'$\bf{{{utils.esc_latex(title)}}}$: $\it{{{utils.esc_latex(description)}}}$.')
    legends = {ax_nm:{'handles':[], 'labels':[]} for ax_nm in named_axes.keys()}

    names_put = ('price', 'volume')
    if names_tolegend and all([v in named_axes.keys() for v in names_put]):
        for v in names_put:
            legends[v]['handles'].append(tuple([Patch(facecolor='none', edgecolor='black', linewidth=1)]))
            legends[v]['labels'].append(rf'$\bf{{{utils.esc_latex(v.capitalize())}}}$')

    for idc in indicators:
        color = color if isinstance(color:=_cm.__plot_indicators[idc]['color'], list) else [color]
        data = _cm.__plot_indicators[idc]['data']
        panel = _cm.__plot_indicators[idc]['panel'] or idc
        rname = _cm.__plot_indicators[idc]['rname'] if legend_rname else idc
        adef = _cm.__plot_indicators[idc]['adef'] if legend_args else {}

        names = None
        if isinstance(data, pd.DataFrame):
            names = data.columns
        elif isinstance(data, np.ndarray) and data.dtype.names:
            names = data.dtype.names
        plot_data = [data[col].tolist() for col in names] if names is not None else [data]

        # Draw dec
        style_ax = _cm.__plot_indicators[idc].get('style', [])
        for stl in style_ax:
            if callable(stl): 
                stl(named_axes[panel], index=x_index, values=plot_data, width=width)

        artists = []
        style_ondraw = _cm.__plot_indicators[idc].get('styleOnDraw', [])

        for i,v in enumerate(plot_data):
            plot_color = None
            if i < len(color) and color[i] and not color[i] in colorin_axes[panel]:
                colorin_axes[panel].append(color[i])
                plot_color = color[i]

            # Draw
            if i < len(style_ondraw) and callable(style_ondraw[i]):
                artist = style_ondraw[i](
                    named_axes[panel], 
                    index=x_index, 
                    values=plot_data,
                    width=width,
                    zorder=0.5,
                ) 
            else: 
                artist = _cm.draw_plot(i, color=plot_color)(
                    named_axes[panel], 
                    index=x_index, 
                    values=plot_data,
                    zorder=0.5,
                )

            if artist: artists.extend(artist if isinstance(artist, list) else [artist])
        legends[panel]['handles'].append(tuple(i for i in artists))
        legends[panel]['labels'].append(mathform(rname.upper(), 
            f'{(', '.join(names) if names is not None else 'line')}{'; '+'; '.join(map(str,adef.values()))if adef else ''}'.lower()))

    for panel in legends:
        if not any(legends[panel]['handles']) or not legends[panel]['labels']:
            continue

        ndivide = max([len(handle) for handle in legends[panel]['handles']])
        named_axes[panel].legend(
            handles=legends[panel]['handles'], 
            labels=legends[panel]['labels'],
            handler_map={tuple: HandlerTuple(ndivide=ndivide, pad=0.5)},
            fontsize=fontsize,
            handlelength=ndivide,
            frameon=True,
            ncol=1,
            loc=2,
        )

def correct_index(index:pd.Index) -> np.ndarray|pd.Index:
    """
    Correct index.

    Correct `index` by converting it to float

    Args:
        index (Index): The `index` of the data to be corrected.

    Returns:
        ndarray|Index: The corrected `index`.
    """

    r_index:np.ndarray|pd.Index = index
    if not all(isinstance(ix, float) for ix in index):
        r_index = date2num(index) # pyrefly: ignore
        logger.warning(utils.text_fix("""
              The 'index' has been automatically corrected. 
              To resolve this, use a valid index.
              """))
    
    return r_index

def get_width(index:pd.Index|np.ndarray) -> float:
    """
    Get width

    Calculate the width of `index` if it has not been calculated already.
    It will generate a warning log.

    Args:
        index (Index|ndarray): The index of the data.

    Returns:
        float: The width of `index`.
    """
    if isinstance(_cm.__data_width, float) and _cm.__data_width > 0: 
        return _cm.__data_width

    logger.warning(utils.text_fix("""
        The 'data_width' has been automatically corrected. 
        To resolve this, use a valid width.
        """))

    return utils.calc_width(index=index)
