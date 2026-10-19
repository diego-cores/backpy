"""
Render module

Contains functions responsible for orchestrating the price chart.

Variables:
    logger (Logger): Logger variable.

Functions:
    draw_price: Draw the price in different styles.
    draw_volume: Draw the volume in different styles.
    draw_positions: Draw the positions in different styles.
    gen_price_axes: Generate the axes to draw the price, volume and indicators.
    calc_indicators: Calculate the result of the saved indicators.
    draw_indicators: Draw the indicators using 'plot_indicators'.
    check_duplicates: Check that there are no duplicates.
    check_gaps: Verify that all candles are present.
    correct_index: Function to correct index by converting it to float.
    get_width: Calculate the width of `index` if it has not been calculated already.
"""

from dataclasses import dataclass
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

def draw_price(draw_style:str, ax:Axes|None, data:pd.DataFrame, 
            width:float = 1, color_up:str = 'g', color_down:str = 'r', 
            alpha:float = 1, zorder:float = 1, lim:bool = True) -> None:
    """
    Draw price

    Draw the price in different styles.

    Draw styles:
        Documentation of this is in 'plot' docstring.

    Args:
        draw_style (str): Drawing style.
        ax (Axes | None): Axis to draw; if it is None, nothing will be drawn.
        data (DataFrame): Data to draw. Need 'close', 'open', 'high', 'low' columns.
        width (float, optional): Width of each point.
        color_up (str, optional): Color of the point when the price rises.
        color_down (str, optional): Color of the point when the price falls.
        alpha (float, optional): Drawing alpha.
        zorder (float, optional): Drawing Z-order, it can vary between the value and +0.02.
        lim (bool, optional): Configure the axis limits.
    """
    if ax is None: return

    match draw_style.lower().strip():
        case 'candle':
            utils.plot_candles(ax=ax, data=data, width=width*0.9, 
                color_up=color_up, color_down=color_down, 
                alpha=alpha, zorder=zorder, lim=lim)
        case 'line':
            utils.plot_line(ax=ax, data=data['close'], width=width, 
                color_up=color_up, color_down=color_down, 
                alpha=alpha, zorder=zorder, lim=lim)
        case 'none' | _:
            pass

def draw_volume(draw_style:str, ax:Axes|None, data:pd.DataFrame, 
            width:float = 1, color:str = 'tab:orange', 
            alpha:float = 1, zorder:float = 1, lim:bool = True) -> None:
    """
    Draw volume

    Draw the volume in different styles.

    Volume draw styles:
        Documentation of this is in 'plot' docstring.

    Args:
        draw_style (str): Drawing style.
        ax (Axes | None): Axis to draw; if it is None, nothing will be drawn.
        data (DataFrame): Data to draw. Need 'volume' column.
        width (float, optional): Width of each point.
        color (str, optional): Volume color.
        alpha (float, optional): Drawing alpha.
        zorder (float, optional): Drawing Z-order.
        lim (bool, optional): Configure the axis limits.
    """
    if ax is None: return

    match draw_style.lower().strip():
        case 'bar':
            utils.plot_volume(ax=ax, data=data['volume'], width=width, 
                            color=color, alpha=alpha, zorder=zorder, lim=lim)
        case 'none' | _:
            pass

def draw_positions(draw_style:str, ax:Axes|None, trades:pd.DataFrame, 
            color_up:str = 'g', color_down:str = 'r', 
            closing_markets:bool = True, markers_zorder:float = 1.1,
            alpha:float = 0.3, alpha_markers:float = 0.8, 
            zorder:float = 0.9) -> list:
    """
    Draw positions

    Draw the positions in different styles.

    Position draw styles:
        Documentation of this is in 'plot' docstring.

    Args:
        draw_style (str): Drawing style.
        ax (Axes | None): Axis to draw; if it is None, nothing will be drawn.
        trades (DataFrame): Trades to draw. Columns: see _commons.__get_trades.
        color_up (str, optional): Color of the point when the price rises.
        color_down (str, optional): Color of the point when the price falls.
        closing_markets (bool, optional): If True, draws markers for closing positions.
        markers_zorder (float, optional): Markers Z-order.
        alpha (float, optional): Drawing alpha.
        alpha_markers (float, optional): Markers alpha.
        zorder (float, optional): Drawing Z-order.

    Return:
        list: List with collections of the drawing.
    """
    if ax is None: return []

    result:list = []
    match draw_style.lower().strip():
        case 'complex'|'simple':
            result = utils.plot_position(
                trades=trades, ax=ax, 
                color_take=color_up,
                color_stop=color_down,
                operation_route=draw_style == 'complex', 
                closing_markets=closing_markets, alpha=alpha, 
                alpha_arrow=alpha_markers, route_zorder=zorder, 
                markers_zorder=markers_zorder)
        case 'none' | _:
            pass

    return result

def gen_price_axes(fig:plt.Figure|None = None, draw_price:bool = True, 
                draw_vol:bool = True, idc_names:Sequence[str]|None = None) -> dict[str, Axes]|None:
    """
    Gen price axes

    Generate the axes to draw the price, volume and indicators.
    Create name and axes pairs for each axes.

    Args:
        fig (Figure|None, optional): Matplotlib figure.
        draw_price (bool, optional): Add axes for price.
        draw_vol (bool, optional): Add axes for volume.
        idc_names (Sequence[str]|None, optional): List of indicators keys to include.

    Returns:
        dict[str,Axes]|None: Dictionary with the name of the panel and its axis.
    """

    # Global
    axes_order = _cm.user_config.graph_panel_order
    diff_cf = _cm.user_config.graph_first_size

    if diff_cf < 1:
        raise exception.PlotError("'diff_cf' cannot be less than 1.")

    needed_axes = []
    if draw_vol: needed_axes = ['volume']
    if idc_names is None: idc_names = []

    for i,v in enumerate(idc_names):
        if not v in _cm.plot_state._plot_indicators.keys():
            continue

        panel = _cm.plot_state._plot_indicators[v]['panel'] or v
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

def calc_indicators(data:pd.DataFrame, indicators:Sequence[str], 
                    width:float, replace:bool = False) -> None:
    """
    Calculate indicators

    Calculate the result of the saved indicators.

    Note:
        The result is stored as follows: {'data':result, 'context':context}.
        Where context is an instance of the frozen dataclass 'IndicatorContext' for
        store data from 'data', with these attributes: 'data_id', 'index', and 'width'.
        If 'data_id' is equal, the indicator is not recalculated unless 'replace' is true; 
        'data_id' is calculated as (len(data), data.index[0], data.index[-1]).

    Args:
        data (DataFrame): Data to calculate the indicators. 
            Columns: 'close', 'open', 'high', 'low', 'volume'. 
        indicators (Sequence[str]): Indicators to be calculated. 
            The indicators must be in 'plot_indicators'.
        width (float): Width of each point, used for drawing decoration.
        replace (bool, optional): Ignore 'context' and calculate the indicator anyway.
    """

    if not indicators:
        return
    elif not all(map(lambda idc: idc in _cm.plot_state._plot_indicators, indicators)):
        raise exception.PlotError("All indicators must be in '_plot_indicators'.")
    elif not all(map(lambda c: c in data.columns, ('close','open','high','low','volume'))):
        raise exception.PlotError(
            "'data' must have these columns: 'close','open','high','low','volume'.")

    @dataclass(frozen=True)
    class IndicatorContext():
        data_id:tuple
        index:np.ndarray
        width:float

    idx = data.index.to_numpy()
    context = IndicatorContext(
        data_id=(len(data), idx[0], idx[-1]), 
        index=idx, width=width)

    for idc in indicators:
        mem = _cm.plot_state._plot_indicators[idc].get('data', None)
        if mem and not replace and context.data_id == mem['context'].data_id:
            continue

        idc_data = _cm.plot_state._plot_indicators[idc]['call'](data)
        _cm.plot_state._plot_indicators[idc].update({
            'data':{
                'data':idc_data,
                'context':context,
            }})

def draw_indicators(indicators:Sequence[str], named_axes:dict[str, Axes], 
                only_idx:Sequence|np.ndarray|None = None) -> None:
    """
    Draw indicators

    Draw the indicators using 'plot_indicators'.
    Calculate the indicators before using 'calc_indicators'.

    Args:
        indicators (Sequence[str]): Indicators to be plot. 
            The indicators must be in 'plot_indicators'.
        named_axes (dict[str, Axes]): Dictionary with panel:axis where each 
            indicator will be drawn.
        only_idx (Sequence|np.ndarray|None, optional): Only draw the indices 
            in this sequence, it will apply a mask.
    """

    if not indicators:
        return
    elif not all(map(lambda idc: idc in _cm.plot_state._plot_indicators, indicators)):
        raise exception.PlotError("All indicators must be in '_plot_indicators'.")

    colorin_axes = {i:[] for i in named_axes.keys()}

    # Global
    names_tolegend = _cm.user_config.graph_legend_pname
    legend_rname = _cm.user_config.graph_legend_rname
    legend_args = _cm.user_config.graph_legend_args
    fontsize = _cm.user_config.graph_legend_fontsize

    mathform = (lambda title, description: 
        rf'$\bf{{{utils.esc_latex(title)}}}$: $\it{{{utils.esc_latex(description)}}}$.')
    legends = {ax_nm:{'handles':[], 'labels':[]} for ax_nm in named_axes.keys()}

    names_put = ('price', 'volume')
    if names_tolegend and all([v in named_axes.keys() for v in names_put]):
        for v in names_put:
            legends[v]['handles'].append(
                tuple([Patch(facecolor='none', edgecolor='black', linewidth=1)]))
            legends[v]['labels'].append(
                rf'$\bf{{{utils.esc_latex(v.capitalize())}}}$')

    for idc in indicators:
        color = color if isinstance(color:=_cm.plot_state._plot_indicators[idc]['color'], list) else [color]
        data = _cm.plot_state._plot_indicators[idc].get('data', None)
        if not data:
            raise RuntimeError(
                "Indicator without pre-calculated data. "
                "You must pre-calculate the data using 'calc_indicators'.")

        ctx_data = data['context']
        data = data['data']
        panel = _cm.plot_state._plot_indicators[idc]['panel'] or idc
        rname = _cm.plot_state._plot_indicators[idc]['rname'] if legend_rname else idc
        adef = _cm.plot_state._plot_indicators[idc]['adef'] if legend_args else {}

        names = None
        if isinstance(data, pd.DataFrame):
            names = data.columns
        elif isinstance(data, np.ndarray) and data.dtype.names:
            names = data.dtype.names

        didxm = np.isin(ctx_data.index, only_idx) if only_idx else slice(None)
        idx = ctx_data.index[didxm]
    
        plot_data = [data[col][didxm].tolist() for col in names] if names is not None else [data[didxm]]

        # Draw dec
        style_ax = _cm.plot_state._plot_indicators[idc].get('style', [])
        for stl in style_ax:
            if callable(stl): 
                stl(named_axes[panel], index=idx, values=plot_data, width=ctx_data.width)

        artists = []
        style_ondraw = _cm.plot_state._plot_indicators[idc].get('styleOnDraw', [])

        for i,v in enumerate(plot_data):
            plot_color = None
            if i < len(color) and color[i] and not color[i] in colorin_axes[panel]:
                colorin_axes[panel].append(color[i])
                plot_color = color[i]

            # Draw
            if i < len(style_ondraw) and callable(style_ondraw[i]):
                artist = style_ondraw[i](
                    named_axes[panel], 
                    index=idx, 
                    values=plot_data,
                    width=ctx_data.width,
                    zorder=0.5,
                ) 
            else: 
                artist = _cm.draw_plot(i, color=plot_color)(
                    named_axes[panel], 
                    index=idx, 
                    values=plot_data,
                    zorder=0.5,
                )

            if artist: artists.extend(artist if isinstance(artist, list) else [artist])
        legends[panel]['handles'].append(tuple(i for i in artists))
        legends[panel]['labels'].append(mathform(rname.upper(), 
            f"{(', '.join(names) if names is not None else 'line')}{'; '+'; '.join(
                map(str,adef.values()))if adef else ''}".lower()))

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

def check_duplicates(data:pd.DataFrame) -> None:
    """
    Check Duplicates

    Check that there are no duplicated index in the data; if there are, print a log.

    Args:
        data (pd.DataFrame): DataFrame.
    """

    duplicates = data.index.duplicated().sum()

    if duplicates > 0:
        logger.warning(
            "There are %i duplicated index, "
            "please verify that your data is correct.", 
            duplicates)

def check_gaps(index:pd.Index, interval:str, tol:float|None = None) -> None:
    """
    Check gaps

    Verify that all candles are present. 
    If there gaps, print a warning.

    Args:
        index (Index): The `index` of the data.
        interval (str): Interval check 'calc_day'.
        tol (float|None, optional): Adjust the tolerance to avoid 
            counting larger gaps. Value * number of candles in 1 day. 
            Leave it as None if you want to count all gaps.
    """

    diffs = np.diff(index.values)
    width = utils.calc_width(index)

    missing = np.rint(diffs / width).astype(int) - 1
    mask = missing > 0
    if tol is not None:
        mask &= missing <= tol * utils.calc_day(interval)

    if mask.any():
        logger.warning(
            "%i gaps found (%i missing candles, largest gap: %i).",
            mask.sum(), missing[mask].sum(), missing[mask].max())

def correct_index(index:pd.Index) -> np.ndarray|pd.Index:
    """
    Correct index.

    Correct index if it is not 'float' or is not in ascending order.

    Args:
        index (Index): The `index` of the data to be corrected.

    Returns:
        ndarray|Index: The corrected `index`.
    """

    r_index:np.ndarray|pd.Index = index

    if not all(isinstance(ix, float) for ix in index):
        r_index = date2num(index) # pyrefly: ignore
    if not all(np.diff(r_index) >= 0):
        r_index = np.arange(len(r_index))

    if not r_index is index:
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
    if isinstance(_cm.data_state._data_width, float) and _cm.data_state._data_width > 0: 
        return _cm.data_state._data_width

    logger.warning(utils.text_fix("""
        The 'data_width' has been automatically corrected. 
        To resolve this, use a valid width.
        """))

    return utils.calc_width(index=index)
