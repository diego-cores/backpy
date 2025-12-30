"""
Utils module

Contains various utility functions for the operation of the main code.

Variables:
    logger (Logger): Logger variable.

Hidden variables:
    _ansi_re (Pattern): ANSI escape sequence pattern.

Classes:
    ProgressBar: Create a loading bar.

Functions:
    statistics_format: Returns statistics into a structured string.
    num_align: Aligns a number to the left or right with a maximum number of digits.
    round_r: Function to round a number to a specified number of significant 
        digits to the right of the decimal point.
    not_na: Function to apply a specified function to two values if neither is 
        `np.nan`, or return the non-`np.nan` value, or `np.nan` if both are 
        `np.nan`.
    correct_index: Function to correct index by converting it to float.
    calc_width: Function to calulate the width of 'index' 
        if it has not been calculated already.
    calc_day: Function to calculate the width of the index that each day has.
    text_fix: Function to fix or adjust text.
    mult_color: Multiply a color.
    diff_color: Makes a color darker or lighter.
    diff_ccolor: Differentiate a color from another color.
    plot_volume: Function to plot volume on a give `Axes`.
    plot_candles: Function to plot candles on a given `Axes`.
    plot_line: Plots a line with data on the provided `Axes`.
    plot_position: Function to plot a trading position.

Hidden Functions:
    _loop_data: Function to extract data from an API with a data per second limit.
"""

from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle 
from matplotlib.axes._axes import Axes
from matplotlib.dates import date2num
import matplotlib.colors
import matplotlib as mpl

from typing import Any, Callable, cast
from time import sleep, time
import threading
import logging
import shutil
import re

import pandas as pd
import numpy as np

from . import _commons as _cm

logger:logging.Logger = logging.getLogger(__name__)
_ansi_re = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

class ProgressBar():
    """
    Progress bar

    Create a loading bar.
    Call 'next' method for the next step.

    Attributes:
        size (int|None): Number of steps the loading bar will have.
        noprint (bool): If true, the loading bar is not printed and 
            is returned as a string.
        show_count (bool): Show the step count.

    Private Attributes:
        _step: Last step.
        _rsize: Number of points within the load bar.
        _adder: Dict with additional text to the loading bar.

    Methods:
        pass
    """

    size:int|None
    noprint:bool
    show_count:bool

    _step:int
    _rsize:int
    _adder:dict

    def __init__(self, size:int|None = None, count:bool = True, 
                rsize:int = 46, noprint:bool = False) -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            size (int|None, optional): Number of steps the loading bar will have.
            count (bool, optional): Show the step count.
            rsize (int, optional): Number of points within the load bar.
            noprint (bool, optional): If true, the loading bar is not printed 
                and is returned as a string.
        """

        if rsize < 1:
            raise ValueError("'rsize' can only be equal to or greater than 1.")

        self.size = size
        self.noprint = noprint
        self.show_count = count

        self._step = 0
        self._adder = {}
        self._rsize = rsize

    def adder_add(self, add:dict) -> None:
        """
        Adder add

        Add additional text to the loading bar.

        Args:
            add (dict): Dictionary where the keys will be the title and 
                the value can be a Callable.
        """

        self._adder.update(add)

    def adder_replace(self, replace:dict) -> None:
        """
        Adder replace

        Replace the dictionary with all the additional text with a new one.

        Args:
            replace (dict): Dictionary where the keys will be the title and 
                the value can be a Callable.
        """

        self._adder = replace

    def adder_clear(self) -> None:
        """
        Adder clear

        Remove all additional text.
        """

        self._adder = {}

    def adder_calc(self) -> str:
        """
        Adder calc

        Based on the '_adder' dictionary, it creates a text 
        that compiles all the additional text with the correct format.

        Return:
            str: Formated additional text.
        """

        if not len(self._adder):
            return ''

        text = ''
        for k,v in self._adder.items():
            if isinstance(v, Callable):
                v = v()

            text += f'| {str(k).strip()}: {str(v).strip()} '
        return text

    def console_adjust(self, text:str) -> str:
        """
        Console adjust

        Adjust the text to the current console size.

        Args:
            text (str): text.

        Return:
            str: Adjusted text.
        """

        console_size = shutil.get_terminal_size().columns
        if len(_ansi_re.sub('', text)) > console_size:
            text = text[:console_size-4] + '...'

        return text

    def reset_size(self, size:int) -> None:
        """
        Reset size

        Reset the loading bar.

        Args:
            size (int): New size.
        """

        self.size = size
        self._step = 0

    def next(self) -> str|None:
        """
        Next

        Call this method to advance the loading bar; 
        if size is None or the loading bar is full, nothing is done.

        Return:
            str|None: Returns the loading bar as a string if 'noprint'.
        """

        if self.size is None or self._step >= self.size:
            return

        self._step += 1
        per = str(int(self._step/self.size*100))
        load = (
            '*'*int(self._rsize*self._step/self.size)
            + ' '*(self._rsize-int(self._rsize*self._step/self.size))
        )

        first = load[:self._rsize//2-int(round(len(per)/2,0))]
        sec = load[self._rsize//2+int(len(per)-round(len(per)/2,0)):]

        text = self.adder_calc()
        progress_bar = (
            f'\r\033[K[{first}{per}%%{sec}] ' 
            + (f' {num_align(self._step, len(str(self.size)))} of {self.size} completed ' 
                if self.show_count else '')
            + (text if text.strip() != '' else '')
        )

        if not self.noprint:
            print(
                self.console_adjust(progress_bar), 
                end='\n' if self._step >= self.size else '', 
                flush=True)

        else: return progress_bar

def statistics_format(data:dict, title:str | None = None, 
                      val_tt:list = ['Metric', 'Value']) -> str:
    """
    Statistics format.

    This function takes a dictionary of metrics and values, optionally with 
    colors, and formats them into a structured string for console output.

    Args:
        data (dict): A dictionary containing metrics and values. Format: 
            {
            "metric":"value",
            "another_metric": ["value", backpy._commons.__COLORS['COLOR']]
            }
            If a metric value is a list, the second element should be a color.
        title (str | None, optional): Statistics title.
        val_tt (list, optional): Name of the metrics and values.

    Returns:
        str: A formatted string representation of the statistics.
    """
    if not bool(title): title = ''

    values = []
    for i in data:
        values.append([i+':'])
        if isinstance(data[i], list) and len(data[i]) > 1:
            values[-1].extend([
                data[i][1]+str(data[i][0])+_cm.__COLORS['RESET'], str(data[i][0])
                ])
        else:
            values[-1].append(
                str(data[i][0]) if isinstance(data[i], list) else str(data[i]))

    min_len = max(
        len(max(values, key=lambda x: len(x[0]))[0]) + 
            len(max(values, key=lambda x: len(x[-1]))[-1]) + 1 - len(title), 
        len(' '.join(val_tt))-len(title),
        4
    )

    text = f"""
    {val_tt[0]+' '*(min_len+len(title)-len(''.join(val_tt)))+val_tt[1]}
    {(_cm.__COLORS['BOLD'] + 
      '-'*(min_len//2) + 
      title + '-'*(min_len//2) + 
      ('-' if min_len//2 != min_len/2 else '') + _cm.__COLORS['RESET'])}
    """
    text +='\n'.join(values[i][0] + 
                     ' '*(min_len+len(title)-len(values[i][0]+values[i][-1])) + 
                     values[i][1] for i in range(len(values)))
    text = text_fix(text, False)

    return text

def num_align(num:float|np.floating, digits:int = 4, side:bool = True) -> str:
    """
    Align a number.

    Aligns a number to the left or right with a maximum number of digits.

    Args:
        num (float|floating): The number to align.
        digits (int, optional): Total number of digits in the result.
        side (bool, optional): Align to the left (False) or right (True).

    Returns:
        str: The aligned number as a string.
    """
    num_ = str(num)
    int_part = str(int(num))

    side_c = lambda x: str(x).rjust(digits) if side else str(x).ljust(digits)

    if len(num_) == digits:
        return num_
    elif len(num_) < digits:
        return side_c(num_)
    elif len(int_part) == digits-1:
        return side_c(int_part)
    elif len(int_part) >= digits:
        simbol = '+' + '-' * (int_part[0] == '-')
        return simbol+int_part[-digits+len(simbol):] 

    data_s = side_c(round(num, digits-len(int_part)-1))
    return data_s if _cm.dots else data_s.replace('.', ',')

def round_r(num:float|np.floating, r:int = 1) -> float|np.floating:
    """
    Round right.

    Rounds `num` to have at most `r` significant digits to the right of the 
    decimal point. If `num` is `np.nan` or evaluates to `None`, it returns 0.

    Args:
        num (float|floating): The number to round.
        r (int, optional): Maximum number of significant digits to the right of 
            the decimal point. Defaults to 1.

    Returns:
        float|floating: The rounded number, or 0 if `num` is `np.nan` or evaluates to 
            `None`.
    """

    if np.isnan(num) or np.isinf(num) or num is None:
        return 0
    
    if int(num) != num:
        num = float(round(num) 
            if len(str(num).split('.')[0]) > r 
            else f'{{:.{r}g}}'.format(num))
            
    return num

def not_na(x:Any, y:Any, f:Callable = max) -> Any:
    """
    If not np.nan.

    Applies function `f` to `x` and `y` if neither of them are `np.nan`. If one 
    of them is `np.nan`, returns the value that is not `np.nan`. If both are 
    `np.nan`, `np.nan` is returned.

    Args:
        x (Any): The first value.
        y (Any): The second value.
        f (Callable, optional): Function to apply to `x` and `y` if neither is 
            `np.nan`. Defaults to `max`.

    Returns:
        Any: The result of applying `f` to `x` and `y`, or the non-`np.nan` value, 
            or `np.nan` if both are `np.nan`.
    """

    return y if np.isnan(x) else x if np.isnan(y) else f(x, y)

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
        r_index = date2num(index) # type: ignore[arg-type]
        logger.warning(text_fix("""
              The 'index' has been automatically corrected. 
              To resolve this, use a valid index.
              """))
    
    return r_index

def calc_width(index:pd.Index|np.ndarray, alert:bool = False) -> float:
    """
    Calc width.
    
    Calculate the width of `index` if it has not been calculated already.

    Args:
        index (Index|ndarray): The index of the data.
        alert (bool, optional): If `True`, an warning will be logged. Defaults to 
            False.

    Returns:
        float: The width of `index`.
    """

    if isinstance(_cm.__data_width, float) and _cm.__data_width > 0: 
        return _cm.__data_width

    if alert:
        logger.warning(text_fix("""
            The 'data_width' has been automatically corrected. 
            To resolve this, use a valid width.
            """))

    return 1. if len(index) <= 1 else np.median(np.diff(index))

def calc_day(interval:str = '1d', width:float = 1) -> float:
    """
    Calc day width.

    Function to calculate the width of the index that each day has.

    Args:
        interval (str, optional): The width interval.
        width (float, optional): The width of each candle.

    Returns:
        float: The width of the day.
    """

    match ''.join(filter(str.isalpha, interval)):
        case 'm' | 'min':
            unit_large = 1440
        case 'h':
            unit_large = 24
        case 'd' | 'day':
            unit_large = 1
        case 's' | 'wk' | 'w':
            unit_large = 1/7
        case 'mo' | 'M':
            unit_large = 1/(365/12)
        case _:
            unit_large = 1

    digit_str = ''.join(filter(str.isdigit, interval))
    return unit_large/(int(digit_str) if digit_str else 1)*width

def text_fix(text:str, newline_exclude:bool = True) -> str:
    """
    Text fix.

    Processes the `text` to remove common leading spaces and to remove line breaks or not. 

    Args:
        text (str): Text to process.
        newline_exclude (bool, optional): If True, excludes line breaks. Default is True.

    Returns:
        str: `text` without the common leading spaces on each line.
    """

    return ''.join(line.lstrip() + ('\n' if not newline_exclude else '')  
                        for line in text.split('\n'))

def mult_color(color:str, multiplier:float = 1) -> tuple[float, ...]:
    """
    Multiply a color

    Args:
        color (str): String allowed by Matplotlib as color.
        multiplier (float): multiplier.

    Return:
        tuple[float, ...]: Rgb color.
    """

    rgb = mpl.colors.to_rgb(color)
    return tuple(min(1., max(c*multiplier, 0)) for c in rgb)

def diff_color(color:str, factor:float = 1, 
               line:float = 0.2) -> tuple[float, ...]:
    """
    Different color

    Makes a color darker or lighter.
    Based on 'factor' the color will be modified to be different than 'color'.

    Args:
        color (str): String allowed by Matplotlib as color.
        factor (float, optional): Darkening/Lightening Factor. 
            Each RGB color with a value between 0 and 1 will 
            be multiplied by this factor.
        line (float, optional): Lightening barrier from 1 to 0 for the color 
            if when darkening it darkens less than the 'line' 
            number the opposite will be done (lightening).

    Returns:
        tuple[float, ...]: Rgb color.
    """

    if line > 1 or line < 0:
        raise ValueError("'line' can only be between 0 and 1 or equal.")
    elif factor > 1 or factor < 0:
        raise ValueError("'factor' can only be between 0 and 1 or equal.")

    rgb = mpl.colors.to_rgb(color) 

    return tuple((dk := c * (1 - factor),
                min(1., c * (1 + factor)) if dk <= line else dk)[1]
                for c in rgb)

def diff_ccolor(color:str, dcolor:str, factor:float = 1, 
               line:float = 0.2) -> tuple[float, ...]:
    """
    Different color to color

    Makes a color darker or lighter.
    Based on 'factor' the color will be modified to be different than 'dcolor'.

    Args:
        color (str): String allowed by Matplotlib as color.
        dcolor (str): Color to differentiate. String allowed by Matplotlib as color.
        factor (float, optional): Darkening/Lightening Factor. 
            Each RGB color with a value between 0 and 1 will 
            be multiplied by this factor.
        line (float, optional): Line of action, if 'color' is within the 
            range of 'dcolor'*(1+line), 'dcolor'*(1-line) the color is 
            modified, otherwise it is not modified.

    Returns:
        tuple[float, ...]: Rgb color.
    """

    if line > 1 or line < 0:
        raise ValueError("'line' can only be between 0 and 1 or equal.")
    elif factor > 1 or factor < 0:
        raise ValueError("'factor' can only be between 0 and 1 or equal.")

    rgb = mpl.colors.to_rgb(color) 
    rgb_d = mpl.colors.to_rgb(dcolor) 

    return tuple(rgb[i] if v >= rgb_d[i]*(1+line) or v <= rgb_d[i]*(1-line) 
                 else min(1., v*(1+factor)) for i,v in enumerate(rgb))

def plot_volume(ax:Axes, data:pd.Series, 
                width:float = 1, color:str = 'tab:orange', 
                alpha:float = 1) -> None:
    """
    Volume draw.

    Plots volume on the provided `ax`.

    Args:
        ax (Axes): The `Axes` object where the volume will be drawn.
        data (Series): Data to draw the volume.
        width (float, optional): Width of each bar. Defaults to 1.
        color (str, optional): Color of the volume. Defaults to 'tab:orange'.
        alpha (float, optional): Opacity of the volume. Defaults to 1.
    """

    x = data.index.to_numpy() - width / 2
    volume = data.to_numpy()

    patches = [Rectangle((xi, 0), width, vol) for xi, vol in zip(x, volume)]

    ax.add_collection(PatchCollection(patches, color=color, alpha=alpha, linewidth=0)) # type: ignore[arg-type]
    ax.set_ylim(top=data.max()*1.1 or 1)
    ax.set_xlim(data.index.values[0]-(width*len(data.index)/10), 
                data.index.values[-1]+(width*len(data.index)/10))

def plot_candles(ax:Axes, data:pd.DataFrame, 
                 width:float = 1, color_up:str = 'g', 
                 color_down:str = 'r', color_n:str = 'k',
                 alpha:float = 1) -> None:
    """
    Candles draw.

    Plots candles on the provided `ax`.

    Args:
        ax (Axes): The `Axes` object where the candles will be drawn.
        data (DataFrame): Data to draw the candles. 
            Need 'close', 'open', 'high', 'low' columns.
        width (float, optional): Width of each candle. Defaults to 1.
        color_up (str, optional): Color of the candle when the price rises. 
            Defaults to 'g'.
        color_down (str, optional): Color of the candle when the price falls. 
            Defaults to 'r'.
        color_n (str, optional): Color of the candle when the price does not move. 
            Defaults to 'k'.
        alpha (float, optional): Opacity of the candles. Defaults to 1.
    """

    color = data.apply(
               lambda x: (color_n if x['close'] == x['open'] else
                   color_up if x['close'] >= x['open'] else color_down), 
               axis=1)

    # Drawing vertical lines.
    segments = [[(x, low), (x, high)] 
                for x, low, high in zip(data.index, data['low'], data['high'])]
    ax.add_collection(LineCollection(segments, colors=color, alpha=alpha, # type: ignore[arg-type]
                                     linewidths=1, zorder=1))

    x = data.index.to_numpy() - width / 2
    y = np.minimum(data.loc[:, 'open'].to_numpy(), data.loc[:, 'close'].to_numpy())
    height = np.abs(data.loc[:, 'close'].to_numpy() - data.loc[:, 'open'])

    # Bar drawing.
    patches = [Rectangle((xi, yi), width, hi) for xi, yi, hi in zip(x, y, height)]
    ax.add_collection(PatchCollection(patches, color=color, alpha=alpha, linewidth=0, zorder=1)) # type: ignore[arg-type]

    ax.set_ylim(_cm.c_tf(data['low'].min())*0.98+1, _cm.c_tf(data['high'].max())*1.02+1)
    ax.set_xlim(data.index.values[0]-(width*len(data.index)/10), 
                data.index.values[-1]+(width*len(data.index)/10))

def plot_line(ax:Axes, data:pd.Series, 
            width:float = 1, color_up:str = 'g', 
            color_down:str = 'r', color_n:str = 'k',
            alpha:float = 1) -> None:
    """
    Line draw.

    Plots a line with data on the provided `ax`.

    Args:
        ax (Axes): The `Axes` object where the line will be drawn.
        data (Series): Data to draw the line.
        width (float, optional): Used to calculate the margin on the sides.
            Width of each point. Defaults to 1.
        color_up (str, optional): Color of the line when the price rises. 
            Defaults to 'g'.
        color_down (str, optional): Color of the line when the price falls. 
            Defaults to 'r'.
        color_n (str, optional): Color of the line when the price does not move. 
            Defaults to 'k'.
        alpha (float, optional): Opacity of the line. Defaults to 1.
    """

    x = data.index.to_numpy()
    y = data.to_numpy()

    groups:dict[str, dict] = {
        'n': {'xy':([],[]), 'color':color_n},
        'u': {'xy':([],[]), 'color':color_up},
        'd': {'xy':([],[]), 'color':color_down},
    }

    for a, b, c, d in zip(x[:-1], y[:-1], x[1:], y[1:]):    
        if b == d:
            key = 'n'
        elif b < d:
            key = 'u'
        else:
            key = 'd'

        groups[key]['xy'][0].extend([a, c, np.nan])
        groups[key]['xy'][1].extend([b, d, np.nan])

    ax.plot(groups['n']['xy'][0], groups['n']['xy'][1], color=groups['n']['color'], zorder=1, alpha=alpha)

    u_z = p_z = 1.01
    if y.sum() > 0:
        u_z = 1.02
    else:
        p_z = 1.02

    ax.plot(groups['u']['xy'][0], groups['u']['xy'][1], color=groups['u']['color'], zorder=u_z, alpha=alpha)
    ax.plot(groups['d']['xy'][0], groups['d']['xy'][1], color=groups['d']['color'], zorder=p_z, alpha=alpha)

    ax.set_ylim(_cm.c_tf(data.min())*0.98+1, _cm.c_tf(data.max())*1.02+1)
    ax.set_xlim(data.index.values[0]-(width*len(data.index)/10), 
                data.index.values[-1]+(width*len(data.index)/10))

def plot_position(trades:pd.DataFrame, ax:Axes, color_take:str = 'green', 
                  color_stop:str = 'red', alpha:float = 1, 
                  alpha_arrow:float = 1, arrow_fact:float = 0.2, 
                  operation_route:bool | None = True) -> None:
    """
    Position Draw.

    Plots positions on your `ax`.

    Args:
        trades (DataFrame): Trades data to draw.
        ax (Axes): Axes where it is drawn.
        color_take (str, optional): Color for positive positions. Default is 'green'.
        color_stop (str, optional): Color for negative positions. Default is 'red'.
        alpha (float, optional): Opacity of the elements. Default is 1.
        alpha_arrow (float, optional): Opacity of arrow, type marker, and close marker. Default is 1.
        operation_route (bool | None, optional): If True, traces the route of the operation. Default is True.
            None, it doesn't draw the color but it does draw the arrow.
        arrow_fact (float, optional): Indicates how much the colors of the arrows darken or lighten.
            If you don't want this to happen, leave it at 0.
        width_exit (callable, optional): Function that specifies how many time points the position 
            extends forward if not closed. Default is a lambda function with a width of 9.

    Info:
        The arrow and 'x' markers indicate where the position was closed.
        The 'V' and '^' markers indicate the direction of the position.
        The 'color_take' indicates where the take profit is placed and the 'color_stop'
        indicates where the stop loss is placed. If there is no take profit, its marker 
        will not be drawn; the same applies to the stop loss.
    """

    # Corrections
    if 'positionDate' not in trades.columns:
        trades['positionDate'] = np.nan
    if 'positionClose' not in trades.columns:
        trades['positionClose'] = np.nan
    if 'profitPer' not in trades.columns:
        trades['profitPer'] = np.nan

    # Draw routes of the operations.
    if operation_route or operation_route is None:

        segments:list = [[(d, o), (pd, c)] for d, o, pd, c in zip(
            trades['date'], 
            trades['positionOpen'],
            trades['positionDate'],
            trades['positionClose'],)]

        ax.add_collection(LineCollection( # type: ignore[arg-type]
            segments,
            colors="grey",
            linestyles=(0, (5, 5)),
            linewidths=0.8,
            alpha=alpha_arrow,
            zorder=cast(int, 0.9)
        ))

    color_close:list = trades.apply(
        lambda x: (
            diff_ccolor('#089991', color_take, factor=0.2) 
            if x['profitPer'] >= 0 
            else diff_ccolor('#f23651', color_stop, factor=0.2)), axis=1).to_list()

    if operation_route:
        routes = [
            Rectangle(
                (_cm.c_tf(xi), _cm.c_tf(yi)), (_cm.c_tf(pdi) - _cm.c_tf(xi)), hi)
            for xi, yi, hi, pdi in zip(
                trades['date'],
                trades['positionOpen'],
                trades['positionClose'] - trades['positionOpen'],
                trades['positionDate']
        )]

        ax.add_collection(PatchCollection(routes, color=color_close, # type: ignore[arg-type]
                                        alpha=alpha, linewidth=0, zorder=0.8))

    # Drawing of the closing marker of the operations.
    if ('positionDate' in trades.columns and 
        'positionClose' in trades.columns):

        ax.scatter(trades['positionDate'].to_numpy(), trades['positionClose'].to_numpy(), 
                  c=color_close, s=30, marker=MarkerStyle('x'), alpha=alpha_arrow, zorder=1.1)

    up_type:Callable = lambda x: x['positionOpen'] if x['typeSide'] == 1 else None
    down_type:Callable = lambda x: x['positionOpen'] if x['typeSide'] != 1 else None

    # Drawing of the position type marker.
    ax.scatter(trades['date'].to_numpy(), 
               trades.apply(up_type, axis=1), 
               color=diff_color(color_take, factor=arrow_fact, line=0.2), s=30, 
               marker=MarkerStyle('^'), alpha=alpha_arrow, zorder=1.1)

    ax.scatter(trades['date'].to_numpy(), 
               trades.apply(down_type, axis=1),
               color=diff_color(color_stop, factor=arrow_fact, line=0.2), s=30, 
               marker=MarkerStyle('v'), alpha=alpha_arrow, zorder=1.1)

def _loop_data(function:Callable, bpoint:Callable, init:int, timeout:float) -> pd.DataFrame:
    """
    Data loop.

    Runs a loop to extract data from an API with a data per second limit.

    Args:
        function (Callable): A callable function that retrieves data from an API or another source. 
            The function must accept an integer (`init`) as its argument and 
            return data in a format that can be converted to a pandas DataFrame.
        bpoint (Callable): A callable function used as a breaking point checker or updater. 
            It should accept the current DataFrame and optionally an integer `init`. 
            If provided with `init`, it should return `True` if the loop should stop. 
            If called without `init`, it should return the updated `init` value.
        init (int): The initial value to pass to the `function`. Commonly represents a starting 
            timestamp or other incremental parameter needed to retrieve paginated data.
        timeout (float): The time in seconds to wait between consecutive function calls, allowing 
            adherence to API rate limits.

    Returns:
        DataFrame: DataFrame with all extracted data.
    """
    data = pd.DataFrame()

    while True:
        data = pd.concat(
            [data, pd.DataFrame(function(init)).iloc[1:-1]], 
            ignore_index=True)

        if bpoint(data, init):
            break

        init = bpoint(data)
        sleep(timeout)

    return data
