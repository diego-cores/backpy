"""
Utils.
----
Different useful functions for the operation of main code.\n
Functions:
---
>>> load_bar
>>> round_r
>>> max_drawdown
>>> candles_plot
"""

from matplotlib.patches import Rectangle
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D

import matplotlib as mpl
import pandas as pd
import numpy as np

def load_bar(size:int, step:int, more:str = '') -> None:
    """
    Loading bar.
    ----
    Print the loading bar.\n
    Parameters:
    --
    >>> size:int
    >>> step:int
    >>> more:str = ''
    \n
    size: \n
    \tNumber of steps.\n
    step: \n
    \tstep.\n
    more: \n
    \tThis string appears to the right of the bar.\n
    """
    per = str(int(step/size*100))
    load = '*'*int(46*step/size) + ' '*(46-int(46*step/size))

    first = load[:46//2-int(round(len(per)/2,0))]
    sec = load[46//2+int(len(per)-round(len(per)/2,0)):]

    print('\r['+first+per+'%%'+sec+']'+f'  {step} of {size} completed '+more, end='')

def round_r(num:float, r:int = 1) -> float:
    """
    Round right.
    ----
    Returns the num rounded to have at most 'r' significant numbers to the right of the '.'.\n
    Parameters:
    --
    >>> num:float
    >>> r:int = 1
    \n
    num: \n
    \tNumber.\n
    r: \n
    \tMaximum significant numbers.\n
    """
    if int(num) != num:
        num = round(num) if len(str(num).split('.')[0]) > r else f'{{:.{r}g}}'.format(num)

    return num

def not_na(x:any, y:any, f:any = max):
    """
    If not np.nan.
    ----
    It passes to 'x' and 'y' by the function 'f'\n
    if neither of them are in np.nan, otherwise it returns the value that is not np.nan,\n
    if both are np.nan, np.nan is returned.\n
    Parameters:
    --
    >>> x:any
    >>> y:any
    >>> f:any = max
    \n
    x: \n
    \tx value.\n
    y: \n
    \ty value.\n
    f: \n
    \tFunction.\n
    """
    return y if np.isnan(x) else x if np.isnan(y) else f(x, y)

def max_drawdown(values:pd.Series) -> float:
    """
    Maximum drawdown.
    ----
    Returns the maximum drawdown.\n
    Parameters:
    --
    >>> values:pd.Series
    \n
    values: \n
    \tThe ordered data.\n
    """
    if values.empty: return 0
    max_drdwn, max_val = 0, values[values.index[0]]

    def calc(x):
        nonlocal max_drdwn, max_val

        if x > max_val: max_val = x
        else: 
            drdwn = (max_val - x) / max_val
            if drdwn > max_drdwn:
                max_drdwn = drdwn
    values.apply(calc)

    return max_drdwn * 100

def candles_plot(ax:Axes, data:pd.DataFrame, width:float = 1, color_up:str = 'g',color_down:str = 'r', alpha:str = 1) -> None:
    """
    Candles draw.
    ----
    Parameters:
    --
    >>> ax:Axes
    >>> data:pd.DataFrame
    >>> width:float = 1
    >>> color_up:str = 'g'
    >>> color_down:str = 'r'
    >>> alpha:str = 1
    \n
    ax: \n
    \tAxes where it is drawn.\n
    data: \n
    \tData to draw.\n
    width: \n
    \tWidth of each candle.\n
    color_up: \n
    \tCandle color when price rises.\n
    color_down: \n
    \tCandle color when price goes down.\n
    aplha: \n
    \tOpacity.\n
    """
    OFFSET = width / 2.

    def draw(row):
        color = color_up if row['Close'] >= row['Open'] else color_down

        line = Line2D(xdata=(row.name, row.name), ydata=(row['Low'], row['High']), color=color, linewidth=0.5)
        rect = Rectangle(xy=(row.name-OFFSET, min(row['Open'], row['Close'])), width=width, height=abs(row['Close']-row['Open']), facecolor=color, edgecolor=color)

        rect.set_alpha(alpha); line.set_alpha(alpha)
        ax.add_line(line); ax.add_patch(rect)

    data.apply(draw, axis=1)
    ax.autoscale_view()
