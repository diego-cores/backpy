"""
Utils.
----
Different useful functions for the operation of main code.

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

def load_bar(size:int, step:int) -> None:
    """
    Loading bar.
    ----
    Print the loading bar.

    Parameters:
    --
    >>> size:int
    >>> step:int
    
    size:
      Number of steps.
    step:
      step.
    """
    per = str(int(step/size*100))
    load = '*'*int(46*step/size) + ' '*(46-int(46*step/size))

    first = load[:46//2-int(round(len(per)/2,0))]
    sec = load[46//2+int(len(per)-round(len(per)/2,0)):]

    print(f'\r[{first}{per}%%{sec}] {step} of {size} completed ', end='')

def round_r(num:float, r:int = 1) -> float:
    """
    Round right.
    ----
    Returns the num rounded to have at most 'r' 
    significant numbers to the right of the '.'.

    Parameters:
    --
    >>> num:float
    >>> r:int = 1
    
    num: 
      Number.
    r:
      Maximum significant numbers.
    """
    if int(num) != num:
        num = (round(num) 
               if len(str(num).split('.')[0]) > r 
               else f'{{:.{r}g}}'.format(num))

    return num

def not_na(x:any, y:any, f:any = max):
    """
    If not np.nan.
    ----
    It passes to 'x' and 'y' by the function 'f'
    if neither of them are in np.nan, otherwise it returns 
    the value that is not np.nan,
    if both are np.nan, np.nan is returned.

    Parameters:
    --
    >>> x:any
    >>> y:any
    >>> f:any = max
    
    x:
      x value.
    y:
      y value.
    f:
      Function.
    """
    return y if np.isnan(x) else x if np.isnan(y) else f(x, y)

def max_drawdown(values:pd.Series) -> float:
    """
    Maximum drawdown.
    ----
    Returns the maximum drawdown.

    Parameters:
    --
    >>> values:pd.Series
    
    values:
      The ordered data.
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

def candles_plot(ax:Axes, data:pd.DataFrame, 
                 width:float = 1, color_up:str = 'g', 
                 color_down:str = 'r', alpha:str = 1) -> None:
    """
    Candles draw.
    ----
    Plot candles on your 'ax'.

    Parameters:
    --
    >>> ax:Axes
    >>> data:pd.DataFrame
    >>> width:float = 1
    >>> color_up:str = 'g'
    >>> color_down:str = 'r'
    >>> alpha:str = 1
    
    ax:
      Axes where it is drawn.
    data:
      Data to draw.
    width:
      Width of each candle.
    color_up:
      Candle color when price rises.
    color_down:
      Candle color when price goes down.
    aplha:
      Opacity.
    """
    OFFSET = width / 2.

    def draw(row):
        color = color_up if row['Close'] >= row['Open'] else color_down

        line = Line2D(xdata=(row.name, row.name), 
                      ydata=(row['Low'], row['High']), 
                      color=color, linewidth=0.5)
        rect = Rectangle(xy=(row.name-OFFSET, min(row['Open'], row['Close'])), 
                         width=width, 
                         height=abs(row['Close']-row['Open']), 
                         facecolor=color, edgecolor=color)

        rect.set_alpha(alpha); line.set_alpha(alpha)
        ax.add_line(line); ax.add_patch(rect)

    data.apply(draw, axis=1)
    ax.autoscale_view()

def text_fix(text:str, newline_exclude:bool = True) -> str:
    """
    Text fix.
    ----
    Returns 'text' without the common leading spaces on each line.

    Parameters:
    --
    >>> text:str
    >>> newline_exclude:bool = True

    text:
      Text to process.
    newline_exclude:
      Leave it true if you want it to exclude line breaks.
    """

    return ''.join(line.lstrip() + ('\n' if not newline_exclude else '')  
                        for line in text.split('\n'))
