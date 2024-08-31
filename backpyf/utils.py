"""
Utils.
----
Different useful functions for the operation of main code.

Functions:
---
>>> load_bar
>>> round_r
>>> not_na
>>> correct_index
>>> calc_width
>>> max_drawdown
>>> plot_candles
>>> text_fix
>>> plot_position
"""

from matplotlib.patches import Rectangle 
from matplotlib.axes._axes import Axes

import matplotlib as mpl
import pandas as pd
import numpy as np

from . import utils
from . import main

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
      - Number of steps.
    
    step:
      - step.
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
      - Number.
    
    r:
      - Maximum significant numbers.
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
      - x value.
    
    y:
      - y value.
    
    f:
      - Function.
    """
    return y if np.isnan(x) else x if np.isnan(y) else f(x, y)

def correct_index(index:pd.Index) -> np.ndarray:
    """
    Correct index.
    ----
    Returns the correct index.
     Correct the index by changing it to float.

    Parameter:
    --
    >>> index:pd.Index

    index:
      - data.index.
    """
    if not all(isinstance(ix, float) for ix in index):
        index = mpl.dates.date2num(index)
        print(utils.text_fix("""
              The 'index' has been automatically corrected. 
              To resolve this, use a valid index.
              """)) if main.alert else None
    
    return index

def calc_width(index:pd.Index, alert:bool = False) -> float:
    """
    Calc width.
    ----
    Returns the width of 'index' if not already calculated.

    Parameters:
    --
    >>> index:pd.Index
    >>> alert:bool = False

    index:
      - data.index.
    alert:
      - If the alert is printed.
    """
    if isinstance(main.__data_width, float) and main.__data_width > 0: 
        return main.__data_width
        
    print(utils.text_fix("""
          The 'data_width' has been automatically corrected. 
          To resolve this, use a valid width.
          """)) if main.alert and alert else None
    
    return np.median(np.diff(index))

def max_drawdown(values:pd.Series) -> float:
    """
    Maximum drawdown.
    ----
    Returns the maximum drawdown.

    Parameter:
    --
    >>> values:pd.Series
    
    values:
      - The ordered data.
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

    return max_drdwn

def plot_candles(ax:Axes, data:pd.DataFrame, 
                 width:float = 1, color_up:str = 'g', 
                 color_down:str = 'r', color_n:str = 'k',
                 alpha:float = 1) -> None:
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
    >>> color_n:str = 'b'
    >>> alpha:float = 1
    
    ax:
      - Axes where it is drawn.
    
    data:
      - Data to draw.
    
    width:
      - Width of each candle.
    
    color_up:
      - Candle color when price rises.
    
    color_down:
      - Candle color when price goes down.

    color_n:
      - Candle color when the price does not move.
    
    aplha:
      - Opacity.
    """
    color = data.apply(
               lambda x: (color_n if x['Close'] == x['Open'] else
                   color_up if x['Close'] >= x['Open'] else color_down), 
               axis=1)

    # Drawing vertical lines.
    ax.vlines(data.index, data['Low'], data['High'], color=color, alpha=alpha)

    # Bar drawing.
    hgh = abs(data['Close']-data['Open'])
    ax.bar(data.index, hgh.where(hgh > 0, data['Close'].diff().mean()), width,
           bottom=data.apply(lambda x: min(x['Open'], x['Close']), axis=1), 
           color=color, alpha=alpha)

    ax.set_ylim(data['Low'].min()*0.98, data['High'].max()*1.02)

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
      - Text to process.
    
    newline_exclude:
      - Leave it true if you want it to exclude line breaks.
    """

    return ''.join(line.lstrip() + ('\n' if not newline_exclude else '')  
                        for line in text.split('\n'))

def plot_position(trades:pd.DataFrame, ax:Axes, 
                  color_take:str = 'green', color_stop:str = 'red', 
                  color_close:str = 'gold', all:bool = True,
                  alpha:float = 1, alpha_arrow:float = 1, 
                  operation_route:bool = True, 
                  width_exit:any = lambda x: 9) -> None:
    """
    Position draw.
    ----
    Plot positions on your 'ax'.

    Parameters:
    --
    >>> trades:pd.DataFrame
    >>> ax:Axes
    >>> color_take:str = 'green'
    >>> color_stop:str = 'red'
    >>> color_close:str = 'gold'
    >>> all:bool = True
    >>> alpha:float = 1
    >>> alpha_arrow:float = 1
    >>> operation_route:bool = True
    >>> width_exit:any = lambda x: 9

    trades:
      - Trades data to draw.
    
    ax:
      - Axes where it is drawn.
    
    color_take:
      - Position color when the position is positive.
    
    color_stop:
      - Position color when the position is negative.
    
    color_close:
      - Color of close marker.

    all:
      - If 'True', everything will be drawn.
      - If 'False', only points will be drawn.
    
    aplha:
      - Opacity.
    
    alpha_arrow:
      - Opacity of arrow, type marker, and close marker.

    operation_route:
      - True or false to trace the route of the operation.
    
    width_exit:
      - How many time points does the position 
       extend forward if it is not closed.
      - It has to be a function that takes 'trade'.
    
    Info:
    --
    The arrow and 'x' marker indicates where the position was closed.

    The 'V' and '^' markers indicate the direction of the position.
    
    The 'color_take' indicates the place where the take profit is placed and 
     the 'color_stop' will indicate the place where the stop loss is placed.
    If there is no 'take profit', 
     its figure will not be drawn; the same applies to the 'stop loss'.
    """

    def draw(row):
        # Drawing of the 'TakeProfit' shape.
        if not np.isnan(row['TakeProfit']) and all: 
            take = Rectangle(xy=(row['Date'], row['Close']), 
                          width=(width_exit(row) 
                                if ('PositionDate' not in row.index or 
                                    np.isnan(row['PositionDate'])) 
                                else row['PositionDate']-row['Date']), 
                          height=row['TakeProfit']-row['Close'], 
                          facecolor=color_take, edgecolor=color_take)
            
            take.set_alpha(alpha)
            ax.add_patch(take)

        # Drawing of the 'StopLoss' shape.
        if not np.isnan(row['StopLoss']) and all: 
            stop = Rectangle(xy=(row['Date'], row['Close']), 
                          width=(width_exit(row) 
                                if ('PositionDate' not in row.index or 
                                    np.isnan(row['PositionDate'])) 
                                else row['PositionDate']-row['Date']), 
                          height=row['StopLoss']-row['Close'], 
                          facecolor=color_stop, edgecolor=color_stop)
            
            stop.set_alpha(alpha)
            ax.add_patch(stop)

        # Draw route of the operation.
        if 'PositionDate' in row.index and not np.isnan(row['PositionDate']):
            if operation_route and all:
                cl = ('green' if (row['Close'] < row['PositionClose'] and 
                                  row['Type'] == 1) or 
                                  (row['Close'] > row['PositionClose'] and 
                                  row['Type'] == 0) else 'red')

                route  = Rectangle(xy=(row['Date'], row['Close']), 
                                width=row['PositionDate']-row['Date'], 
                                height=row['PositionClose']-row['Close'], 
                                facecolor=cl, edgecolor=cl)
            
                route.set_alpha(alpha)
                ax.add_patch(route)
      
            # Arrow drawing.
            ax.arrow(row['Date'], row['Close'], 
                    row['PositionDate']-row['Date'], 
                    row['PositionClose']-row['Close'], 
                    linestyle='-', color='grey', alpha=alpha_arrow, 
                    width=abs(row['PositionClose']-row['Close'])*0.00001)

    trades.apply(draw, axis=1)

    # Drawing of the closing marker of the operation.
    if ('PositionDate' in trades.columns and 
        'PositionClose' in trades.columns):
        ax.scatter(trades['PositionDate'], trades['PositionClose'], 
                  c=color_close, s=30, marker='x', alpha=alpha_arrow)

    # Drawing of the position type marker.
    ax.scatter(trades['Date'], 
               trades.apply(lambda x: x['Low'] - (x['High'] - x['Low']) / 2 
                            if x['Type'] == 1 else None, axis=1), 
               c=color_take, s=30, marker='^', alpha=alpha_arrow)
    
    ax.scatter(trades['Date'], 
               trades.apply(lambda x: x['High'] + (x['High'] - x['Low']) / 2 
                            if x['Type'] != 1 else None, axis=1),
               c=color_stop, s=30, marker='v', alpha=alpha_arrow)
