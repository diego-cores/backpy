"""
Utils Module.

Contains various utility functions for the operation of the main code.

Functions:
    load_bar: Function to print a loading bar.
    round_r: Function to round a number to a specified number of significant 
        digits to the right of the decimal point.
    not_na: Function to apply a specified function to two values if neither is 
        `np.nan`, or return the non-`np.nan` value, or `np.nan` if both are 
        `np.nan`.
    correct_index: Function to correct index by converting it to float.
    calc_width: Function to calulate the width of 'index' 
        if it has not been calculated already.
    max_drawdown: Function to return the maximum drawdown from the given data.
    text_fix: Function to fix or adjust text.
    plot_candles: Function to plot candles on a given `Axes`.
    plot_position: Function to plot a trading position.
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

    Prints a loading bar.

    Args:
        size (int): Number of steps in the loading bar.
        step (int): Current step in the loading process.
    """

    per = str(int(step/size*100))
    load = '*'*int(46*step/size) + ' '*(46-int(46*step/size))

    first = load[:46//2-int(round(len(per)/2,0))]
    sec = load[46//2+int(len(per)-round(len(per)/2,0)):]

    print(f'\r[{first}{per}%%{sec}] {step} of {size} completed ', end='')

def round_r(num:float, r:int = 1) -> float:
    """
    Round right.

    Rounds `num` to have at most `r` significant digits to the right of the 
    decimal point. If `num` is `np.nan` or evaluates to `None`, it returns 0.

    Args:
        num (float): The number to round.
        r (int, optional): Maximum number of significant digits to the right of 
            the decimal point. Defaults to 1.

    Returns:
        float: The rounded number, or 0 if `num` is `np.nan` or evaluates to 
            `None`.
    """

    if np.isnan(num) or not bool(num):
        return 0

    if int(num) != num:
        num = (round(num) 
               if len(str(num).split('.')[0]) > r 
               else f'{{:.{r}g}}'.format(num))

    return num

def not_na(x:any, y:any, f:callable = max):
    """
    If not np.nan.

    Applies function `f` to `x` and `y` if neither of them are `np.nan`. If one 
    of them is `np.nan`, returns the value that is not `np.nan`. If both are 
    `np.nan`, `np.nan` is returned.

    Args:
        x (any): The first value.
        y (any): The second value.
        f (callable, optional): Function to apply to `x` and `y` if neither is 
            `np.nan`. Defaults to `max`.

    Returns:
        any: The result of applying `f` to `x` and `y`, or the non-`np.nan` value, 
            or `np.nan` if both are `np.nan`.
    """

    return y if np.isnan(x) else x if np.isnan(y) else f(x, y)

def correct_index(index:pd.Index) -> np.ndarray:
    """
    Correct index.

    Correct `index` by converting it to float

    Args:
        index (pd.Index): The `index` of the data to be corrected.

    Returns:
        np.ndarray: The corrected `index`.
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
    
    Calculate the width of `index` if it has not been calculated already.

    Args:
        index (pd.Index): The index of the data.
        alert (bool, optional): If `True`, an alert will be printed. Defaults to 
            False.

    Returns:
        float: The width of `index`.
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

    Calculate the maximum drawdown of `values`.

    Args:
        values (pd.Series): The ordered data to calculate the maximum drawdown.

    Returns:
        float: The maximum drawdown from the given data.
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

def plot_candles(ax:Axes, data:pd.DataFrame, 
                 width:float = 1, color_up:str = 'g', 
                 color_down:str = 'r', color_n:str = 'k',
                 alpha:float = 1) -> None:
    """
    Candles draw.

    Plots candles on the provided `ax`.

    Args:
        ax (Axes): The `Axes` object where the candles will be drawn.
        data (pd.DataFrame): Data to draw the candles.
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

def plot_position(trades:pd.DataFrame, ax:Axes, 
                  color_take:str = 'green', color_stop:str = 'red', 
                  color_close:str = 'gold', all:bool = True,
                  alpha:float = 1, alpha_arrow:float = 1, 
                  operation_route:bool = True, 
                  width_exit:any = lambda x: 9) -> None:
    """
    Position Draw.

    Plots positions on your `ax`.

    Args:
        trades (pd.DataFrame): Trades data to draw.
        ax (Axes): Axes where it is drawn.
        color_take (str, optional): Color for positive positions. Default is 'green'.
        color_stop (str, optional): Color for negative positions. Default is 'red'.
        color_close (str, optional): Color of the close marker. Default is 'gold'.
        all (bool, optional): If True, draws all elements. If False, only draws points. Default is True.
        alpha (float, optional): Opacity of the elements. Default is 1.
        alpha_arrow (float, optional): Opacity of arrow, type marker, and close marker. Default is 1.
        operation_route (bool, optional): If True, traces the route of the operation. Default is True.
        width_exit (any, optional): Function that specifies how many time points the position 
            extends forward if not closed. Default is a lambda function with a width of 9.

    Info:
        The arrow and 'x' markers indicate where the position was closed.
        The 'V' and '^' markers indicate the direction of the position.
        The 'color_take' indicates where the take profit is placed and the 'color_stop'
        indicates where the stop loss is placed. If there is no take profit, its marker 
        will not be drawn; the same applies to the stop loss.
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
