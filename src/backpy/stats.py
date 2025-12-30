"""
Stats module

This module contains functions to calculate different metrics.

Variables:
    logger (Logger): Logger variable.

Functions:
    average_ratio: Based on the take profit and stop loss 
            positions, it calculates an average ratio.
    profit_fact: Calculate the profit factor of the values.
    math_hope: Calculate the mathematical expectation of the values.
    math_hope_relative: Calculate the relative mathematical 
            expectation based on the average_ratio and the profits.
    winnings: Calculate the percentage of positive numbers in the series.
    sharpe_ratio: Calculate the Sharpe ratio using the 
            returns / sqrt(days of the year) / standard deviation of the data.
    sortino_ratio: Calculate the Sortino ratio with a calculation similar to the 
            Sharpe ratio but only with the standard deviation of negative data.
    payoff_ratio: Calculates the payout rate using the absolute 
            mean of positive numbers/mean of negative numbers.
    expectation: Calculate the expectation based on payoff.
    long_exposure: Calculate the percentage of 1 in the given Series.
    var_historical: Calculate the historical var.
    var_parametric: Calculate the parametric var.
    max_drawdown: Function to return the maximum drawdown from the given data.
    get_drawdowns: Calculate the drawdowns from the given.
    perf_tzone_chart: Chart the best and worst hours/minutes of your strategy.
    monte_carlo_chart: Displays graphs with Monte Carlo statistics.
    monte_carlo_bsim: Calculates Monte Carlo simulations.
    correlation: Measure correlation between strategies.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from typing import Literal, Callable
import random as rd
import logging

from . import custom_plt as cpl
from . import _commons as _cm
from . import exception
from . import strategy
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

def average_ratio(trades:pd.DataFrame) -> float:
    """
    Average ratio.

    Based on the profit, it calculates an average ratio.

    Args:
        trades (DataFrame): A dataframe with 'profitPer' column.

    Returns:
        float: Average ratio.
    """

    if 'profitPer' in trades.columns:

        return ((trades['profitPer'][_cm.c_tf(trades['profitPer']) > 0].mean()
                / abs(trades.loc[:, 'profitPer'][_cm.c_tf(trades['profitPer']) < 0]).mean()))
    return 0

def profit_fact(profits:pd.Series) -> float:
    """
    Profit fact.

    Calculate the profit factor of the values.

    Args:
        profits (Series): Returns on each operation.

    Returns:
        float: Profit fact.
    """

    if (not pd.isna(profits).all() 
        and (profits>0).sum() > 0 
        and (profits<=0).sum() > 0):

        return (profits[profits>0].sum()
                / abs(profits[profits<=0].sum()))
    return 0

def math_hope(profits:pd.Series) -> float:
    """
    Math hope.

    Calculate the mathematical expectation of the values.

    Args:
        profits (Series): Returns on each operation.

    Returns:
        float: Math hope.
    """

    return (((profits > 0).sum()/len(profits.index)
            * profits[profits > 0].mean())
                - ((profits < 0).sum()/len(profits.index)
            * -profits[profits < 0].mean()))

def math_hope_relative(trades:pd.DataFrame, profits:pd.Series) -> float:
    """
    Math hope relative.

    Calculate the relative mathematical 
        expectation based on the average_ratio and the profits.

    Args:
        trades (DataFrame): A dataframe with 'profit' column.
        profits (Series): Returns on each operation.

    Returns:
        float: Math hope relative.
    """

    return winnings(profits)*float(average_ratio(trades))-(1-winnings(profits))

def winnings(profits:pd.Series) -> float:
    """
    Winnings percentage.

    Calculate the percentage of positive numbers in the series.

    Args:
        profits (Series): Returns on each operation.

    Returns:
        float: Winnings percentage.
    """

    if (not ((profits>0).sum() == 0 
        or profits.count() == 0)):

        return (profits>0).sum()/profits.count()
    return 0


def sharpe_ratio(ann_av:float|np.floating, year_days:int, diary_per:pd.Series) -> float:
    """
    Sharpe ratio.

    Calculate the Sharpe ratio using the 
        returns / sqrt(days of the year) / standard deviation of the data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float|floating): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (Series): Daily return.

    Returns:
        float: Sharpe ratio.
    """
    std_dev = np.std(diary_per.dropna(), ddof=1)
    if std_dev < 1e-2: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def sortino_ratio(ann_av:float|np.floating, year_days:int, diary_per:pd.Series) -> float:
    """
    Sortino ratio.

    Calculate the Sortino ratio with a calculation similar to the 
        Sharpe ratio but only with the standard deviation of negative data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float|floating): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (Series): Daily return.

    Returns:
        float: Sortino ratio.
    """

    std_dev = np.std(diary_per.loc[diary_per < 0].dropna(), ddof=1)
    if std_dev < 1e-2: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def payoff_ratio(profits:pd.Series) -> float:
    """
    Payoff ratio.

    Calculates the payout rate using the absolute 
        mean of positive numbers/mean of negative numbers.

    Args:
        profits (Series): Returns on each operation..

    Returns:
        float: Payoff ratio.
    """

    return (profits.loc[profits > 0].dropna().mean() 
            / abs(profits.loc[profits < 0].dropna().mean()))

def expectation(profits:pd.Series) -> float:
    """
    Expectation.

    Calculate the expectation based on payoff.

    Args:
        profits (Series): Returns on each operation.

    Returns:
        float: Expectation.
    """

    return ((winnings(profits)*payoff_ratio(profits)) 
            - (1-winnings(profits)))

def long_exposure(types:pd.Series) -> float:
    """
    Long exposure.

    Calculate the percentage of 1 in the 'types'.

    Args:
        types (Series): Type of each operation, 1 for long, 0 for short.

    Returns:
        float: Percentages of longs.
    """

    return (types==1).sum()/types.count()

def var_historical(data:list | pd.Series | np.ndarray, 
                   confidence_level:int = 95) -> float:
    """
    Var historical.

    Calculate the historical var.

    Args:
        data (list | pd.Series | np.ndarray): 
            List of data which will calculate the var.
        confidence_level (int, optional): Percentile.
    
    Returns:
        float: The historical var.
    """

    return np.sort(data)[int((100 - confidence_level) / 100 * len(data))]

def var_parametric(data:list | pd.Series | np.ndarray, 
                   z_alpha:float = -1.645) -> float:
    """
    Var parametric.

    Calculate the parametric var.

    Args:
        data (list | pd.Series | np.ndarray): 
            List of data which will calculate the var.
        z_alpha (float, optional): Critical value of the standard normal 
            distribution corresponding to the confidence level.

    Returns:
        float: The parametric var.
    """

    return np.average(data)-z_alpha*np.std(data, ddof=1)

def max_drawdown(values:pd.Series) -> float:
    """
    Maximum drawdown.

    Calculate the maximum drawdown of `values`.

    Args:
        values (Series): The ordered data to calculate the maximum drawdown.

    Returns:
        float: The maximum drawdown from the given data.
    """

    if values.empty: return 0
    max_drdwn, max_val = 0, values.iloc[0]

    def calc(x):
        nonlocal max_drdwn, max_val

        if x > max_val: max_val = x
        else: 
            drdwn = (max_val - x) / max_val
            if drdwn > max_drdwn:
                max_drdwn = drdwn
    values.apply(calc)

    return max_drdwn

def get_drawdowns(
        values:list | pd.Series | np.ndarray
    ) -> Literal[0] | list | pd.Series | np.ndarray:
    """
    Get drawdowns.

    Calculate the drawdowns of `values`.

    Args:
        values (list | pd.Series | np.ndarray): 
            The ordered data to calculate the drawdowns.

    Returns:
        Literal[0] | list | pd.Series | np.ndarray: The drawdowns from the given data.
    """

    if len(values) == 0:
        return 0

    max_values = np.maximum.accumulate(values)
    drawdowns = (values - max_values) / max_values

    return drawdowns

def perf_tzone_chart(names:list[str|int|None]|str|int|None = None,
                     view:str = 'p/d', col:str|None = 'profitPer', 
                     panel:str = 'new', style:str|None = 'last', 
                     style_c:dict|None = None, block:bool = True) -> None:
    """
    Performance time zones chart

    See how your strategy performs based on the opening or closing time of each trade.

    Available Graphics:
    - 'p' = Sum of profit per hour depending on the closing date.
    - 'd' = Sum of profit per hour depending on the opening date.
    - 'mp' = Sum of profit per minute depending on the closing date.
    - 'md' = Sum of profit per minute depending on the opening date.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Args:
        names (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        view (str, optional): Specifies which graphics to display. 
            Default is 'p/d'. Maximum 8.
        col (str|None, optional): Column to display statistics, 
            only 'profit' and 'profitPer' are supported, 
            None uses 'profitPer'.
        panel (str, optional): To create a new window or add a panel, 
            only 'new' or 'add' are possible.
        style (str | None, optional): Color style. 
            If you leave it as 'last' the last one will be used.
        style_c (dict | None, optional): Customize the defined style by 
            modifying the dictionary. To know what to modify, 
            read the docstring of 'def_style'.
        block (bool, optional): If True, pauses script execution until all figure 
            windows are closed. If False, the script continues running after 
            displaying the figures. Default is True.
    """

    # Exceptions.
    panel = panel.lower()
    valid_style = {'random', 'last'} | set(_cm.__plt_styles.keys())

    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif panel not in ('new', 'add'):
        raise exception.StatsError(
            f"'{panel}' Not a valid option for: 'panel'.")
    elif (not style is None and not (style:=style.lower()) in valid_style):
        raise exception.StatsError(f"'{style}' Not a style.")
    col = col or 'profitPer'

    trades = _cm.__get_trades(names=names)
    name = list(names)[0] if isinstance(names, (tuple, set, list)) else names
    trades_data = _cm.__get_strategy(name=name)

    if trades.empty:
        raise exception.StatsError('Trades not loaded.')

    hour = lambda index: ((index % trades_data['d_width_day']) 
                          / trades_data['d_width_day'] * 24).astype(int)
    minute = lambda index: ((index % (trades_data['d_width_day']/60)) 
                          / (trades_data['d_width_day']/60) * 60).astype(int)

    if style == 'last':
        style = _cm.plt_style
    if style is None:
        style = list(_cm.__plt_styles.keys())[0]
    elif style == 'random':
        style = rd.choice(list(_cm.__plt_styles.keys()))

    plt_colors = _cm.__plt_styles[style]
    _cm.plt_style = style

    if isinstance(style_c, dict):
        plt_colors.update(style_c)

    gdir = plt_colors.get('gdir', False)
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    fig = plt.figure(figsize=(16,8))
    fig.subplots_adjust(left=0, right=1, top=1, 
                        bottom=0, wspace=0, hspace=0)

    graphics = ['p', 'd', 'mp', 'md']
    axes, v_view = cpl.ax_view(view=view, graphics=graphics)

    def time_graph(legend:str, time_col:str, func:Callable) -> None:
        """
        Time graph

        Bar chart with the specific time.

        Args:
            legend (str): Graph name.
            time_col (str): Column for statistics, 'positionDate' or 'date'.
            func (Callable): Function to obtain the time of each trade.
        """

        trades['time_close'] = func(trades[time_col].dropna())
        hourly_sums:pd.Series[float] = trades.groupby('time_close')[col].sum()
        colors = np.where(hourly_sums>0, market_colors.get('u'), 
                                            market_colors.get('d'))

        ax.bar(hourly_sums.index.values+1, hourly_sums.to_numpy(), color=colors)
        ax.legend([legend], loc='upper left')

    for i,v in enumerate(v_view):
        ax = axes[i]
    
        cpl.custom_ax(ax, plt_colors['bg'], edge=gdir)
        ax.tick_params('x', which='both', bottom=False, 
                        top=False, labelbottom=False)
        ax.tick_params('y', which='both', left=False, 
                        right=False, labelleft=False)

        ax.yaxis.set_major_formatter(lambda y, _: str(y.real))
        ax.xaxis.set_major_formatter(lambda x, _: str(x.real))

        match v:
            case 'p':
                time_graph('Position close hours.', 'positionDate', hour)
            case 'd':
                time_graph('Position opening hours.', 'date', hour)
            case 'mp':
                time_graph('Position close minutes.', 'positionDate', minute)
            case 'md':
                time_graph('Position opening minutes.', 'date', minute)
            case _: pass

    cpl.add_window(
        fig=fig,
        title=f'Performance in time - {style}',
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total'
    )

def monte_carlo_chart(data:list[pd.DataFrame], view:str = 's/d',
                      n_trades:int|None = None, col:str|None = 'profitPer',
                      panel:str = 'new', style:str|None = 'last', 
                      style_c:dict|None = None, block:bool = True) -> None:
    """
    Monte Carlo chart

    Takes data from a Monte Carlo simulation 
    and generates graphs with statistics.

    Available Graphics:
    - 's' = Simulation chart.
    - 'd' = Distribution of results with this you can see 
        what percentage of simulations win.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Args:
        data (list[pd.DataFrame]): Data extracted from a Monte Carlo simulation.
            You can extract data from 'monte_carlo_bsim' function.
        view (str, optional): Specifies which graphics to display. 
            Default is 'd/p/b'. Maximum 8.
        n_trades (int|None, optional): For graph 'd' how many simulations 
            will be shown.
        col (str|None, optional): Column to display statistics, 
            only 'profit' and 'profitPer' are supported, 
            None uses 'profitPer' and calculates equity curve.
        panel (str, optional): To create a new window or add a panel, 
            only 'new' or 'add' are possible.
        style (str | None, optional): Color style. 
            If you leave it as 'last' the last one will be used.
        style_c (dict | None, optional): Customize the defined style by 
            modifying the dictionary. To know what to modify, 
            read the docstring of 'def_style'.
        block (bool, optional): If True, pauses script execution until all figure 
            windows are closed. If False, the script continues running after 
            displaying the figures. Default is True.
    """
    # Exceptions.
    panel = panel.lower()
    valid_style = {'random', 'last'} | set(_cm.__plt_styles.keys())

    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif panel not in ('new', 'add'):
        raise exception.StatsError(
            f"'{panel}' Not a valid option for: 'panel'.")
    elif n_trades and n_trades <= 1 and n_trades > len(data):
        raise exception.StatsError(utils.text_fix("""
                        'n_trades' can only be greater than 1 and 
                        less than or equal to the length of 'data'.
                        """, newline_exclude=True))
    elif (not style is None and not (style:=style.lower()) in valid_style):
        raise exception.StatsError(f"'{style}' Not a style.")

    if style == 'last':
        style = _cm.plt_style
    if style is None:
        style = list(_cm.__plt_styles.keys())[0]
    elif style == 'random':
        style = rd.choice(list(_cm.__plt_styles.keys()))

    plt_colors = _cm.__plt_styles[style]
    _cm.plt_style = style

    if isinstance(style_c, dict):
        plt_colors.update(style_c)

    gdir = plt_colors.get('gdir', False)
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    fig = plt.figure(figsize=(16,8))
    fig.subplots_adjust(left=0, right=1, top=1, 
                        bottom=0, wspace=0, hspace=0)

    graphics = ['s','d']
    axes, v_view = cpl.ax_view(view=view, graphics=graphics)

    for i,v in enumerate(v_view):
        ax = axes[i]
    
        cpl.custom_ax(ax, plt_colors['bg'], edge=gdir)
        ax.tick_params('x', which='both', bottom=False, 
                        top=False, labelbottom=False)
        ax.tick_params('y', which='both', left=False, 
                        right=False, labelleft=False)

        ax.yaxis.set_major_formatter(lambda y, _: str(y.real))
        ax.xaxis.set_major_formatter(lambda x, _: str(x.real))

        match v:
            case 's':
                for i in range(n_trades if n_trades else len(data)):
                    curve = (data[i][col].cumsum().dropna() 
                             if isinstance(col, str) else 
                             (1 + data[i]['profitPer'] / 100).cumprod().dropna()-1 )
                    ax.plot(range(0, len(curve)), curve, alpha=0.5)

                ax.legend(['Simulations.'], loc='upper left')
                ax.set_xlim(-1, len(data[0].index))
            case 'd':
                data_last = lambda df: (df[col].cumsum().dropna().iloc[-1] 
                                    if isinstance(col, str) else 
                                    (np.cumprod(1 + df[i]['profitPer'] / 100).dropna()-1).iloc[-1])
                last_result = np.array([data_last(df) for df in data])

                parts = np.array_split(np.sort(last_result), 100)
                means:list[float] = [np.mean(part) for part in parts if len(part) > 0]

                color_u = lambda x: utils.mult_color(
                    color=market_colors['u'], multiplier=x)
                color_d = lambda x: utils.mult_color(
                    color=market_colors['d'], multiplier=x)
                colors = np.array([
                    color_u(val/np.max(means)+1) if val >= 0 else color_d(1-val/np.min(means))
                    for val in means if val != 0
                ])

                ax.bar(list(range(len(means))), means, # type: ignore[arg-type]
                       width=0.8, color=colors)
                ax.legend(['Distribution.'], loc='upper left')
            case _: pass

    cpl.add_window(
        fig=fig,
        title=f'Monte Carlo simulation - {style}',
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total'
    )

def monte_carlo_bsim(names:list[str|int|None]|str|int|None = None, 
                    n_trades:int|None = None, n_sim:int|None = 10000, 
                    percentiles:list[int|float] = [1,5,10,24,50,75], 
                    col:str|None = 'profitPer', prnt:bool = True 
                    ) -> tuple[list[pd.DataFrame], str]:
    """
    Monte Carlo bootstrap simulation

    Calculate a Monte Carlo bootstrap simulation and gives statistics.

    For documentation of statistics, read the 'stats_trades' docstring.

    Args:
        names (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        n_trades (int|None, optional): Number of trades per simulation, 
            None = length of loaded trades.
        n_sim (int|None, optional): Number of simulations.
        percentiles (list[int|float], optional): Percentiles for statistics.
        col (str|None, optional): Column to do the simulation, 
            only 'profit' and 'profitPer' are supported, 
            None uses 'profitPer' and calculates equity curve.
        prnt (bool, optional): If True, the statistics are 
            printed on the console.

    Return:
        tuple[list[DataFrame],str]: 
            Tuple with: list with all simulations and statistics test.
    """

    # Exceptions.
    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif n_trades and n_trades <= 1:
        raise exception.StatsError(
            "'n_trades' can only be greater than 1.")
    elif n_sim and n_sim <= 0:
        raise exception.StatsError(
            "'n_trades' can only be greater than 0.")

    trades = _cm.__get_trades(names=names)
    name = list(names)[0] if isinstance(names, (tuple,set,list)) else names
    trades_data = _cm.__get_strategy(name=name)
    sim = []

    if trades.empty:
        raise exception.StatsError('Trades not loaded.')

    stats = {
        'profit_fact':[],
        'max_drawdown':[],
        'avg_drawdown':[],
        'max_drawdown$':[],
        'avg_drawdown$':[],
        'expectation':[],
        'winrate':[],
    }

    for i in range(n_sim or 10000):
        trades_s = trades.sample(
            n=n_trades or len(trades), replace=True)

        trades_calc = trades_s
        trades_calc['multiplier'] = 1 + trades_calc['profitPer'] / 100

        stats['profit_fact'].append(profit_fact(trades.loc[:, 'profit']))
        stats['expectation'].append(expectation(trades_s.loc[:, 'profitPer']))
        stats['max_drawdown'].append(
            max_drawdown(pd.Series(np.cumprod(trades_s['multiplier'].dropna()))))
        stats['avg_drawdown'].append(
            np.mean(get_drawdowns(np.cumprod(trades_s['multiplier'].dropna()))))
        stats['max_drawdown$'].append(
            max_drawdown(trades['profit'].cumsum().dropna()
                         +trades_data['init_funds']))
        stats['avg_drawdown$'].append(
            np.mean(get_drawdowns(trades['profit'].cumsum().dropna()
                                  +trades_data['init_funds'])))
        stats['winrate'].append(winnings(trades.loc[:, 'profitPer'])*100)

        sim.append(trades_s)

    data_last = lambda df: (df[col].cumsum().dropna().iloc[-1] 
                        if isinstance(col, str) else 
                        (np.cumprod(1 + df['profitPer'] / 100).dropna()-1).iloc[-1])
    last_result = np.array([data_last(df) for df in sim])
    percentiles_r = np.percentile(last_result, percentiles)

    percentiles_t = {
        f'Percentile {percentiles[i]}':[
            round(v, 2), _cm.__COLORS['GREEN'] if v > 0 else _cm.__COLORS['RED']
        ] for i,v in enumerate(percentiles_r)}

    text = {
        'Average return':[(md_rtrn:=round(np.average(last_result), 1)),
                         _cm.__COLORS['GREEN'] if md_rtrn > 0 else _cm.__COLORS['RED']],
        'Profit fact avg':[(prft_fact:=utils.round_r(np.average(stats['profit_fact']), 3)),
                              _cm.__COLORS['GREEN'] if prft_fact > 1 else _cm.__COLORS['RED']],
        'Max drawdown avg':[str(round(np.average(stats['max_drawdown'])*100, 1)) + '%'],
        'Average drawdown avg':[str(-round(np.average(stats['avg_drawdown'])*100, 1)) + '%'],
        'Max drawdown$ avg':[str(round(np.average(stats['max_drawdown$'])*100,1)) + '%'],
        'Average drawdown$ avg':[str(-round(np.average(stats['avg_drawdown$'])*100, 1)) + '%'],
        'Expectation avg':[utils.round_r(np.average(stats['expectation']))],
        'Winnings avg':[str(round(np.average(stats['winrate']), 1)) + '%',
                           _cm.__COLORS['GREEN']],
        f'\n{_cm.__COLORS['CYAN']}Percentiles{_cm.__COLORS['RESET']}':['']
    }
    text.update(percentiles_t)

    text = utils.statistics_format(text, f"---Statistics of Monte Carlo---")

    text = text if _cm.dots else text.replace('.', ',')
    if prnt:print(text) 

    return (sim, text)

def correlation(names:list[str|int|None], col:str|None = None, 
                method:str|None = None) -> pd.DataFrame:
    """
    Correlation

    Measures correlation with DataFrame.corr.

    Args:
        names (list[str|int|None]): Backtest names which measure correlation.
        col (str|None, optional): Column used to measure correlation, 
            only 'profit' and 'profitPer' are supported, None = 'profitPer'.
        method (str|None, optional): Correlation method: 'pearson', 
            'kendall', 'spearman'. None = 'pearson'.

    Returns:
        DataFrame: Correlation.
    """

    # Exceptions.
    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif method and method.lower() not in ('pearson', 'kendall', 'spearman'):
        raise exception.StatsError(
            "'method' only 'pearson', 'kendall', 'spearman' or None is supported.")

    trades = _cm.__get_dtrades(names=names)

    daily_profit = {
        k: v.groupby('positionDate')[col or 'profitPer'].sum().cumsum()
        for k, v in trades.items()
    }

    returns = pd.concat(
        daily_profit, 
        axis=1, 
        join='outer').sort_index().ffill().pct_change().dropna()

    return returns.corr(method=method.lower() if method else 'pearson')
