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
    earnings_intime: Statistics of earnings each 'x' amount of days.
    stats_icon: Shows statistics related to the financial icon.
    stats_trades: Statistics of the trades.
    trades_op_years: Return the number of years operated.
    trades_group_duration: Return the duration of trades in days.
    trades_group_year: Returns 'Series' by grouping each trade by year.
    trades_group_day: Returns 'Series' by grouping each trade by days.
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

        return ((trades['profitPer'][trades['profitPer'] > 0].mean()
                / abs(trades.loc[:, 'profitPer'][trades['profitPer'] < 0]).mean()))
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

def stats_icon(prnt:bool = True, data:pd.DataFrame | None = None, 
               data_icon:str | None = None, 
               data_interval:str | None = None) -> str | None:
    """
    Icon Statistics.

    Displays statistics of the uploaded data.

    Args:
        prnt (bool, optional): If True, prints the statistics. If False, returns
            the statistics as a string. Default is True.
        data (DataFrame | None, optional): The data with which the statistics 
            are calculated, if left to None the loaded data will be used.
            The DataFrame must contain the following columns: 
            ('close', 'open', 'high', 'low', 'volume').
        data_icon (str | None, optional): Icon shown in the statistics, 
            if you leave it at None the loaded data will be the one used.
        data_interval (str | None, optional): Interval shown in the statistics, 
            if you leave it at None the loaded data will be the one used.

    Returns:
        str|None: Statistics.
    """

    data_interval = _cm.__data_interval if data_interval is None else data_interval
    data_icon = _cm.__data_icon if data_icon is None else data_icon
    data = _cm.__data if data is None else data

    # Exceptions.
    if data is None: 
        raise exception.StatsError('Data not loaded.')
    elif not data_icon is None and type(data_icon) != str: 
        raise exception.StatsError('Icon bad type.')
    elif not data_interval is None and type(data_interval) != str: 
        raise exception.StatsError('Interval bad type.')

    if isinstance(data.index[0], pd.Timestamp):
        s_date = ".".join(str(val) for val in 
                        [data.index[0].day, data.index[0].month, 
                        data.index[0].year])

        idx_last = data.index[-1]
        e_date = ".".join(str(val) for val in 
                        [idx_last.day, idx_last.month, 
                        idx_last.year]
                        ) if isinstance(idx_last, pd.Timestamp) else ""

        r_date = f"{s_date}~{e_date}"
    else: r_date = ""

    text = utils.statistics_format({
        'Last price':[utils.round_r(data['close'].iloc[-1],2),
                      _cm.__COLORS['BOLD']],
        'Maximum price':[utils.round_r(data['high'].max(),2),
                         _cm.__COLORS['GREEN']],
        'Minimum price':[utils.round_r(data['low'].min(),2),
                         _cm.__COLORS['RED']],
        'Maximum volume':[utils.round_r(data['volume'].max(), 2),
                          _cm.__COLORS['CYAN']],
        'Sample size':[len(data.index)],
        'Standard deviation':[utils.round_r(
            np.std(data['close'].dropna(), ddof=1),2)],
        'Average price':[utils.round_r(data.loc[:, 'close'].mean(),2),
                         _cm.__COLORS['YELLOW']],
        'Average volume':[utils.round_r(data.loc[:, 'volume'].mean(),2),
                          _cm.__COLORS['YELLOW']],
        f"'{data_icon}'":[f'{r_date} ~ {data_interval}',
                          _cm.__COLORS['CYAN']],
    }, f"---Statistics of '{data_icon}'---")

    text = text if _cm.dots else text.replace('.', ',')
    if prnt:print(text) 
    else: return text

def stats_trades(data:bool = False, name:list[str|int|None]|str|int|None = None, 
                 prnt:bool = True) -> str | None:
    """
    Trades Statistics.

    Statistics of the results.

    Args:
        data (bool, optional): If True, `stats_icon` is also returned.
        name (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        prnt (bool, optional): If True, prints the statistics. If False, returns 
            the statistics as a string. Default is True.

    Info:
        - Trades: The number of operations performed.
        - Op years: Years operated from the first to the last.
        - Return: The total equity earned.
        - Profit: The total amount earned.
        - Gross earnings: Only the profits.
        - Gross losses: Only the losses.
        - Max return: The historical maximum of returns.
        - Return from max: Returns from the all-time high.
        - Days from max: Days from the all-time return high.
        - Return ann: The annualized return.
        - Profit ann: The annualized profit.
        - Return ann vol: The annualized daily standard deviation of return.
        - Profit ann vol: The annualized daily standard deviation of profit.
        - Average ratio: The average ratio.
        - Average return: The average percentage earned.
        - Average profit: The average profit earned.
        - Profit fact: The profit factor is calculated by dividing 
                total profits by total losses.
        - Return diary std: The standard deviation of daily return, 
                which indicates the variability in performance.
        - Profit diary std: The standard deviation of daily profit, 
                which indicates the variability in performance.
        - Math hope: The mathematical expectation (or expected value) of returns, 
                calculated as (Win rate * Average win) - (Loss rate * Average loss).
        - Math hope r: The mathematical expectation, 
                calculated as (Win rate * Average ratio) - (Loss rate * 1).
        - Historical var: The Value at Risk (VaR) estimated using historical data, 
                calculated as the profit at the (100 - confidence level) percentile.
        - Parametric var: The Value at Risk (VaR) calculated assuming a normal distribution, 
                defined as the mean profit minus z-alpha times the standard deviation.
        - Sharpe ratio: The risk-adjusted return, calculated as the 
                annualized return divided by the standard deviation of return.
        - Sharpe ratio$: The risk-adjusted return, calculated as the annualized 
                profit divided by the standard deviation of profits.
        - Sortino ratio: The risk-adjusted return, calculated as the annualized 
                return divided by the standard deviation of negative return.
        - Sortino ratio$: The risk-adjusted return, calculated as the annualized 
                profit divided by the standard deviation of negative profits.
        - Duration ratio: It measures the average duration of trades relative 
                to the total time traded, indicating whether the trades are 
                short- or long-term. A low value suggests quick trades, 
                while a high value indicates longer positions.
        - Payoff ratio: Ratio between the average profit of winning trades and 
                the average loss of losing trades (in absolute value).
        - Expectation: Expected value per trade, calculated as 
                (Win rate * Average win) - (Loss rate * Average loss).
        - Skewness: It measures the asymmetry of the return distribution. 
                A positive skewness indicates tails to the right (potentially large gains), 
                while a negative skewness indicates tails to the left (potentially large losses).
        - Kurtosis: It measures the "tailedness" or extremity of the return distribution. 
                A high kurtosis indicates heavy tails (more frequent extreme returns, both gains and losses), 
                while a low kurtosis suggests light tails (returns are more consistently close to the mean).
        - Average winning op: Average winning trade is calculated as 
                the average of only the winning trades.
        - Average losing op: Average losing trade is calculated as 
                the average of only the losing trades.
        - Average duration winn: Calculate the average duration 
                of each winner trade. 1 = 1 day.
        - Average duration loss: Calculate the average duration 
                of each losing trade. 1 = 1 day.
        - Daily frequency op: It is calculated by dividing the number of t
                ransactions by the number of trading days, where high 
                values mean high frequency and low values mean the opposite.
        - Max consecutive winn: Maximum consecutive winnings count. 
        - Max consecutive loss: Maximum consecutive loss count. 
        - Max losing streak: Maximum number of lost trades in drawdown.
        - Max drawdown:  The biggest drawdown the equity has ever had.
        - Average drawdown: The average of all drawdowns of equity curve, 
                indicating the typical loss experienced before recovery.
        - Max drawdown$: The biggest drawdown the profit has ever had.
        - Average drawdown$: The average of all drawdowns, 
                indicating the typical loss experienced before recovery.
        - Long exposure: What percentage of traders are long.
        - Winnings: Percentage of operations won.

    Returns:
        str|None: Statistics.
    """

    trades = _cm.__get_trades(name)

    name = list(name)[0] if isinstance(name, (tuple, set, list)) else name
    trades_data = _cm.__get_strategy(name=name)

    # Exceptions.
    if trades.empty: 
        raise exception.StatsError('Trades not loaded.')
    elif not 'profitPer' in trades.columns:  
        raise exception.StatsError('There is no data to see.')
    elif np.isnan(trades['profitPer'].mean()):
        raise exception.StatsError('There is no data to see.') 

    # Number of years operated.
    op_years = trades_op_years(
        trades['date'], trades_data['d_width_day'], trades_data['d_year_days'])

    # Annualized trades calc.
    trades['year'] = trades_group_year(trades['date'], op_years)
    trades['diary'] = trades_group_day(
        trades['date'], op_years, trades_data['d_year_days'])
    trades['duration'] = trades_duration(
        trades['positionDate'], trades['date'], trades_data['d_width_day'])

    ann_profit = trades.groupby('year')['profit'].sum()
    diary_profit = trades.groupby('diary')['profit'].sum()

    # Consecutive trades calc.
    trades_count_cs = trades['profitPer'].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
    trades_count_cs = pd.concat(
        [pd.Series([0]), trades_count_cs], ignore_index=True)

    group = (
        (trades_count_cs != trades_count_cs.shift()) 
        & (trades_count_cs != 0) 
        & (trades_count_cs.shift() != 0)
    ).cumsum()
    
    trades_csct = trades_count_cs.groupby(group).cumsum()

    # Trade streak calc.
    trades_streak = (trades_count_cs.cumsum() 
                     - np.maximum.accumulate(trades_count_cs.cumsum()))

    with np.errstate(over='ignore'):
        trades['multiplier'] = 1 + trades['profitPer'] / 100

        nan_inf = lambda x: x.where(~np.isinf(x), np.nan)
        multiplier_cumprod = nan_inf(trades.loc[:, 'multiplier'].cumprod().dropna())

        ann_return = nan_inf(trades.groupby('year')['multiplier'].prod())
        diary_return = nan_inf(trades.groupby('diary')['multiplier'].prod())

        text = utils.statistics_format({
        'Trades':[len(trades.index),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['CYAN']],

        'Op years':[utils.round_r(op_years, 2), _cm.__COLORS['CYAN']],

        'Return':[str(_return:=utils.round_r((_cm.c_tf(trades.loc[:, 'multiplier'].prod())-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return) > 0 else _cm.__COLORS['RED'],],

        'Profit':[str(_profit:=utils.round_r(np.nansum(trades['profit'].to_numpy()),2)),
                _cm.__COLORS['GREEN'] if float(_profit) > 0 else _cm.__COLORS['RED'],],

        'Gross earnings':[utils.round_r((trades['profit'][trades['profit']>0].sum()
                           if not pd.isna(trades['profit']).all() else 0), 4),
                        _cm.__COLORS['GREEN']],

        'Gross losses':[utils.round_r(abs(trades['profit'][trades['profit']<=0].sum())
                           if not pd.isna(trades['profit']).all() else 0, 4),
                        _cm.__COLORS['RED']],

        'Max return':[str(utils.round_r((multiplier_cumprod.max()-1)*100,2))+'%'],

        'Return from max':[str(utils.round_r(
            -((multiplier_cumprod.max()-1)
            - (_cm.c_tf(trades.loc[:, 'multiplier'].prod())-1))*100,2))+'%'],

        'Days from max':[str(utils.round_r(
            (trades['date'].dropna().iloc[-1]
                - trades['date'].dropna().loc[
                np.argmax(multiplier_cumprod)])
            / trades_data['d_width_day'], 2)),
            _cm.__COLORS['CYAN']],

        'Return ann':[str(_return_ann:=utils.round_r((ann_return.prod()**(1/op_years)-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return_ann) > 0 else _cm.__COLORS['RED'],],

        'Profit ann':[str(_profit_ann:=utils.round_r(float(ann_profit.mean()),2)),
                  _cm.__COLORS['GREEN'] if float(_profit_ann) > 0 else _cm.__COLORS['RED'],],

        'Return ann vol':[utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1)
                                        *np.sqrt(trades_data['d_year_days']), 2),
                          _cm.__COLORS['YELLOW']],

        'Profit ann vol':[utils.round_r(np.std(diary_profit.dropna(),ddof=1)
                                    *np.sqrt(trades_data['d_year_days']), 2),
                        _cm.__COLORS['YELLOW']],

        'Average ratio':[utils.round_r(average_ratio(trades), 2),
                        _cm.__COLORS['YELLOW'],],

        'Average return':[str(round((
                trades.loc[:, 'multiplier'].dropna().to_numpy().mean()-1)*100,2))+'%',
            _cm.__COLORS['YELLOW'],],

        'Average profit':[str(round(trades.loc[:, 'profit'].mean(),2))+'%',
                    _cm.__COLORS['YELLOW'],],

        'Profit fact':[_profit_fact:=utils.round_r(profit_fact(trades.loc[:, 'profit']), 3),
                _cm.__COLORS['GREEN'] if float(_profit_fact) > 1 else _cm.__COLORS['RED'],],

        'Return diary std':[(_return_std:=utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1), 2)),
                    _cm.__COLORS['YELLOW'] if float(_return_std) > 1 else _cm.__COLORS['GREEN'],],

        'Profit diary std':[(_profit_std:=utils.round_r(np.std(diary_profit.dropna(),ddof=1), 2)),
                      _cm.__COLORS['YELLOW'] if float(_profit_std) > 1 else _cm.__COLORS['GREEN'],],

        'Math hope':[_math_hope:=round(math_hope(trades.loc[:, 'profit']), 3),
            _cm.__COLORS['GREEN'] if float(_math_hope) > 0 else _cm.__COLORS['RED'],],

        'Math hope r':[_math_hope_r:=round(
                math_hope_relative(trades, trades.loc[:, 'profitPer']), 3),
            _cm.__COLORS['GREEN'] if float(_math_hope_r) > 0 else _cm.__COLORS['RED'],],

        'Historical var':[0 if trades['profit'].dropna().empty else utils.round_r(
                            var_historical(trades.loc[:, 'profit'].dropna()), 2)],

        'Parametric var':[0 if trades['profit'].dropna().empty else utils.round_r(
                            var_parametric(trades.loc[:, 'profit'].dropna()), 2)],

        'Sharpe ratio':[utils.round_r(sharpe_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            trades_data['d_year_days'],
            (diary_return.dropna()-1)*100), 2)],

        'Sharpe ratio$':[utils.round_r(sharpe_ratio(
            np.average(ann_profit),
            trades_data['d_year_days'],
            diary_profit), 2)],

        'Sortino ratio':[utils.round_r(sortino_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            trades_data['d_year_days'],
            (diary_return.dropna()-1)*100), 2)],

        'Sortino ratio$':[utils.round_r(sortino_ratio(
            np.average(ann_profit),
            trades_data['d_year_days'],
            diary_profit), 2)],

        'Duration ratio':[utils.round_r(
            np.nansum(trades['duration'].to_numpy())/len(trades.index), 2),
            _cm.__COLORS['CYAN']],

        'Payoff ratio':[utils.round_r(payoff_ratio(trades.loc[:, 'profitPer']), 3)],

        'Expectation':[utils.round_r(expectation(trades.loc[:, 'profitPer']))],

        'Skewness':[utils.round_r((diary_return.dropna()-1).skew(), 2)],

        'Kurtosis':[utils.round_r((diary_return.dropna()-1).kurt(), 2)],

        'Average winning op':[str(utils.round_r(trades.loc[:, 'profitPer'][
                trades['profitPer'] > 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['GREEN']],

        'Average losing op':[str(utils.round_r(trades.loc[:, 'profitPer'][
                trades['profitPer'] < 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['RED']],

        'Average duration winn':[str(utils.round_r(trades.loc[:, 'duration'][
                trades['profitPer'] > 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Average duration loss':[str(utils.round_r(trades.loc[:, 'duration'][
                trades['profitPer'] < 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Daily frequency op':[utils.round_r(
            len(trades.index) / (op_years*trades_data['d_year_days']), 2),
            _cm.__COLORS['CYAN']],

        'Max consecutive winn':[trades_csct.max(),
                                _cm.__COLORS['GREEN']],

        'Max consecutive loss':[abs(trades_csct.min()),
                                _cm.__COLORS['RED']],

        'Max losing streak':[abs(trades_streak.min())],

        'Max drawdown':[str(round(
            max_drawdown(multiplier_cumprod)*100,1)) + '%'],

        'Average drawdown':[str(-round(np.mean(
            get_drawdowns(multiplier_cumprod))*100, 1)) + '%'],

        'Max drawdown$':[str(round(
            max_drawdown(trades['profit'].dropna().cumsum()+
                               trades_data['init_funds'])*100,1)) + '%'],

        'Average drawdown$':[str(-round(np.mean(
            get_drawdowns(trades['profit'].dropna().cumsum()+
                                trades_data['init_funds']))*100, 1)) + '%'],

        'Long exposure':[str(round(
            long_exposure(trades.loc[:, 'typeSide'])*100)) + '%',
            _cm.__COLORS['GREEN']],

        'Winnings':[str(round(winnings(trades.loc[:, 'profitPer'])*100)) + '%'],

        }, f"---Statistics of '{trades_data['name']}'---")

    text = text if _cm.dots else text.replace('.', ',')
    if data: 
        text += (stats_icon(False) or '')

    if prnt: print(text)
    else: return text

def trades_op_years(trades_date:pd.Series, day_width:float, year_days:int) -> float:
    """
    Trades operated years

    Return the number of years operated.

    Args:
        trades_date (Series): 'Series' with the dates.
        day_width (float): Width of each day.
        year_days (int): Operated days per year.

    Return:
        float: Years operated on float.
    """

    return abs(
        (trades_date.iloc[-1] - trades_date.iloc[0])/
        (day_width*year_days))

def trades_duration(position_date:pd.Series, date:pd.Series, 
                    day_width:float) -> pd.Series:
    """
    Trades duration in days

    Return the duration of trades in days.

    Args:
        position_date (Series): 'Series' with closing date for each trade.
        date (Series): 'Series' with the dates.
        day_width (float): Width of each day.

    Return:
        Series: Years operated on float.
    """

    if day_width <= 0:
        raise exception.StatsError("'day_width' cannot be less or equal than 0.")

    return (position_date-date)/day_width

def trades_group_year(trades_date:pd.Series, op_years:float) -> pd.Series:
    """
    Trades group by year

    Returns 'Series' with the same size as 'trades_date' 
        where each index is grouped by year.

    Args:
        trades_date (Series): 'Series' with the dates.
        op_years (float): Years operated.

    Return:
        Series: Year groups.
    """

    if op_years <= 0:
        raise exception.StatsError("'op_years' cannot be less or equal than 0.")

    return ((trades_date - trades_date.iloc[0]) / 
            (trades_date.iloc[-1] - trades_date.iloc[0]) * 
            op_years).astype(int)

def trades_group_day(trades_date:pd.Series, op_years:float, year_days:int) -> pd.Series:
    """
    Trades group by day

    Returns 'Series' with the same size as 'trades_date' 
        where each index is grouped by day.

    Args:
        trades_date (Series): 'Series' with the dates.
        op_years (float): Years operated.
        year_days (int): Operated days per year.

    Return:
        Series: Day groups.
    """

    if op_years <= 0:
        raise exception.StatsError("'op_years' cannot be less or equal than 0.")
    elif year_days <= 0:
        raise exception.StatsError("'year_days' cannot be less or equal than 0.")

    return ((trades_date - trades_date.iloc[0]) / 
                (trades_date.iloc[-1] - trades_date.iloc[0]) * 
                op_years*year_days).astype(int)

def earnings_intime(names:list[str|int|None]|str|int|None = None,
                    in_days:float = 365, profit:bool = True, 
                    prnt:bool = True) -> tuple[pd.Series, str]|None:
    """
    Earnings in time

    Statistics of earnings each 'in_days' amount of days.

    Note:
        This function can send a text to the console that is too long; 
        in that case, it is possible that not all of it will be printed.

    Args:
        names (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        in_days (float, optional): Number of days per group, cannot be less than 1.
        profit (bool, optional): If it is True, it is calculated with 'profit', 
            otherwise with 'profitPer'.
        prnt (bool, optional): Print the statistics or return them.

    Return:
        tuple[Series, str]|None: 
            If 'prnt' is False, a tuple is returned with the text 
            that would be printed and the statistics.
    """

    if in_days < 1:
        raise exception.StatsError("'in_days' cannot be less than 1.")

    trades = _cm.__get_trades(names=names)
    name = list(names)[0] if isinstance(names, (tuple,set,list)) else names
    trades_data = _cm.__get_strategy(name=name)

    if trades.empty:
        raise exception.StatsError('Trades not loaded.')

    # Date trades calc.
    op_years = trades_op_years(
        trades['date'], trades_data['d_width_day'], trades_data['d_year_days'])
    day_series = trades_group_day(
        trades['date'], op_years, trades_data['d_year_days'])
    trades['day'] = day_series

    diary_profit = trades.groupby('day')['profit' if profit else 'profitPer'].sum()
    indays_profit = diary_profit.groupby(
        np.arange(len(diary_profit)) // in_days).sum()

    data = {}
    for i,v in enumerate(indays_profit):
        value = {f'{i+1}-{in_days} days':[round(v, 1), 
            _cm.__COLORS['GREEN'] if round(v, 1) > 0 else _cm.__COLORS['RED']]}
        data.update(value)

    data.update({'Winrate':[
        str(round(winnings(indays_profit)*100)) + '%',
        _cm.__COLORS['CYAN']]})
    text = utils.statistics_format(data, f"---Earnings per days---")

    text = text if _cm.dots else text.replace('.', ',')
    if prnt: print(text) 

    return (indays_profit, text)
