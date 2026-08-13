"""
Stats module

This module contains functions to calculate different metrics.

Variables:
    logger (Logger): Logger variable.

Functions:
    average_ratio: Based on the take profit and stop loss 
            positions, it calculates an average ratio.
    profit_fact: Calculate the profit factor of the values.
    gain_loss_diff: Calculate the difference between the sum of gains and the sum of losses.
    math_hope: Calculate the mathematical expectation of the values.
    math_hope_relative: Calculate the relative mathematical 
            expectation based on the average_ratio and the profits.
    winnings: Calculate the percentage of positive numbers in the series.
    sharpe_ratio: Calculate the Sharpe ratio using the 
            returns / sqrt(days of the year) / standard deviation of the data.
    sharpe_ratio_woa: Calculate the Sharpe ratio without annualizing.
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
    percentile_rank: Calculate the percentile of 'x' in 'data'.
    z_score: Calculate the number of standard deviations that separate 'x'.
    perf_tzone_chart: Chart the best and worst hours/minutes of your strategy.
    distribution_chart: Displays distribution graphs.
    monte_carlo_bsim: Calculates Monte Carlo simulations.
    permutation: Generate a permutation test.
    correlation: Measure correlation between strategies.
    earnings_intime: Statistics of earnings each 'x' amount of days.
    stats_icon: Shows statistics related to the financial icon.
    stats_trades: Statistics of the trades.
    trades_op_years: Return the number of years operated.
    trades_duration: Return the duration of trades in days.
    trades_group_year: Returns 'Series' by grouping each trade by year.
    trades_group_day: Returns 'Series' by grouping each trade by days.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Callable, Sequence
from time import time
import logging

from . import custom_plt as cpl
from . import _commons as _cm
from . import flex_data
from . import exception
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

def profit_fact(profits:pd.Series|np.ndarray) -> float:
    """
    Profit fact.

    Calculate the profit factor of the values.

    Args:
        profits (Series|ndarray): Returns on each operation.

    Returns:
        float: Profit fact.
    """

    profits = profits[~np.isnan(profits)]

    pos = (profits>0)
    neg = (profits<=0)
    if (pos.sum() > 0 
        and neg.sum() > 0):

        return (profits[pos].sum()
                / abs(profits[neg].sum()))
    return 0

def gain_loss_diff(profits:pd.Series|np.ndarray) -> float:
    """
    Gain-loss difference.

    Calculate the difference between the sum of gains and the sum of losses.

    Args:
        profits (Series|ndarray): Returns on each operation.

    Returns:
        float: Gain-loss difference.
    """
    profits = profits[~np.isnan(profits)]
    if ((profits>0).sum() > 0 
        and (profits<=0).sum() > 0):

        return (profits[profits>0].sum()
                - abs(profits[profits<=0].sum()))
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

    return winnings(profits)*average_ratio(trades)-(1-winnings(profits))

def winnings(profits:pd.Series|np.ndarray) -> float:
    """
    Winnings percentage.

    Calculate the percentage of positive numbers in the series.

    Args:
        profits (Series|ndarray): Returns on each operation.

    Returns:
        float: Winnings percentage.
    """

    if (not ((profits>0).sum() == 0 
        or len(profits) == 0)):

        return (profits>0).sum()/len(profits)
    return 0

def sharpe_ratio(ann_av:float|np.floating, year_days:int, diary_per:pd.Series|np.ndarray) -> float:
    """
    Sharpe ratio.

    Calculate the Sharpe ratio using the 
        returns / sqrt(days of the year) / standard deviation of the data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float|floating): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (Series|ndarray): Daily return.

    Returns:
        float: Sharpe ratio.
    """

    diary_per = diary_per[~np.isnan(diary_per)]

    std_dev = np.std(diary_per, ddof=1)
    if std_dev < 1e-9: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def sharpe_ratio_woa(returns:pd.Series|np.ndarray) -> float:
    """
    Sharpe ratio without annualization.

    Calculate the Sharpe ratio without annualizing the returns:
        returns / standard deviation of the data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        returns (Series|ndarray): Returns.

    Returns:
        float: Sharpe ratio.
    """

    returns = returns[~np.isnan(returns)]

    std_dev = np.std(returns, ddof=1)
    if std_dev < 1e-9: return 0

    return returns.mean() / std_dev

def sortino_ratio(ann_av:float|np.floating, year_days:int, diary_per:pd.Series|np.ndarray) -> float:
    """
    Sortino ratio.

    Calculate the Sortino ratio with a calculation similar to the 
        Sharpe ratio but only with the standard deviation of negative data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float|floating): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (Series|ndarray): Daily return.

    Returns:
        float: Sortino ratio.
    """

    diary_per = diary_per[~np.isnan(diary_per)]

    std_dev = np.std(diary_per[diary_per < 0], ddof=1)
    if std_dev < 1e-9: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def payoff_ratio(profits:pd.Series|np.ndarray) -> float:
    """
    Payoff ratio.

    Calculates the payout rate using the absolute 
        mean of positive numbers/mean of negative numbers.

    Args:
        profits (Series|ndarray): Returns on each operation..

    Returns:
        float: Payoff ratio.
    """

    profits = profits[~np.isnan(profits)]

    return (profits[profits > 0].mean() 
            / abs(profits[profits < 0].mean()))

def expectation(profits:pd.Series|np.ndarray) -> float:
    """
    Expectation.

    Calculate the expectation based on payoff.

    Args:
        profits (Series|ndarray): Returns on each operation.

    Returns:
        float: Expectation.
    """

    wins = winnings(profits)
    return ((wins*payoff_ratio(profits)) - (1-wins))

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

def max_drawdown(values:list|pd.Series|np.ndarray) -> float:
    """
    Maximum drawdown.

    Calculate the maximum drawdown of `values`.
    Wrapper of 'get_drawdowns'.

    Args:
        values (list|Series|ndarray): The ordered data to calculate the maximum drawdown.

    Returns:
        float: The abs maximum drawdown from the given data.
    """

    return -get_drawdowns(values=values).min()

def get_drawdowns(values:list|pd.Series|np.ndarray) -> np.ndarray:
    """
    Get drawdowns.

    Calculate the drawdowns of `values`.

    Args:
        values (list|pd.Series|np.ndarray): 
            The ordered data to calculate the drawdowns.

    Returns:
        np.ndarray: The drawdowns from the given data.
    """

    values = np.asarray(values)
    if len(values) == 0:
        return np.array([])

    max_values = np.maximum.accumulate(values)
    drawdowns = (values - max_values) / max_values

    return drawdowns

def percentile_rank(data:np.ndarray, x:float, form:str = 'mean') -> float:
    """
    Percentile rank

    Calculate the percentile of 'x' in 'data'.

    Args:
        data (np.ndarray): Data.
        x (float): Value to calculate.
        form (str, optional): Method of calculation, available: '
            'mean', 'strict', 'weak', 'rank'.

    Returns:
        float: Percentile fraction.
    """
    form = form.strip().lower()
    len_data = len(data)

    if form not in ('mean', 'strict', 'weak', 'rank'):
        raise exception.StatsError(f"'{form}' is not a valid form.")
    elif len_data < 1:
        len_data = 1

    left  = np.sum(data < x)
    right = np.sum(data <= x)

    forms = {
        'mean': lambda: (left + right) / 2,
        'strict': lambda: left,
        'weak': lambda: right,
        'rank': lambda: (left + right + (left != right)) / 2,
    }
    return forms[form]() / len_data

def z_score(data:np.ndarray, x:float) -> float:
    """
    Z Score

    Calculate the number of standard deviations that separate 'x'.

    Args:
        data (np.ndarray): Data.
        x (float): Value to calc.

    Return:
        float: Return the metric.
    """

    std_dev = np.std(data, ddof=1)
    if std_dev < 1e-9: return np.nan

    return (x-data.mean())/std_dev

def perf_tzone_chart(names:Sequence[str|int|None]|str|int|None = None,
                     view:str = 'p/d', col:str|None = 'profitPer', 
                     panel:str = 'add', style:str|None = 'last', 
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
        names (Sequence[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        view (str, optional): Specifies which graphics to display. 
            Default is 'p/d'. Maximum 8.
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

    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif panel not in ('new', 'add'):
        raise exception.StatsError(
            f"'{panel}' Not a valid option for: 'panel'.")

    plt_colors, style_name = cpl.style_def(name=style, update=style_c)
    col = col or 'profitPer'

    trades = _cm.__get_trades(names=names)
    name = list(names)[0] if isinstance(names, (tuple, set, list)) else names
    trades_data = _cm.__get_strategy(name=name)

    if trades.empty:
        logger.warning('Trades not loaded'); return
    elif not 'profit' in trades.columns or not 'profitPer' in trades.columns:
        logger.warning('No closed trades.'); return

    hour = lambda index: ((index % trades_data['d_width_day']) 
                          / trades_data['d_width_day'] * 24).astype(int)
    minute = lambda index: ((index % (trades_data['d_width_day']/60)) 
                          / (trades_data['d_width_day']/60) * 60).astype(int)

    gdir = plt_colors.get('gdir', False)
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    fig = plt.figure(figsize=(16,8),dpi=_cm.graph_dpi)
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
        cpl.config_ax(ax, bg_color=plt_colors['bg'], date=False, gdir=gdir)

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
        title=f'Performance in time - {style_name}',
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total'
    )

def distribution_chart(data:tuple[dict[str, np.ndarray|list], str], view:str = 's/d',
                      n_trades:int|None = None, diff_ini:bool = True, progress:bool = True,
                      panel:str = 'add', style:str|None = 'last', style_c:dict|None = None, 
                      block:bool = True) -> None:
    """
    Distribution chart

    Generates distribution graphs.

    Available Graphics:
    - 's' = Simulation chart.
    - 'd' = Total return distribution, progressive graph.
    - 'pf' = Profit factor distribution bell.
    - 'sr' = Sharpe ratio distribution.
    - 'tr' = Total return distribution.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Note:
        The profit factor here is calculated from percentage returns, not dollar profit.
        This makes it valid for relative comparison between the original and the simulations, 
        but not comparable to a dollar-based profit factor.

    Args:
        data (tuple[dict[str,ndarray|list],str]: Data extracted from a simulation.
            You can extract data from 'monte_carlo_bsim' or 'permutation' functions.
            Tuple with (data of simulations, title).
            Data dictionary should have two names: 'stats' and 'data'. 'data' will contain each 
            value per simulation, and 'stats' the previously calculated statistics in a structured array. 
            This is done to obtain the total statistics and reduce the amount of data.
            Columns used in 'stats': 'last_result', 'profit_fact', 'sharpe_ratio'. 
            If not are found, it will be calculated based on 'data'.
        view (str, optional): Specifies which graphics to display. 
            Default is 'd/p/b'. Maximum 8.
        n_trades (int|None, optional): For graph 'd' how many simulations 
            will be shown.
        diff_ini (bool, optional): The original strategy trades stands out in all the charts.
        progress (bool, optional): If True, shows a progress bar and timer.
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
    data_data = data[0].get('data', [])

    data_stats = data[0].get('stats', np.array([]))
    assert isinstance(data_stats, np.ndarray), "'stats' It must be a structured array."
    data_stats_names = data_stats.dtype.names if hasattr(data_stats.dtype, 'names') else None

    dvalues = [i[~np.isnan(i)] for i in data_data]
    if not dvalues or all([len(i)<1 for i in dvalues]):
        logger.warning("Given data are empty."); return
    elif panel not in ('new', 'add'):
        raise exception.StatsError(
            f"'{panel}' Not a valid option for: 'panel'.")
    elif n_trades and n_trades <= 1 and n_trades > len(dvalues):
        raise exception.StatsError(utils.text_fix("""
                        'n_trades' can only be greater than 1 and 
                        less than or equal to the length of given data.
                        """, newline_exclude=True))

    plt_colors, style_name = cpl.style_def(name=style, update=style_c)
    gdir = plt_colors.get('gdir', False)
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    fig = plt.figure(figsize=(16,8),dpi=_cm.graph_dpi)
    fig.subplots_adjust(left=0, right=1, top=1, 
                        bottom=0, wspace=0, hspace=0)

    graphics = ['s','d','pf','sr','tr']
    axes, v_view = cpl.ax_view(view=view, graphics=graphics)

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    if progress:
        load_prgs.reset_size(len(v_view))

    for i,v in enumerate(v_view):
        ax = axes[i]
        cpl.config_ax(ax, bg_color=plt_colors['bg'], date=False, gdir=gdir)

        match v:
            case 's':
                for i in range(n_trades if n_trades else len(dvalues)):
                    alpha = 0.5
                    zorder = 2
                    color = market_colors.get('u', 'g') if diff_ini else None
                    if i==0 and diff_ini:
                        zorder = 3
                        color = market_colors.get('d', 'r')
                        alpha = 1

                    curve = dvalues[i].cumsum()
                    ax.plot(range(0, len(curve)), curve, alpha=alpha, zorder=zorder, color=color)

                ax.legend(['Simulations.'], loc='upper left')
                ax.set_xlim(-1, len(dvalues[0]))
            case 'd':
                if data_stats_names and 'last_result' in data_stats_names:
                    last_result = data_stats['last_result']
                else: last_result = np.array([ar.sum() for ar in dvalues])

                sorted_results = np.sort(last_result)
                parts = np.array_split(sorted_results, 100)
                means:list[float] = [np.mean(part) for part in parts if len(part) > 0]

                color_u = lambda x: utils.mult_color(
                    color=market_colors['u'], multiplier=x)
                color_d = lambda x: utils.mult_color(
                    color=market_colors['d'], multiplier=x)
                colors = np.array([
                    color_u(val/np.max(means)+1) if val >= 0 else color_d(1-val/np.min(means))
                    for val in means if val != 0
                ])

                if diff_ini:
                    pos_in_sorted = np.searchsorted(sorted_results, last_result[0])
                    bin_index = int(pos_in_sorted / len(sorted_results) * len(means))

                    ax.axvline(x=bin_index, color=market_colors.get('d', 'r'), linewidth=1.2,
                    linestyle='--', zorder=2, label='Original.')

                ax.bar(list(range(len(means))), means, width=0.8, color=colors, 
                    label='Total return distribution.')
                ax.legend(loc='upper left')
            case 'pf':
                if data_stats_names and 'profit_fact'  in data_stats_names:
                    pf_result = data_stats['profit_fact']
                else: pf_result = np.array([profit_fact(ar) for ar in dvalues])

                count, hist = np.histogram(pf_result, bins=25)
                color_u = lambda x: utils.mult_color(
                    color=market_colors['u'], multiplier=x)
                colors = np.array([color_u((val/np.max(count))+0.2) for val in count if val != 0])

                if diff_ini:
                    ax.axvline(x=pf_result[0], color=market_colors.get('d', 'r'), linewidth=1.2,
                    linestyle='--', zorder=2, label='Original.')

                ax.bar(hist[:-1], count, width=np.mean(np.diff(hist))*0.8, color=colors, 
                    label='Profit factor distribution.')
                ax.legend(loc='upper left')
            case 'sr':
                if data_stats_names and 'sharpe_ratio'  in data_stats_names:
                    sr_result = data_stats['sharpe_ratio']
                else: sr_result = np.array([sharpe_ratio_woa(ar) for ar in dvalues])

                count, hist = np.histogram(sr_result, bins=25)
                color_u = lambda x: utils.mult_color(
                    color=market_colors['u'], multiplier=x)
                colors = np.array([color_u((val/np.max(count))+0.2) for val in count if val != 0])

                if diff_ini:
                    ax.axvline(x=sr_result[0], color=market_colors.get('d', 'r'), linewidth=1.2,
                    linestyle='--', zorder=2, label='Original.')

                ax.bar(hist[:-1], count, width=np.mean(np.diff(hist))*0.8, color=colors, 
                    label='Sharpe ratio distribution.')
                ax.legend(loc='upper left')
            case 'tr':
                if data_stats_names and 'last_result'  in data_stats_names:
                    tr_result = data_stats['last_result']
                else: tr_result = np.array([ar.sum() for ar in dvalues])

                count, hist = np.histogram(tr_result, bins=25)
                color_u = lambda x: utils.mult_color(
                    color=market_colors['u'], multiplier=x)
                colors = np.array([color_u((val/np.max(count))+0.2) for val in count if val != 0])

                if diff_ini:
                    ax.axvline(x=tr_result[0], color=market_colors.get('d', 'r'), linewidth=1.2,
                    linestyle='--', zorder=2, label='Original.')

                ax.bar(hist[:-1], count, width=np.mean(np.diff(hist))*0.8, color=colors, 
                    label='Total return distribution.')
                ax.legend(loc='upper left')
            case _: pass
        load_prgs.next()

    title = data[1].strip()
    cpl.add_window(
        fig=fig,
        title=f'{title if title else "Distribution"} - {style_name}',
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total'
    )

def monte_carlo_bsim(names:Sequence[str|int|None]|str|int|None = None, 
                    n_trades:int|None = None, n_sims:int|None = 10000, 
                    percentiles:Sequence[int|float] = [1,5,10,24,50,75],
                    replace:bool = True, full_output:bool|int = False, 
                    prnt:bool = True, progress:bool = True) -> tuple[dict[str,np.ndarray|list],str]:
    """
    Monte Carlo bootstrap simulation

    Calculate a Monte Carlo bootstrap simulation and gives statistics.

    For documentation of statistics, read the 'stats_trades' docstring.

    Note:
        The profit factor here is calculated from percentage returns, not dollar profit.
        This makes it valid for relative comparison between the original and the simulations, 
        but not comparable to a dollar-based profit factor.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        n_trades (int|None, optional): Number of trades per simulation, 
            None = length of loaded trades.
        n_sims (int|None, optional): Number of simulations.
        percentiles (Sequence[int|float], optional): Percentiles for statistics.
        replace (bool, optional): If False change the Monte Carlo simulation, 
            remove the possibility of duplicates; 'n_trades' 
            cannot exceed the length of the trades.
        full_output (bool|int, optional): How many simulations are returned 
            in full; statistics are always computed over all 'n_sims' regardless of this value.
            - True: return every simulation in full, can use a lot of memory.
            - False: return every simulation in full unless total simulated 
                data exceeds '_max_elements' values, in which case only the first 
                100 are returned.
            - int: same as False, but caps the returned simulations at 
                this value instead of 100.
        prnt (bool, optional): If True, the statistics are 
            printed on the console.
        progress (bool, optional): If True, shows a progress bar and timer.

    Returns
        tuple[dict[str,ndarray|list],str]: 
            Tuple with: tuple with all data and title text.
    """

    # Exceptions.
    if n_trades and n_trades <= 1:
        raise exception.StatsError(
            "'n_trades' can only be greater than 1.")
    elif n_sims and n_sims <= 0:
        raise exception.StatsError(
            "'n_sims' can only be greater than 0.")

    trades = _cm.__get_trades(names=names)['profitPer'].dropna().to_numpy(dtype=np.float32)
    sim = [trades[:n_trades]]
    n_sims = n_sims or 10000

    if n_trades and not replace and (n_trades <= 1 or n_trades > len(trades)):
        raise exception.StatsError(
            f"'n_trades' has to be greater than 1 and less than the total number of trades ({len(trades)}).")
    elif len(trades) < 1:
        logger.warning('Trades not loaded.'); return ({}, '')

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    if progress:
        load_prgs.reset_size(n_sims)
    skip = max(1, n_sims // _cm.max_bar_updates)

    # Data limitation
    rn_trades = (n_trades or len(trades))
    maxsims = full_output if type(full_output) is int and full_output > 0 else 100
    alldata = full_output if isinstance(full_output, bool) else False

    total_data = n_sims * rn_trades
    if total_data > _cm._max_elements and not alldata:
        total_data = min(maxsims * rn_trades, total_data)
    if total_data != n_sims * rn_trades:
        logger.warning(f'Returned data limited to {total_data//rn_trades} sims')

    stats = []
    def calc_stats(returns:np.ndarray) -> None:
        """
        Calculate statistics

        Args:
            returns (ndarray): List of returns.
        """
        nonlocal stats
        stats_mult = 1 + returns / 100
        stats_mult_cumprod = stats_mult.cumprod()

        stats.append({
            'last_result':returns.sum(),
            'profit_fact':profit_fact(returns),
            'sharpe_ratio':sharpe_ratio_woa(returns),
            'expectation':expectation(returns),
            'max_drawdown':max_drawdown(stats_mult_cumprod),
            'avg_drawdown':get_drawdowns(stats_mult_cumprod).mean(),
            'winrate':winnings(returns)*100,
        })
    calc_stats(sim[0])

    # Monte carlo sim
    for i in range(n_sims):
        trades_s = np.random.choice(trades, size=rn_trades, replace=replace)
        calc_stats(trades_s)

        if (len(sim)-1)*rn_trades < total_data:
            sim.append(trades_s)
        if i % skip == 0 or i+1 >= n_sims:
            load_prgs._step = i-(0 if i == n_sims-1 else 1)
            load_prgs.next()

    dtype = [(k, type(v)) for k, v in stats[0].items()]
    stats_carr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

    returned = ({'data':sim,'stats':stats_carr}, 'Monte Carlo simulation')
    if not prnt: return returned

    percentiles_r = np.percentile(stats_carr['last_result'], percentiles)
    percentiles_t = {
        f'Percentile {percentiles[i]}':[
            round(v, 2), _cm.__COLORS['GREEN'] if v > 0 else _cm.__COLORS['RED']
        ] for i,v in enumerate(percentiles_r)}

    text = {
        'Average return':[(md_rtrn:=round(np.average(stats_carr['last_result']), 1)),
                         _cm.__COLORS['GREEN'] if md_rtrn > 0 else _cm.__COLORS['RED']],
        'Profit fact avg':[(prft_fact:=utils.round_r(np.average(stats_carr['profit_fact']), 3)),
                              _cm.__COLORS['GREEN'] if prft_fact > 1 else _cm.__COLORS['RED']],
        'Max drawdown avg':[str(round(np.average(stats_carr['max_drawdown'])*100, 1)) + '%'],
        'Average drawdown avg':[str(-round(np.average(stats_carr['avg_drawdown'])*100, 1)) + '%'],
        'Expectation avg':[utils.round_r(np.average(stats_carr['expectation']), 2)],
        'Winnings avg':[str(round(np.average(stats_carr['winrate']), 1)) + '%',
                           _cm.__COLORS['GREEN']],
        f"\n{_cm.__COLORS['CYAN']}Percentiles{_cm.__COLORS['RESET']}":['']
    }
    text.update(percentiles_t)

    text = utils.statistics_format(text, f"---Statistics of Monte Carlo---")
    text = text if _cm.dots else text.replace('.', ',')

    print(text)
    return returned

def permutation(names:Sequence[str|int|None]|str|int|None = None,
                n_sims:int = 1000, max_concrr:int|None = None, cost:bool = True, 
                frag_attps:int = 3, full_output:bool|int = False, prnt:bool = True, 
                progress:bool = True) -> tuple[dict[str,np.ndarray|list],str]:
    """
    Permutation

    Generate a permutation test.
    Check if your strategy is better than randomness.

    Note:
        Random simulations are based solely on the duration of each trade, 
        so if your backtest uses closing or opening prices other than the 
        candle's close, it may cause inaccuracies in the test.
        The profit factor here is calculated from percentage returns, not dollar profit.
        This makes it valid for relative comparison between the original and the simulations, 
        but not comparable to a dollar-based profit factor.

    Info:
        - Z-score: Calculate the number of standard deviations that separate the value.
        - P-value: Calculate how unusual the backtest is compared to the simulations.
        - Percentile-rank: The percentile of the backtest in the distribution of the simulations.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        n_sims (int, optional): Number of random simulations.
        max_concrr (int|None, optional): Maximum number of open positions at the same time.
        cost (bool, optional): Simulate cost (spread and slippage).
        frag_attps (int, optional): How many attempts before losing the simulation, due to fragmented data.
        full_output (bool|int, optional): How many simulations are returned 
            in full; statistics are always computed over all 'n_sims' regardless of this value.
            - True: return every simulation in full, can use a lot of memory.
            - False: return every simulation in full unless total simulated 
                data exceeds 10000 values, in which case only the first 
                100 are returned.
            - int: same as False, but caps the returned simulations at 
                this value instead of 100.
        prnt (bool, optional): If True, the statistics are 
            printed on the console.
        progress (bool, optional): If True, shows a progress bar and timer.

    Returns:
        tuple[dict[str,ndarray|list],str]: 
            Tuple with: list with all simulations and title text.
            The total number of simulations may be less than required; 
            to avoid this, use a higher 'max_concrr' value or add more data.
    """

    # Exceptions.
    if _cm.__data is None or not type(_cm.__data) is pd.DataFrame or _cm.__data.empty:
        raise exception.StatsError('Data not loaded.')
    elif max_concrr is not None and max_concrr < 1:
        raise exception.StatsError("'max_concrr' must be >= 1.")
    elif frag_attps < 1:
        raise exception.StatsError("'frag_attps' cannot be less than 1.")

    trades = _cm.__get_trades(names=names)
    if trades.empty:
        logger.warning('Trades not loaded.'); return ({}, '')
    if not 'positionClose' in trades.columns or trades['positionClose'].isna().all():
        logger.warning('No closed trades.'); return ({}, '')

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    if progress:
        load_prgs.reset_size(n_sims)
    skip = max(1, n_sims // _cm.max_bar_updates)

    signals = []
    for t in trades.values.tolist():
        if np.isnan(t[7]): continue

        signals.append([t[7]-t[0], t[4]]) # [duration, typeSide]

    if max_concrr is None:
        total_duration = np.asarray(signals)[:, 0].sum()
        max_concrr = 1+int(total_duration/(_cm.__data.index[-1]-_cm.__data.index[0]))
        logger.warning("'max_concrr' not specified, using estimated value: "+str(max_concrr))

    # Data limitation
    maxsims = full_output if type(full_output) is int and full_output > 0 else 100
    alldata = full_output if isinstance(full_output, bool) else False

    total_data = n_sims * len(trades)
    if total_data > _cm._max_elements and not alldata:
        total_data = min(maxsims * len(trades), total_data)
    if total_data != n_sims * len(trades):
        logger.warning(f'Returned data limited to {total_data//len(trades)} sims')

    spread_pct = _cm.__spread_pct or flex_data.CostsValue(0)
    slippage_pct = _cm.__slippage_pct or flex_data.CostsValue(0)

    data_idx = _cm.__data.index.values
    data_idx_len = len(data_idx)
    data_closes = _cm.__data['close'].to_numpy()

    test:list[np.ndarray] = [trades['profitPer'].dropna().to_numpy()]
    stats = []
    def calc_stats(returns:np.ndarray) -> None:
        """
        Calculate statistics

        Args:
            returns (ndarray): List of returns.
        """
        nonlocal stats

        stats.append({
            'last_result':returns.sum(),
            'profit_fact':profit_fact(returns),
            'sharpe_ratio':sharpe_ratio_woa(returns),
        })
    calc_stats(test[0])

    all_indices = np.arange(data_idx_len)
    signals.sort(key=lambda rep: rep[0], reverse=True)
    signals = np.asarray(signals)

    sff_sig_sum = np.cumsum(signals[:, 0][::-1])[::-1]/max_concrr
    signals_range = range(len(signals))

    max_idxs:np.ndarray = data_idx.searchsorted(data_idx[-1]-sff_sig_sum)

    if (max_idxs-1 < 0).any():
        raise exception.StatsError(
            "Insufficient data set: a trade duration exceeds the dataset length. "
            "Increase 'max_concrr' or use the original data set."
        )

    def placement() -> np.ndarray:
        """
        Placement

        Run the random placement simulation.
        Raise 'DataFragError' when the data becomes fragmented and 
            there is no data left for the rest of the trades.

        Returns:
            ndarray: Returns array with return.
        """
        nonlocal signals, signals_range, all_indices, data_closes, data_idx_len, \
            data_idx, slippage_pct, spread_pct, max_concrr, sff_sig_sum, test_ls

        coverage = np.zeros(data_idx_len+1)
        re_ls = []

        for i in signals_range:
            candidate_indices = all_indices[:max_idxs[i]]
            exit_indices = data_idx.searchsorted(data_idx[candidate_indices]+signals[i][0])

            in_range = exit_indices < data_idx_len
            blocked_prefix = (coverage>=max_concrr).cumsum()
            blocked_in_range = (
                blocked_prefix[exit_indices]-blocked_prefix[candidate_indices]
            ) <= 0

            valid_indices = candidate_indices[in_range & blocked_in_range]
            if len(valid_indices) == 0:
                raise exception.DataFragError

            idx = valid_indices[np.random.randint(len(valid_indices))]
            exit_idx = int(exit_indices[idx])
            coverage[idx:exit_idx] += 1

            entry = data_closes[idx]
            entry_cost = data_closes[idx] * ((
                (spread_pct.get_taker()/100/2) + (slippage_pct.get_taker()/100)) if cost else 0)
    
            exit = data_closes[exit_idx] 
            exit_cost = data_closes[exit_idx] * ((
                (spread_pct.get_taker()/100/2) + (slippage_pct.get_taker()/100)) if cost else 0)

            re_ls.append((idx,((exit-exit_cost)-(entry+entry_cost) 
                if signals[i][1] else (entry-entry_cost)-(exit+exit_cost))/entry))

        re_ls = np.asarray(re_ls)
        re_ls = re_ls[re_ls[:, 0].argsort()]

        re_ls = re_ls[:, 1]*100
        return re_ls

    for n_s in range(n_sims):
        for _at in range(frag_attps):
            try:
                test_ls = placement()
                calc_stats(test_ls)

                if (len(test)-1)*len(trades) < total_data and len(test_ls) > 0:
                    test.append(test_ls)
                break
            except exception.DataFragError:
                logger.debug(
                    'Permutation test data fragmentation, '+
                    ('simulation deleted' if _at >= frag_attps-1 else 'trying again'))

        if n_s % skip == 0 or n_s+1 >= n_sims:
            load_prgs._step = n_s-(0 if n_s == n_sims-1 else 1)
            load_prgs.next()

    dtype = [(k, type(v)) for k, v in stats[0].items()]
    stats_carr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

    returned = ({'data':test, 'stats':stats_carr}, 'Permutation test')
    if not prnt: return returned

    last_result = stats_carr['last_result']
    sharp_result = stats_carr['sharpe_ratio']
    profitf_result = stats_carr['profit_fact']

    text = {
        f"{_cm.__COLORS['CYAN']}Z-score{_cm.__COLORS['RESET']}":[''],
        'Z return':[round(z_s:=z_score(last_result, last_result[0]), 3), 
            _cm.__COLORS['GREEN'] if z_s > 0 else _cm.__COLORS['RED'] if z_s < 0 else ''],
        'Z profit fact':[round(z_spf:=z_score(profitf_result, profitf_result[0]), 3), 
            _cm.__COLORS['GREEN'] if z_spf > 0 else _cm.__COLORS['RED'] if z_spf < 0 else ''],
        'Z sharpe ratio':[round(z_ssp:=z_score(sharp_result, sharp_result[0]), 3), 
            _cm.__COLORS['GREEN'] if z_ssp > 0 else _cm.__COLORS['RED'] if z_ssp < 0 else ''],

        f"\n{_cm.__COLORS['CYAN']}P-value{_cm.__COLORS['RESET']}":[''],
        'P return':[round(p_v:=np.mean(last_result >= last_result[0]), 3), 
            _cm.__COLORS['RED'] if p_v > 0.05 else _cm.__COLORS['GREEN']],
        'P profit fact':[round(p_vpf:=np.mean(profitf_result >= profitf_result[0]), 3), 
            _cm.__COLORS['RED'] if p_vpf > 0.05 else _cm.__COLORS['GREEN']],
        'P sharpe ratio':[round(p_vsp:=np.mean(sharp_result >= sharp_result[0]), 3), 
            _cm.__COLORS['RED'] if p_vsp > 0.05 else _cm.__COLORS['GREEN']],

        f"\n{_cm.__COLORS['CYAN']}Others{_cm.__COLORS['RESET']}":[''],
        'percentile rank':[str(round(percentile_rank(last_result[1:], last_result[0], form='mean')*100, 2))+'%', 
            _cm.__COLORS['GREEN']],
    }
    text = utils.statistics_format(text, f"---Statistics of permutation test---")

    text = text if _cm.dots else text.replace('.', ',')

    print(text)
    return returned

def correlation(names:Sequence[str|int|None], col:str|None = None, 
                method:str|None = None, prnt:bool = True) -> pd.DataFrame:
    """
    Correlation

    Measures correlation with DataFrame.corr.

    Args:
        names (Sequence[str|int|None]): Backtest names which measure correlation.
        col (str|None, optional): Column used to measure correlation, 
            only 'profit' and 'profitPer' are supported, None = 'profitPer'.
        method (str|None, optional): Correlation method: 'pearson', 
            'kendall', 'spearman'. None = 'pearson'.
        prnt (bool, optional): Print the result to the console.

    Returns:
        DataFrame: Correlation.
    """

    # Exceptions.
    if col and col not in ('profit', 'profitPer'):
        raise exception.StatsError(
            "'col' only 'profit', 'profitPer' or None is supported.")
    elif isinstance(names, str) or len(names) <= 1:
        raise exception.StatsError(
            "'names' can only be a Sequence not str, with len > 1."
        )
    elif method and method.lower() not in ('pearson', 'kendall', 'spearman'):
        raise exception.StatsError(
            "'method' only 'pearson', 'kendall', 'spearman' or None is supported.")

    trades = _cm.__get_dtrades(names=names)

    for v in trades.values():
        if v.empty:
            logger.warning('Trades not loaded.'); return pd.DataFrame()
        elif not 'positionDate' in v.columns:
            logger.warning('No closed trades.'); return pd.DataFrame()

    daily_profit = {
        k: v.groupby('positionDate')[col or 'profitPer'].sum().cumsum()
        for k, v in trades.items()
    }

    returns = pd.concat(
        daily_profit, 
        axis=1, 
        join='outer').sort_index().ffill().pct_change().dropna()
    result = returns.corr(method=method.lower() if method else 'pearson') # pyrefly: ignore

    if prnt: print(result)
    return result

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
    elif data_icon is not None and type(data_icon) != str: 
        raise exception.StatsError('Icon bad type.')
    elif data_interval is not None and type(data_interval) != str: 
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

def stats_trades(data:bool = False, name:Sequence[str|int|None]|str|int|None = None, 
                 prnt:bool = True) -> str|None:
    """
    Trades Statistics.

    Statistics of the results.

    Args:
        data (bool, optional): If True, `stats_icon` is also returned.
        name (Sequence[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        prnt (bool, optional): If True, prints the statistics. If False, returns 
            the statistics as a string. Default is True.

    Info:
        - Trades: The number of operations performed.
        - Win trades: The number of winners operations.
        - Loss trades: The number of losers operations.
        - Op years: Years operated from the first to the last.
        - Return: The total equity earned.
        - Profit: The total amount earned.
        - Gross earnings: Only the profits.
        - Gross losses: Only the losses.
        - Commission cost: Total commissions.
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
        - Average return abs: The average percentage earned absolute values.
        - Average profit abs: The average profit earned absolute values.
        - Profit fact: The profit factor is calculated by dividing 
                total profits by total losses.
        - Gain loss diff: The difference between total gains and total losses.
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
        logger.warning('Trades not loaded.'); return
    elif (not 'profitPer' in trades.columns) or np.isnan(trades['profitPer'].mean()):  
        logger.warning('No closed trades.'); return

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

        'Win trades':[(trades['profit'] > 0).sum(),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['GREEN']],

        'Loss trades':[(trades['profit'] <= 0).sum(),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['RED']],

        'Op years':[utils.round_r(op_years, 2), _cm.__COLORS['CYAN']],

        'Return':[str(_return:=utils.round_r((_cm.c_tf(trades.loc[:, 'multiplier'].prod())-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if _return > 0 else _cm.__COLORS['RED'],],

        'Profit':[str(_profit:=utils.round_r(np.nansum(trades['profit'].to_numpy()),2)),
                _cm.__COLORS['GREEN'] if _profit > 0 else _cm.__COLORS['RED'],],

        'Gross earnings':[utils.round_r((trades['profit'][trades['profit']>0].sum()
                           if not pd.isna(trades['profit']).all() else 0), 4),
                        _cm.__COLORS['GREEN']],

        'Gross losses':[utils.round_r(abs(trades['profit'][trades['profit']<=0].sum())
                           if not pd.isna(trades['profit']).all() else 0, 4),
                        _cm.__COLORS['RED']],

        'Commission cost':[utils.round_r(float(trades['commission'].sum()), 4), _cm.__COLORS['RED']],

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
                  _cm.__COLORS['GREEN'] if _return_ann > 0 else _cm.__COLORS['RED'],],

        'Profit ann':[str(_profit_ann:=utils.round_r(ann_profit.mean(),2)),
                  _cm.__COLORS['GREEN'] if _profit_ann > 0 else _cm.__COLORS['RED'],],

        'Return ann vol':[utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1)
                                        *np.sqrt(trades_data['d_year_days']), 2),
                          _cm.__COLORS['YELLOW']],

        'Profit ann vol':[utils.round_r(np.std(diary_profit.dropna(),ddof=1)
                                    *np.sqrt(trades_data['d_year_days']), 2),
                        _cm.__COLORS['YELLOW']],

        'Average ratio':[utils.round_r(average_ratio(trades), 2),
                        _cm.__COLORS['YELLOW'],],

        'Average return':[str(round(
                trades.loc[:, 'profitPer'].dropna().to_numpy().mean(),2))+'%',
            _cm.__COLORS['YELLOW'],],

        'Average profit':[str(round(trades.loc[:, 'profit'].mean(),2)),
                    _cm.__COLORS['YELLOW'],],

        'Average return abs':[str(round(
                trades.loc[:, 'profitPer'].dropna().abs().to_numpy().mean(),2))+'%',
            _cm.__COLORS['YELLOW'],],

        'Average profit abs':[str(round(trades.loc[:, 'profit'].abs().mean(),2)),
                    _cm.__COLORS['YELLOW'],],

        'Profit fact':[_profit_fact:=utils.round_r(profit_fact(trades.loc[:, 'profit']), 3),
                _cm.__COLORS['GREEN'] if _profit_fact > 1 else _cm.__COLORS['RED'],],

        'Gain loss diff':[_gain_diff:=utils.round_r(gain_loss_diff(trades.loc[:, 'profit']), 3),
                _cm.__COLORS['GREEN'] if _gain_diff > 0 else _cm.__COLORS['RED'],],

        'Return diary std':[(_return_std:=utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1), 2)),
                    _cm.__COLORS['YELLOW'] if _return_std > 1 else _cm.__COLORS['GREEN'],],

        'Profit diary std':[(_profit_std:=utils.round_r(np.std(diary_profit.dropna(),ddof=1), 2)),
                      _cm.__COLORS['YELLOW'] if _profit_std > 1 else _cm.__COLORS['GREEN'],],

        'Math hope':[_math_hope:=round(math_hope(trades.loc[:, 'profit']), 3),
            _cm.__COLORS['GREEN'] if _math_hope > 0 else _cm.__COLORS['RED'],],

        'Math hope r':[_math_hope_r:=round(
                math_hope_relative(trades, trades.loc[:, 'profitPer']), 3),
            _cm.__COLORS['GREEN'] if _math_hope_r > 0 else _cm.__COLORS['RED'],],

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

        'Expectation':[utils.round_r(expectation(trades.loc[:, 'profitPer']), 2)],

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

        'Max consecutive loss':[abs(trades_csct.min()), # pyrefly: ignore
                                _cm.__COLORS['RED']],

        'Max losing streak':[abs(trades_streak.min())],

        'Max drawdown':[str(round(
            max_drawdown(multiplier_cumprod)*100, 1)) + '%'],

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

        'Winnings':[str(round(winnings(trades.loc[:, 'profitPer'])*100, 1)) + '%'],

        }, f"---Statistics of '{trades_data['name']}'---")

    text = text if _cm.dots else text.replace('.', ',')
    if data: 
        text += (stats_icon(False) or '')

    if prnt: print(text)
    else: return text

def trades_op_years(trades_date:pd.Series, day_width:float, year_days:int) -> float:
    """
    Trades operated years

    Returns the number of years operated.

    Args:
        trades_date (Series): 'Series' with the dates.
        day_width (float): Width of each day.
        year_days (int): Operated days per year.

    Returns:
        float: Years operated on float.
    """

    return abs(
        (trades_date.iloc[-1] - trades_date.iloc[0])/
        (day_width*year_days))

def trades_duration(position_date:pd.Series, date:pd.Series, 
                    day_width:float) -> pd.Series:
    """
    Trades duration in days

    Returns the duration of trades in days.

    Args:
        position_date (Series): 'Series' with closing date for each trade.
        date (Series): 'Series' with the dates.
        day_width (float): Width of each day.

    Returns:
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

    Returns:
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

    Returns:
        Series: Day groups.
    """

    if op_years <= 0:
        raise exception.StatsError("'op_years' cannot be less or equal than 0.")
    elif year_days <= 0:
        raise exception.StatsError("'year_days' cannot be less or equal than 0.")

    return ((trades_date - trades_date.iloc[0]) / 
                (trades_date.iloc[-1] - trades_date.iloc[0]) * 
                op_years*year_days).astype(int)

def earnings_intime(names:Sequence[str|int|None]|str|int|None = None,
                    in_days:float = 365, profit:bool = True, 
                    prnt:bool = True) -> tuple[pd.Series, str]|None:
    """
    Earnings in time

    Statistics of earnings each 'in_days' amount of days.

    Note:
        This function can send a text to the console that is too long; 
        in that case, it is possible that not all of it will be printed.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        in_days (float, optional): Number of days per group, cannot be less than 1.
        profit (bool, optional): If it is True, it is calculated with 'profit', 
            otherwise with 'profitPer'.
        prnt (bool, optional): Print the statistics or return them.

    Returns:
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

    indays_profit = trades.groupby(trades['day'] // in_days)[
        'profit' if profit else 'profitPer'].sum()

    data = {}
    for i,v in enumerate(indays_profit):
        value = {f'Group: {i+1}':[round(v, 1), 
            _cm.__COLORS['GREEN'] if round(v, 1) > 0 else _cm.__COLORS['RED']]}
        data.update(value)

    data.update({
        'Winrate':[
        str(round(winnings(indays_profit)*100)) + '%',
        _cm.__COLORS['CYAN']],
        'Average':[round(indays_profit.mean(), 1)]
    })
    text = utils.statistics_format(data, f"---Earnings per {in_days} days---")

    text = text if _cm.dots else text.replace('.', ',')
    if prnt: print(text) 

    return (indays_profit, text)
