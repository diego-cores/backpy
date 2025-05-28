"""
Stats Module.

This module contains functions to calculate different metrics.

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
    long_exposure: Calculate the percentage of 1 in the given Series.
    expectation: Calculate the expectation based on payoff.
    var_historical: Calculate the historical var.
    var_parametric: Calculate the parametric var.
    max_drawdown: Function to return the maximum drawdown from the given data.
    get_drawdowns: Calculate the drawdowns from the given.
"""

import pandas as pd
import numpy as np

def average_ratio(trades:pd.DataFrame) -> float:
    """
    Average ratio.

    Based on the take profit and stop loss 
        positions, it calculates an average ratio.

    If the 'TakeProfit' or 'StopLoss' columns are 
        not present, 0 will be returned.

    Args:
        trades (pd.DataFrame): A dataframe with these columns: 
            'TakeProfit','StopLoss','Close'.

    Returns:
        float: Average ratio.
    """

    if (
        'TakeProfit' in trades.columns and 'StopLoss' in trades.columns
        and not trades['TakeProfit'].apply(lambda x: x is None or x <= 0).all() 
        and not trades['StopLoss'].apply(lambda x: x is None or x <= 0).all()
        ):

        return ((abs(trades['Close']-trades['TakeProfit']) 
                / abs(trades['Close']-trades['StopLoss'])).mean())
    return 0

def profit_fact(profits:pd.Series) -> float:
    """
    Profit fact.

    Calculate the profit factor of the values.

    Args:
        profits (pd.Series): Returns on each operation.

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
        profits (pd.Series): Returns on each operation.

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
        trades (pd.DataFrame): A dataframe with these columns: 
            'TakeProfit','StopLoss','Close'.
        profits (pd.Series): Returns on each operation.

    Returns:
        float: Math hope relative.
    """

    return winnings(profits)*float(average_ratio(trades))-(1-winnings(profits))

def winnings(profits:pd.Series) -> float:
    """
    Winnings percentage.

    Calculate the percentage of positive numbers in the series.

    Args:
        profits (pd.Series): Returns on each operation..

    Returns:
        float: Winnings percentage.
    """

    if (not ((profits>0).sum() == 0 
        or profits.count() == 0)):

        return (profits>0).sum()/profits.count()
    return 0


def sharpe_ratio(ann_av:float, year_days:int, diary_per:pd.Series) -> float:
    """
    Sharpe ratio.

    Calculate the Sharpe ratio using the 
        returns / sqrt(days of the year) / standard deviation of the data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (pd.Series): Daily return.

    Returns:
        float: Sharpe ratio.
    """
    std_dev = np.std(diary_per.dropna(), ddof=1)
    if std_dev < 1e-2: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def sortino_ratio(ann_av:float, year_days:int, diary_per:pd.Series) -> float:
    """
    Sortino ratio.

    Calculate the Sortino ratio with a calculation similar to the 
        Sharpe ratio but only with the standard deviation of negative data.

    If the standard deviation is too close to 0, returns 0 to avoid inflated values.

    Args:
        ann_av (float): Annual returns.
        year_days (int): Operable days of the year (normally 252).
        diary_per (pd.Series): Daily return.

    Returns:
        float: Sortino ratio.
    """
    std_dev = np.std(diary_per[diary_per < 0].dropna(), ddof=1)
    if std_dev < 1e-2: return 0

    return (ann_av / np.sqrt(year_days) / std_dev)

def payoff_ratio(profits:pd.Series) -> float:
    """
    Payoff ratio.

    Calculates the payout rate using the absolute 
        mean of positive numbers/mean of negative numbers.

    Args:
        profits (pd.Series): Returns on each operation..

    Returns:
        float: Payoff ratio.
    """

    return (profits[profits > 0].dropna().mean() 
            / abs(profits[profits < 0].dropna().mean()))

def expectation(profits:pd.Series) -> float:
    """
    Expectation.

    Calculate the expectation based on payoff.

    Args:
        profits (pd.Series): Returns on each operation..

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
        types (pd.Series): Type of each operation, 1 for long, 0 for short.

    Returns:
        float: Percentages of longs.
    """

    return (types==1).sum()/types.count()

def var_historical(data:list, confidence_level:int = 95) -> float:
    """
    Var historical.

    Calculate the historical var.

    Args:
        data (list): List of data which will calculate the var.
        confidence_level (int, optional): Percentile.
    
    Returns:
        float: The historical var.
    """

    return np.sort(data)[int((100 - confidence_level) / 100 * len(data))]

def var_parametric(data:list, z_alpha:float = -1.645) -> float:
    """
    Var parametric.

    Calculate the parametric var.

    Args:
        data (list): List of data which will calculate the var.
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

def get_drawdowns(values:list) -> list:
    """
    Get drawdowns.

    Calculate the drawdowns of `values`.

    Args:
        values (pd.Series): The ordered data to calculate the drawdowns.

    Returns:
        list: The drawdowns from the given data.
    """
    if len(values) == 0:
        return 0

    max_values = np.maximum.accumulate(values)
    drawdowns = (values - max_values) / max_values

    return drawdowns
