"""
Main.
----
Here are all the main functions of BackPy where:\n
the graphs are displayed, the strategies are processed and the data is loaded.\n
Functions:
---
>>> load_yfinance_data
>>> load_data
>>> run
>>> plot
>>> plot_strategy
>>> icon_stats
>>> trades_stats

Hidden variables:
--
>>> _init_funds # Initial funds.
>>> __data_interval # Data interval.
>>> __data_icon # Data icon.
>>> __data # Saved data.
>>> __trades # Saved trades.
"""

import matplotlib.pyplot
import matplotlib as mpl

import pandas as pd
import numpy as np

import types
from time import time

from . import utils
from . import strategy
from . import exception

__data_interval = None
__data_icon = None
__data = None

__trades = pd.DataFrame()
_init_funds = 0

def load_yfinance_data(tickers:str = any, start:str = None, end:str = None, interval:str = '1d', statistics:bool = True, progress:bool = True) -> None:
    """
    Load yfinance data.
    ----
    Load all data using the yfinance module.\n
    Parameters:
    --
    >>> tickers:str = any
    >>> start:str = None
    >>> end:str = None
    >>> interval:str = '1d'
    >>> statistics:bool = True
    >>> progress:bool = True
    \n
    tickers: \n
    \tString of ticker to download.\n
    start: \n
    \tDownload start date string (YYYY-MM-DD) or _datetime, inclusive.\n
    \tDefault is 99 years ago.\n
    end: \n
    \tDownload end date string (YYYY-MM-DD) or _datetime, exclusive.\n
    \tDefault is now.\n
    interval: \n
    \tValid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.\n
    \tIntraday data cannot extend last 60 days.\n
    statistics: \n
    \tPrint statistics of the downloaded data.\n
    progress: \n
    \tProgress bar and timer.\n
    Example:
    --
    >>> load_yfinance_data(
    >>> tickers="BTC-USD", 
    >>> start="2023-02-01", 
    >>> end="2024-02-01", 
    >>> interval="1d", 
    >>> statistics=False, 
    >>> progress=True)
    """
    global __data_interval, __data_icon, __data

    try:
        import yfinance as yf

        t = time() if progress else None

        yf.set_tz_cache_location('.\yfinance_cache')
        __data = yf.download(tickers, start=start, end=end, interval=interval, progress=progress)
        if __data.empty: raise exception.YfinanceError('The symbol does not exist.')

        if progress: print('DataTimer:',round(time()-t,2))
    except ModuleNotFoundError: raise exception.YfinanceError('Yfinance is not downloaded.')
    except: raise exception.YfinanceError('Yfinance parameters error.')
    
    __data_interval = interval
    __data_icon = tickers

    if statistics: stats_icon(prnt=True)

def load_data(data:pd.DataFrame = any, icon:str = None, interval:str = None, statistics:bool = True) -> None: 
    """
    Load any data.
    ----
    Load data.\n
    Parameters:
    Parameters:
    --
    >>> data:str = any
    >>> icon:str = None
    >>> interval:str = None
    >>> statistics:bool = True
    \n
    data: \n
    \tpd.Dataframe with all the data.\n
    \tYou need to have these columns:\n
    \t['Open', 'High', 'Low', 'Close', 'Volume']\n
    icon: \n
    \tString of the data icon.\n
    interval: \n
    \tString of the data interval.\n
    statistics: \n
    \tPrint statistics of the loaded data.\n
    """
    global __data, __data_icon, __data_interval
    if not all(col in data.columns.to_list() for col in ['Open', 'High', 'Low', 'Close', 'Volume']): raise exception.DataError("Some columns are missing columns: ['Open', 'High', 'Low', 'Close', 'Volume']")

    __data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    __data.index.name = 'Date'
    __data_icon = icon
    __data_interval = interval

    if statistics: stats_icon(prnt=True)

def run(strategy_class:'strategy.StrategyClass' = any, initial_funds:int = 10000, commission:float = 0, prnt:bool = True, progress:bool = True, beta_fastm:bool = False) -> str:
    """
    Run your strategy.
    ----
    Run your strategy.\n
    Parameters:
    --
    >>> strategy_class:'strategy.StrategyClass' = any
    >>> initial_funds:int = 10000
    >>> commission:int = 0
    >>> prnt:bool = True
    >>> progress:bool = True
    >>> beta_fastm:bool = False
    \n
    strategy_class: \n
    \tA class that is inherited from StrategyClass\n
    \twhere you create your strategy in the next function.\n
    initial_funds:\n
    \tIt is the initial amount you start with.\n
    \tIt is used for some statistics.\n
    commission:\n
    \tIt is the commission in percentage for each trade.\n
    \tIt is used for some statistics.\n
    prnt: \n
    \tIf it is true, trades_stats will be printed.\n
    \tIf it is false, an string will be returned.\n
    progress: \n
    \tProgress bar and timer.\n
    beta_fastm: \n
    \tEach sail's loop is calculated differently and may be faster than normal mode.\n
    \tThis mode does not contain a loading bar.\n
    \tFunction not yet finished.\n

    Alert:\n
    If strategy_class.next() prints something to the console the loading bar will not work as expected.\n
    Recommend: 
    >>> progress = False

    Example:
    --
    >>> run(
    >>> strategy_class=FristStrategy,
    >>> prnt=True,
    >>> progress=True)

    FristStrategy:
    >>> class FristStrategy(backpy.StrategyClass)
    """
    global __trades, _init_funds

    if __data is None: raise exception.RunError('Data not loaded.')
    if initial_funds < 0: raise exception.RunError("'initial_funds' cannot be less than 0.")
    if commission < 0: raise exception.RunError("'commission' cannot be less than 0.")
    if not __trades.empty: __trades = pd.DataFrame()

    _init_funds = initial_funds

    act_trades = pd.DataFrame()
    t = time(); step_t = time()
    
    if not beta_fastm:
        for f in range(__data.shape[0]):
            if progress: utils.load_bar(__data.shape[0], f+1, f'/ Step time: {round(time()-step_t,3)}'); step_t = time()

            instance = strategy_class(__data[:f+1], __trades, act_trades, commission, initial_funds)
            act_trades, __trades = instance._StrategyClass__before()
    else: 
        def m_loop(x):
            global __trades; nonlocal act_trades

            instance = strategy_class(__data.loc[:x.name], __trades, act_trades, commission, initial_funds)
            act_trades, __trades = instance._StrategyClass__before()
        __data.apply(m_loop, axis=1)

    if progress: print('\nRunTimer:',round(time()-t,2))
    
    if not act_trades.empty: __trades = pd.concat([__trades, act_trades.dropna(axis=1, how='all')], ignore_index=True)
    
    try: 
        return stats_trades(prnt=prnt)
    except exception.StatsError: pass

def plot(log:bool = False, progress:bool = True, block:bool = True) -> None:
    """
    Plot graph with trades.
    ----
    Plot your data showing the trades made.\n
    Parameters:
    --
    >>> log:bool = Flase
    >>> progress:bool = True
    >>> block:bool = True
    \n
    log: \n
    \tPlot your data using logarithmic scale.\n
    progress: \n
    \tProgress bar and timer.\n
    """

    if __data is None: raise exception.PlotError('Data not loaded.')
    if progress: utils.load_bar(9, 1);t = time()

    mpl.pyplot.style.use('ggplot')
    fig = mpl.pyplot.figure(figsize=(16,8))
    ax1 = mpl.pyplot.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = mpl.pyplot.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1); ax2.set_yticks([])
    
    if log: ax1.semilogy(__data['Close'], alpha=0); ax2.semilogy(alpha=0)
    if progress: utils.load_bar(9, 2)

    fig.tight_layout(); fig.subplots_adjust(hspace=0)

    width = (mpl.dates.date2num(__data.index).max()-mpl.dates.date2num(__data.index).min())/__data.shape[0]

    if progress: utils.load_bar(9, 3)

    candle_data = __data.copy(); candle_data.index = mpl.dates.date2num(__data.index)
    utils.candles_plot(ax1, candle_data, width*0.9)

    if progress: utils.load_bar(9, 4)

    ax2.bar(mpl.dates.date2num(__data.index), round(__data['Volume'],0), width=width)

    if progress: utils.load_bar(9, 5)

    def t_scatter(function = any, color:str = any, marker:str = any, date_label:str = 'Date'):
        if not isinstance(function, types.FunctionType): raise TypeError("'function' is not a function.")
        elif not date_label in __trades.columns: return
        ax1.scatter(__trades[date_label].apply(lambda x: mpl.dates.date2num(x) if x != np.nan else None) if not __trades.empty else [] , __trades.apply(function, axis=1), c=color, s = 30, marker=marker)

    t_scatter(lambda row: row['PositionClose'] if row['ProfitPer'] > 0 else None, 'gold', 'x', 'PositionDate')
    t_scatter(lambda row: row['PositionClose'] if row['ProfitPer'] <= 0 else None, 'purple', 'x', 'PositionDate')
    if progress: utils.load_bar(9, 6)

    t_scatter(lambda row: row['TakeProfit'] if "TakeProfit" in row.index else None, 'y', '2')
    t_scatter(lambda row: row['StopLoss'] if "StopLoss" in row.index else None, 'y', '1')
    if progress: utils.load_bar(9, 7)

    t_scatter(lambda row: (row['Low'] - (row['High'] - row['Low']) / 2 if row['Type'] else None),'g','^')
    t_scatter(lambda row: (row['High'] + (row['High'] - row['Low']) / 2 if not row['Type'] else None),'r','v')
    if progress: utils.load_bar(9, 8)

    date_format = mpl.dates.DateFormatter('%H:%M %d-%m-%Y')
    ax1.xaxis.set_major_formatter(date_format); fig.autofmt_xdate()
    mpl.pyplot.gcf().canvas.manager.set_window_title(f'Back testing: \'{__data_icon}\' {".".join(str(val) for val in [__data.index[0].day,__data.index[0].month,__data.index[0].year])+"~"+".".join(str(val) for val in [__data.index[-1].day,__data.index[-1].month,__data.index[-1].year]) if isinstance(__data.index[0], pd.Timestamp) else ""}')

    if progress: utils.load_bar(9, 9); print('\nPlotTimer:',round(time()-t,2))
    mpl.pyplot.show(block=block)

def plot_strategy(log:bool = False, view:str = 'p/w/r/n', block:bool = True) -> None:
    """
    Plot strategy statistics.
    ----
    Plot your strategy statistics.\n
    View available graphics:\n
    - 'p' = Profit graph.
    - 'r' = Return graph.
    - 'w' = Winnings graph.
    Parameters:
    --
    >>> log:bool = Flase
    >>> block:bool = True
    >>> view:str = 'p/w/r/n'
    \n
    log: \n
    \tPlot your data using logarithmic scale.\n
    view: \n
    \tPlot your data the way you prefer.\n
    \tThere are 4 shapes available and they all take up the entire window.\n
    """

    if __trades.empty: raise exception.StatsError('Trades not loaded.')
    if not 'Profit' in __trades.columns:  raise exception.StatsError('There is no data to see.')
    view = view.lower().strip().split('/')
    view = [i for i in view if i in ('p','w','r')]
    if len(view) > 4 or len(view) < 1: raise exception.StatsError("'view' allowed format: 's/s/s/s' where s is the name of the graph.\nAvailable graphics: 'p','w','r'")

    mpl.pyplot.style.use('ggplot')
    fig = mpl.pyplot.figure(figsize=(16,8))

    loc = [(0,0), (3,0), (3,1), (0,1)]; ax = None

    for i,v in enumerate(view):
        match len(view):
            case 1: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[0], rowspan=6, colspan=2)
            case 2: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], rowspan=3, colspan=2, sharex=ax, sharey=ax)
            case 3: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], rowspan=3, colspan=2 if i==0 else 1, sharex=ax)
            case 4: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], rowspan=3, colspan=1, sharex=ax, sharey=ax) 

        match v:
            case 'p':
                ax.plot(__trades.index,__trades['Profit'].cumsum(), c='black', label='Profit.')
                if log: ax.set_yscale('symlog')
            case 'w':
                ax.plot(__trades.index,(__trades['ProfitPer'].apply(lambda row: 1 if row>0 else -1)).cumsum(), c='black', label='Winnings.')
            case 'r':
                ax.plot(__trades.index,__trades['ProfitPer'].cumsum(), c='black', label='Return.')
                if log: ax.set_yscale('symlog')
            case _: pass
        ax.legend(loc='upper left')

    mpl.pyplot.xticks([])
    mpl.pyplot.gcf().canvas.manager.set_window_title(f'Strategy statistics.')
    mpl.pyplot.show(block=block)

def stats_icon(prnt:bool = True) -> str:
    """
    Icon statistics.
    ----
    Statistics of the uploaded data.\n
    Parameters:
    --
    >>> prnt:bool = True
    \n
    prnt: \n
    \tIf it is true, statistics will be printed.\n
    \tIf it is false, an string will be returned.\n
    """
    if __data is None: raise exception.StatsError('Data not loaded.')

    data_s = f"""
Statistics of {__data_icon}:
----
Last price: {utils.round_r(__data['Close'].iloc[-1],2)}
Maximum price: {utils.round_r(__data['High'].max(),2)}
Minimum price: {utils.round_r(__data['Low'].min(),2)}
Maximum volume: {__data['Volume'].max()}
Sample size: {len(__data.index)}
Standard deviation: {utils.round_r(__data['Close'].std(),2)}
Average price: {utils.round_r(__data['Close'].mean(),2)}
Average volume: {utils.round_r(__data['Volume'].mean(),2)}
----
{".".join(str(val) for val in [__data.index[0].day,__data.index[0].month,__data.index[0].year])+"~"+".".join(str(val) for val in [__data.index[-1].day,__data.index[-1].month,__data.index[-1].year]) if isinstance(__data.index[0], pd.Timestamp) else ""} ~ {__data_interval} ~ {__data_icon}
    """
    if prnt:print(data_s) 
    else: return data_s

def stats_trades(data:bool = False, prnt:bool = True):
    """
    Trades statistics.
    ----
    Statistics of the results.\n
    Parameters:
    --
    >>> prnt:bool = True
    \n
    prnt: \n
    \tIf it is true, statistics will be printed.\n
    \tIf it is false, an string will be returned.\n
    """
    if __trades.empty: raise exception.StatsError('Trades not loaded.')
    elif not 'ProfitPer' in __trades.columns:  raise exception.StatsError('There is no data to see.')
    elif np.isnan(__trades['ProfitPer'].mean()): raise exception.StatsError('There is no data to see.') 

    data_s = f"""
Statistics of strategy.
----
Trades: {len(__trades.index)}

Return: {utils.round_r(__trades['ProfitPer'].sum(),2)}%
Average return: {utils.round_r(__trades['ProfitPer'].mean(),2)}%
Average ratio: {utils.round_r((abs(__trades['Close']-__trades['TakeProfit']) / abs(__trades['Close']-__trades['StopLoss'])).mean() if not __trades['TakeProfit'].isnull().all() and not __trades['StopLoss'].isnull().all() else 0, 2)}

Profit: {utils.round_r(__trades['Profit'].sum(),2)}
Profit fact: {utils.round_r((__trades['Profit']>0).sum()/(__trades['Profit']<=0).sum(),2) if (__trades['Profit']>0).sum() > 0 and (__trades['Profit']<=0).sum() > 0 and not pd.isna(__trades['Profit']).all() else 0}
Duration ratio: {utils.round_r(__trades['PositionDate'].apply(lambda x: x.timestamp() if not pd.isna(x) else 0).mean()/__trades['PositionDate'].apply(lambda x: x.timestamp() if not pd.isna(x) else 0).sum(),2) if not __trades['PositionDate'].isnull().all() else np.nan}

Max drawdown: {round(utils.max_drawdown(__trades['Profit'].dropna().cumsum()+_init_funds),1)}%
Long exposure: {round((__trades['Type']==1).sum()/__trades['Type'].count()*100,1)}%
Winnings: {round((__trades['ProfitPer']>0).sum()/__trades['ProfitPer'].count()*100,1) if not ((__trades['ProfitPer']>0).sum() == 0 or __trades['ProfitPer'].count() == 0) else 0}%
----
    """
    if data: data_s += stats_icon(False)
    
    if prnt: print(data_s)
    else: return data_s
