"""
Main.
----
Here are all the main functions of BackPy where:
the graphs are displayed, the strategies are processed and the data is loaded.

Functions:
---
>>> load_yfinance_data
>>> load_data
>>> run
>>> plot
>>> plot_strategy
>>> icon_stats
>>> trades_stats

Hidden functions:
---
>>> _data_info

Variables:
--
>>> alert = True # Show alerts in the console.

Hidden variables:
--
>>> _init_funds # Initial funds.
>>> __data_interval # Data interval.
>>> __data_width # Data index width.
>>> __data_icon # Data icon.
>>> __data # Saved data.
>>> __trades # Saved trades.
"""

import matplotlib.pyplot
import matplotlib as mpl

import pandas as pd
import numpy as np

from time import time

from . import utils
from . import strategy
from . import exception

alert = True

__data_interval = None
__data_width = None
__data_icon = None
__data = None

__trades = pd.DataFrame()
_init_funds = 0

def _data_info() -> tuple:
    """
    Data info.
    ----
    Returns all 'data' variables except '__data'.

    Order:
    --
    >>> (__data_interval, __data_width, __data_icon)
    """
    return __data_interval, __data_width, __data_icon

def load_yfinance_data(tickers:str = any, 
                       start:str = None, end:str = None, interval:str = '1d', 
                       statistics:bool = True, progress:bool = True) -> None:
    """
    Load yfinance data.
    ----
    Load all data using the yfinance module.

    Parameters:
    --
    >>> tickers:str = any
    >>> start:str = None
    >>> end:str = None
    >>> interval:str = '1d'
    >>> statistics:bool = True
    >>> progress:bool = True
    
    tickers:
      - String of ticker to download.

    start:
      - Download start date string (YYYY-MM-DD).
      - Default is 99 years ago.

    end:
      - Download end date string (YYYY-MM-DD).
      - Default is now.

    interval:
      - Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.
      - Intraday data cannot extend last 60 days.

    statistics:
      - Print statistics of the downloaded data.

    progress:
      - Progress bar and timer.

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
    global __data_interval, __data_width, __data_icon, __data

    try:
        import yfinance as yf

        t = time() if progress else None

        yf.set_tz_cache_location('.\yfinance_cache')
        __data = yf.download(tickers, start=start, end=end, 
                             interval=interval, progress=progress)
        
        if __data.empty: 
            raise exception.YfinanceError('The symbol does not exist.')
        
        __data.index = mpl.dates.date2num(__data.index)
        __data_width = utils.calc_width(__data.index)

        if progress: 
            print('DataTimer:',round(time()-t,2))
    
    except ModuleNotFoundError: 
        raise exception.YfinanceError('Yfinance is not installed.')
    except: 
        raise exception.YfinanceError('Yfinance parameters error.')
    
    __data_interval = interval.strip()
    __data_icon = tickers.strip()

    if statistics: stats_icon(prnt=True)

def load_data(data:pd.DataFrame = any, icon:str = None, 
              interval:str = None, statistics:bool = True) -> None: 
    """
    Load any data.
    ----
    Load data.

    Parameters:
    --
    >>> data:str = any
    >>> icon:str = None
    >>> interval:str = None
    >>> statistics:bool = True
    
    data:
      - pd.Dataframe with all the data.
      - You need to have these columns:
       ['Open', 'High', 'Low', 'Close', 'Volume']

    icon:
      - String of the data icon.

    interval:
      - String of the data interval.

    statistics:
      - Print statistics of the loaded data.
    """
    global __data_interval, __data_width, __data_icon, __data
    # Exceptions.
    if not all(
        col in data.columns.to_list() 
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']): 
        
        raise exception.DataError(
            utils.text_fix("""
            Some columns are missing columns: 
            ['Open', 'High', 'Low', 'Close', 'Volume']
            """, newline_exclude=True))

    __data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    __data.index.name = 'Date'
    __data.index = utils.correct_index(__data.index)
    __data_width = utils.calc_width(__data.index)

    __data_icon = icon.strip()
    __data_interval = interval.strip()

    if statistics: stats_icon(prnt=True)

def run(strategy_class:'strategy.StrategyClass' = any, 
        initial_funds:int = 10000, commission:float = 0, 
        prnt:bool = True, progress:bool = True, fast_mode:bool = False) -> str:
    """
    Run your strategy.
    ----
    Run your strategy.

    Parameters:
    --
    >>> strategy_class:'strategy.StrategyClass' = any
    >>> initial_funds:int = 10000
    >>> commission:int = 0
    >>> prnt:bool = True
    >>> progress:bool = True
    >>> fast_mode:bool = False
    
    strategy_class:
      - A class that is inherited from StrategyClass
       where you create your strategy in the next function.

    initial_funds:
      - It is the initial amount you start with.
      - It is used for some statistics.

    commission:
      - It is the commission in percentage for each trade.
      - It is used for some statistics.

    prnt:
      - If it is true, trades_stats will be printed.
      - If it is false, an string will be returned.

    progress:
      - Progress bar and timer.

    fast_mode:
      - Each sail's loop is calculated differently and 
       may be faster than normal mode.
      - This mode does not contain a loading bar.
      - Function not yet finished.

    Alert:
    --
    If strategy_class.next() prints something to the 
     console the loading bar will not work as expected.

    Example:
    --
    >>> run(
    >>> strategy_class=FristStrategy,
    >>> prnt=True,
    >>> progress=True)

    FristStrategy:
    >>> class FristStrategy(backpy.StrategyClass)
    """
    global __trades, _init_funds, __data_width
    # Exceptions.
    if __data is None: 
        raise exception.RunError('Data not loaded.')
    elif initial_funds < 0: 
        raise exception.RunError("'initial_funds' cannot be less than 0.")
    elif commission < 0: 
        raise exception.RunError("'commission' cannot be less than 0.")
    # Corrections.
    __data.index = utils.correct_index(__data.index)
    __data_width = utils.calc_width(__data.index, True)

    _init_funds = initial_funds
    instance = strategy_class(commission=commission, init_funds=initial_funds)
    t = time()
    
    if fast_mode:
        __data.apply(
            lambda x: instance._StrategyClass__before(
                data=__data.iloc[:x.name]), 
            axis=1)
        
        act_trades = instance._StrategyClass__trades_ac
        __trades = instance._StrategyClass__trades_cl

        if progress:
            print('\nRunTimer:',round(time()-t,2))
    else: 
        step_t = time()

        for f in range(1, __data.shape[0]+2):
            if progress and __data.shape[0] >= f:
                utils.load_bar(size=__data.shape[0], step=f) 
                print(f'/ Step time: {round(time()-step_t,3)}', end='')
                step_t = time()
            elif progress:
                print('\nRunTimer:',round(time()-t,2))
                break

            instance._StrategyClass__before(data=__data.iloc[:f])
        
        act_trades = instance._StrategyClass__trades_ac
        __trades = instance._StrategyClass__trades_cl
    
    if not act_trades.empty: __trades = pd.concat([
        __trades, act_trades.dropna(axis=1, how='all')
        ], ignore_index=True)

    try: 
        return stats_trades(prnt=prnt)
    except exception.StatsError: pass
    
def plot(log:bool = False, progress:bool = True, 
         position:str = 'complex', block:bool = True) -> None:
    """
    Plot graph with trades.
    ----
    Plot your data showing the trades made.

    Color guide:
    - gold: 'x' = Position close.
    - green, '^' = Buy position.
    - red, 'v' = Sell position.

    Parameters:
    --
    >>> log:bool = Flase
    >>> progress:bool = True
    >>> position:str = 'complex'
    >>> block:bool = True
    
    log:
      - Plot your data using logarithmic scale.

    progress:
      - Progress bar and timer.

    position:
      - The way the positions were drawn.
      - Available options: ('complex', 'simple').
      - If left at None or 'none', positions will not be drawn.
      - The "complex" option takes a long time to process.

    block:
      - If 'True', the script execution is paused until
       all figure windows are closed. 
      - If 'False', the script continues running after displaying the figures.
    """
    # Exceptions.
    if __data is None or not type(__data) is pd.DataFrame or __data.empty: 
        raise exception.PlotError('Data not loaded.')
    elif position and not position.lower() in ('complex', 'simple', 'none'):
        raise exception.PlotError(
            f"'{position}' Not a valid option for: 'position'.")
    # Corrections.
    __data.index = utils.correct_index(__data.index)
    __data_width = utils.calc_width(__data.index, True)
    
    if progress: 
        t = time()
        utils.load_bar(size=4, step=0)

    mpl.pyplot.style.use('ggplot')
    fig = mpl.pyplot.figure(figsize=(16,8))
    ax1 = mpl.pyplot.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = mpl.pyplot.subplot2grid((6,1), (5,0), rowspan=1, 
                                  colspan=1, sharex=ax1)
    ax2.set_yticks([])
    
    if log: 
        ax1.semilogy(__data['Close'], alpha=0); ax2.semilogy(alpha=0)

    fig.tight_layout(); fig.subplots_adjust(hspace=0)

    if progress: 
        utils.load_bar(size=4, step=1)

    utils.plot_candles(ax1, __data, __data_width*0.9)

    if progress: 
        utils.load_bar(size=4, step=2)

    if __data['Volume'].max() > 0:
      ax2.fill_between(__data.index, __data['Volume'], step='mid')
      ax2.set_ylim(None, __data['Volume'].max()*1.5)

    if position and position.lower() != 'none' and not __trades.empty:
        utils.plot_position(__trades, ax1, 
                          all=True if position.lower() == 'complex' else False,
                          alpha=0.3, alpha_arrow=0.8, 
                          width_exit=lambda x: __data.index[-1]-x['Date'])
    
    if progress: 
        utils.load_bar(size=4, step=3)

    date_format = mpl.dates.DateFormatter('%H:%M %d-%m-%Y')
    ax1.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    ix_date = mpl.dates.num2date(__data.index)

    s_date = ".".join(str(val) for val in 
                    [ix_date[0].day, ix_date[0].month, 
                    ix_date[0].year])
    
    e_date = ".".join(str(val) for val in 
                    [ix_date[-1].day, ix_date[-1].month, 
                    ix_date[-1].year])
    
    mpl.pyplot.gcf().canvas.manager.set_window_title(
        f"Back testing: '{__data_icon}' {s_date}~{e_date}")

    if progress: 
        utils.load_bar(size=4, step=4)
        print('\nPlotTimer:',round(time()-t,2))

    mpl.pyplot.show(block=block)

def plot_strategy(log:bool = False, view:str = 'p/w/r/n', 
                  block:bool = True) -> None:
    """
    Plot strategy statistics.
    ----
    Plot your strategy statistics.

    View available graphics:
    - 'p' = Profit graph.
    - 'r' = Return graph.
    - 'w' = Winnings graph.

    Parameters:
    --
    >>> log:bool = Flase
    >>> view:str = 'p/w/r/n'
    >>> block:bool = True
    
    log:
      - Plot your data using logarithmic scale.

    view:
      - Plot your data the way you prefer.
      - There are 4 shapes available and they all take up the entire window.

    block:
      - If True, the script execution is paused until
       all figure windows are closed. 
      - If False, the script continues running after displaying the figures.
    """
    view = view.lower().strip().split('/')
    view = [i for i in view if i in ('p','w','r')]

    # Exceptions.
    if __trades.empty: 
        raise exception.StatsError('Trades not loaded.')
    elif not 'Profit' in __trades.columns:  
        raise exception.StatsError('There is no data to see.')
    elif len(view) > 4 or len(view) < 1: 
        raise exception.StatsError(utils.text_fix("""
            'view' allowed format: 's/s/s/s' where s is the name of the graph.
            Available graphics: 'p','w','r'.
            """, newline_exclude=True))

    mpl.pyplot.style.use('ggplot')
    fig = mpl.pyplot.figure(figsize=(16,8))

    loc = [(0,0), (3,0), (3,1), (0,1)]; ax = None

    for i,v in enumerate(view):
        match len(view):
            case 1: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[0], 
                                             rowspan=6, colspan=2)
            case 2: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], 
                                             rowspan=3, colspan=2, 
                                             sharex=ax, sharey=ax)
            case 3: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], rowspan=3, 
                                             colspan=2 if i==0 else 1, 
                                             sharex=ax)
            case 4: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], 
                                             rowspan=3, colspan=1, 
                                             sharex=ax, sharey=ax) 

        match v:
            case 'p':
                ax.plot(__trades.index,__trades['Profit'].cumsum(), 
                        c='black', label='Profit.')
                
                if log: ax.set_yscale('symlog')
            case 'w':
                ax.plot(__trades.index,
                        (__trades['ProfitPer'].apply(
                            lambda row: 1 if row>0 else -1)).cumsum(), 
                        c='black', label='Winnings.')
            case 'r':
                ax.plot(__trades.index,__trades['ProfitPer'].cumsum(), 
                        c='black', label='Return.')

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
    Statistics of the uploaded data.

    Parameters:
    --
    >>> prnt:bool = True
    
    prnt:
      - If it is true, statistics will be printed.
      - If it is false, an string will be returned.
    """
    # Exceptions.
    if __data is None: raise exception.StatsError('Data not loaded.')

    if isinstance(__data.index[0], pd.Timestamp):
        s_date = ".".join(str(val) for val in 
                        [__data.index[0].day, __data.index[0].month, 
                        __data.index[0].year])
        
        e_date = ".".join(str(val) for val in 
                        [__data.index[-1].day, __data.index[-1].month, 
                        __data.index[-1].year]
                        ) if isinstance(__data.index[0], pd.Timestamp) else ""
        
        r_date = f"{s_date}~{e_date}"
    else: r_date = ""

    data_s = utils.text_fix(f"""
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
    {r_date} ~ {__data_interval} ~ {__data_icon}""", newline_exclude=False)

    if prnt:print(data_s) 
    else: return data_s

def stats_trades(data:bool = False, prnt:bool = True) -> str:
    """
    Trades statistics.
    ----
    Statistics of the results.

    Parameters:
    --
    >>> prnt:bool = True
    
    prnt: 
      - If it is true, statistics will be printed.
      - If it is false, an string will be returned.

    Info:
    --
    Trades: 
      The number of operations performed.
    Return:
      The total percentage earned.
    Average return:
      The average percentage earned.
    Average ratio:
      The average ratio.
    Profit:
      The total amount earned.
    Profit fact:
      The profit factor is calculated by dividing total profits by total losses.
    Max drawdown:
      The biggest drawdown the 'profit' has ever had.
    Long exposure:
      What percentage of traders are long.
    Winnings:
      Percentage of operations won.
    """
    # Exceptions.
    if __trades.empty: 
        raise exception.StatsError('Trades not loaded.')
    elif not 'ProfitPer' in __trades.columns:  
        raise exception.StatsError('There is no data to see.')
    elif np.isnan(__trades['ProfitPer'].mean()): 
        raise exception.StatsError('There is no data to see.') 

    data_s = utils.text_fix(f"""
    Statistics of strategy.
    ----
    Trades: {len(__trades.index)}

    Return: {utils.round_r(__trades['ProfitPer'].sum(),2)}%
    Average return: {utils.round_r(__trades['ProfitPer'].mean(),2)}%
    Average ratio: {utils.round_r(
        (abs(__trades['Close']-__trades['TakeProfit']) / 
            abs(__trades['Close']-__trades['StopLoss'])).mean() 
        if not __trades['TakeProfit'].isnull().all() and 
        not __trades['StopLoss'].isnull().all() else 0, 2)}

    Profit: {utils.round_r(__trades['Profit'].sum(),2)}
    Profit fact: {
        utils.round_r(__trades[__trades['Profit']>0]['Profit'].sum()/
                      abs(__trades[__trades['Profit']<=0]['Profit'].sum()),2) 
        if not pd.isna(__trades['Profit']).all() and
         (__trades['Profit']>0).sum() > 0 and
          (__trades['Profit']<=0).sum() > 0 else 0}

    Max drawdown: {round(
        utils.max_drawdown(__trades['Profit'].dropna().cumsum()+
                           _init_funds)*100,1)}%
    Long exposure: {
        round((__trades['Type']==1).sum()/__trades['Type'].count()*100,1)}%
    Winnings: {round(
        (__trades['ProfitPer']>0).sum()/__trades['ProfitPer'].count()*100,1) 
        if not ((__trades['ProfitPer']>0).sum() == 0 or 
                __trades['ProfitPer'].count() == 0) else 0}%
    ----""", newline_exclude=False)
    if data: data_s += stats_icon(False)
    
    if prnt: print(data_s)
    else: return data_s
