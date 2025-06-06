"""
Main Module.

This module contains the main functions of BackPy, including data loading, 
strategy processing, and graph display.

Functions:
    load_binance_data: Loads data using the binance-connector module.
    load_yfinance_data: Loads data using the yfinance module.
    load_data: Loads user-provided data.
    run: Executes the backtesting process.
    plot: Plots your data, highlighting the trades made.
    plot_strategy: Plots statistics for your strategy.
    plot_strategy_decorator: Decorator function for the 'plot_strategy_add' function.
    plot_strategy_add: Add functions and then see them graphed with 'plot_strategy'.
    stats_icon: Shows statistics related to the financial icon.
    stats_trades: Statistics of the trades.

Hidden Functions:
    __load_binance_data: Load data from Binance using a client.
"""

from datetime import datetime

import matplotlib.pyplot
import matplotlib as mpl

import pandas as pd
import numpy as np

from time import time

from . import _commons as _cm
from . import flexdata as flx
from . import exception
from . import strategy
from . import utils
from . import stats

def __load_binance_data(client:callable, symbol:str = 'BTCUSDT', 
                        interval:str = '1d', start_time:str = None, 
                        end_time:str = None, statistics:bool = True, 
                        progress:bool = True, data_extract:bool = False) -> tuple:
    """
    Load Binance data.

    Loads data using the Binance client.

    Args:
        client (callable): Bianance client.
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str): Start date for load data in YYYY-MM-DD format.
        end_time (str): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
            the module variables will not be assigned with them.

    Returns:
        tuple: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """

    # Exceptions.
    if start_time is None or end_time is None:
        raise exception.BinanceError(
            "'start_time' and 'end_time' cannot be None.")

    if progress:
        t = time()
        size = None
        ini_time = None

    def __loop_def(st_t):
        dt = client.klines(symbol=symbol, 
                        interval=interval, 
                        startTime=st_t, 
                        endTime=end, 
                        limit=1000)

        if progress:
            nonlocal size, ini_time

            if not size:
                ini_time = dt[0][0]
                size = (end-ini_time)//(dt[-1][0]-dt[0][0])

            step_time = (end-ini_time)//size
            step = int(round((dt[-1][0]-ini_time)/step_time,0))

            text = f'| DataTimer: {utils.num_align(time()-t)} '
            utils.load_bar(size=size, step=step, text=text)

        return dt
    start = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
    if ((end:=int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)) 
        > (now:=int(datetime.now().timestamp() * 1000))):
        end = now

    client = client()
    data = utils._loop_data(
        function=__loop_def,
        bpoint=lambda x, y=None: y == int(x[0].iloc[-1]) if y else int(x[0].iloc[-1]),
        init = start,
        timeout = _cm.__binance_timeout
        ).astype(float)

    if progress:
        print(end='\n')
    
    data.columns = ['timestamp', 
                    'Open', 
                    'High', 
                    'Low', 
                    'Close', 
                    'Volume', 
                    'Close_time', 
                    'Quote_asset_volume', 
                    'Number_of_trades', 
                    'Taker_buy_base', 
                    'Taker_buy_quote', 
                    'Ignore']

    data.index = data['timestamp']
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    if data.empty: 
        raise exception.BinanceError('Data empty error.')

    data.index = mpl.dates.date2num(data.index)
    data_width = utils.calc_width(data.index)

    if statistics: stats_icon(prnt=True, 
                            data=data, 
                            data_icon=symbol.strip(),
                            data_interval=interval.strip())

    if data_extract:
        return data, data_width

    _cm.__data = data
    _cm.__data_width = data_width
    _cm.__data_icon = symbol.strip()
    _cm.__data_interval = interval.strip()
    _cm.__data_width_day = utils.calc_day(interval, data_width)
    _cm.__data_year_days = 365

def load_binance_data_futures(symbol:str = 'BTCUSDT', interval:str = '1d', 
                            start_time:str = None, end_time:str = None,
                            statistics:bool = True, progress:bool = True,
                            data_extract:bool = False) -> tuple:
    """
    Load Binance data from futures.

    Loads data using the binance-connector module from futures.

    Why this differentiation?
        Binance futures data is different from spot data, 
        so it's up to you to decide which one to use based on how you plan to trade.

    Args:
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str): Start date for load data in YYYY-MM-DD format.
        end_time (str): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
            the module variables will not be assigned with them.

    Returns:
        tuple: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """
    try:
        from binance.um_futures import UMFutures as Client

        __load_binance_data(client=Client, 
                            symbol=symbol, 
                            interval=interval, 
                            start_time=start_time, 
                            end_time=end_time, 
                            statistics=statistics, 
                            progress=progress, 
                            data_extract=data_extract)

    except ModuleNotFoundError: 
        raise exception.BinanceError('Binance futures connector is not installed.')
    except: 
        raise exception.BinanceError('Binance parameters error.')

def load_binance_data_spot(symbol:str = 'BTCUSDT', interval:str = '1d', 
                            start_time:str = None, end_time:str = None,
                            statistics:bool = True, progress:bool = True,
                            data_extract:bool = False) -> tuple:
    """
    Load Binance data from spot.

    Loads data using the binance-connector module from spot.

    Why this differentiation?
        Binance spot data is different from futures data, 
        so it's up to you to decide which one to use based on how you plan to trade.

    Args:
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str): Start date for load data in YYYY-MM-DD format.
        end_time (str): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
                        the module variables will not be assigned with them.

    Returns:
        tuple: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """
    try:
        from binance.spot import Spot as Client

        __load_binance_data(client=Client, 
                            symbol=symbol, 
                            interval=interval, 
                            start_time=start_time, 
                            end_time=end_time, 
                            statistics=statistics, 
                            progress=progress, 
                            data_extract=data_extract)

    except ModuleNotFoundError: 
        raise exception.BinanceError('Binance connector is not installed.')
    except: 
        raise exception.BinanceError('Binance parameters error.')

def load_yfinance_data(tickers:str = any, 
                       start:str = None, end:str = None, interval:str = '1d', 
                       days_op:int = 365, statistics:bool = True, 
                       progress:bool = True, data_extract:bool = False) -> tuple:
    """
    Load yfinance Data.

    Loads data using the yfinance module.

    Args:
        tickers (str): String of ticker symbols to download.
        start (str, optional): Start date for download in YYYY-MM-DD format. 
                              Default is 99 years ago.
        end (str, optional): End date for download in YYYY-MM-DD format. 
                            Default is the current date.
        interval (str, optional): Data interval. Valid values are '1m', '2m', '5m', '15m', 
                        '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', 
                        '3mo'. Intraday data cannot extend past the last 60 days.
        days_op (int, optional): Number of operable days in 1 year. This will be 
                        stored to calculate some statistics. Normal values: 365, 252.
        statistics (bool, optional): If True, prints statistics of the downloaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
                        the module variables will not be assigned with them.

    Returns:
        tuple: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """
    days_op = int(days_op)
    if days_op > 365 or days_op < 1:
        raise exception.YfinanceError(f"'days_op' cant be: '{days_op}'.")

    try:
        import yfinance as yf

        t = time() if progress else None

        yf.set_tz_cache_location('.\yfinance_cache')
        
        data = yf.download(tickers, start=start, 
                           end=end, interval=interval, 
                           progress=progress, auto_adjust=False)
        
        if data.empty: 
            raise exception.YfinanceError('The symbol does not exist.')
        
        data.columns = data.columns.droplevel(1)
        data.index = mpl.dates.date2num(data.index)
        data_width = utils.calc_width(data.index)

        if progress: 
            print('\033[F\033[{}C| DataTimer:'.format(59), utils.num_align(time()-t,2))

        if statistics: stats_icon(prnt=True, 
                                  data=data, 
                                  data_icon=tickers.strip(),
                                  data_interval=interval.strip())

        if data_extract:
            return data, data_width
        
        _cm.__data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        _cm.__data_width = data_width
        _cm.__data_icon = tickers.strip()
        _cm.__data_interval = interval.strip()
        _cm.__data_width_day = days_op
        _cm.__data_width_day = utils.calc_day(interval, data_width)

    except ModuleNotFoundError: 
        raise exception.YfinanceError('Yfinance is not installed.')
    except: 
        raise exception.YfinanceError('Yfinance parameters error.')

def load_data(data:pd.DataFrame = any, icon:str = None, 
              interval:str = None, days_op:int = 365, 
              statistics:bool = True, progress:bool = True) -> None: 
    """
    Load Any Data.

    Loads data into the system.

    Args:
        data (pd.DataFrame): DataFrame containing the data to load. Must have the 
                            following columns: ['Open', 'High', 'Low', 'Close', 
                            'Volume'].
        icon (str, optional): String representing the data icon.
        interval (str, optional): String representing the data interval.
        days_op (int, optional): Number of operable days in 1 year. This will be 
                stored to calculate some statistics. Normal values: 365, 252.
        statistics (bool): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
    """
    # Exceptions.
    if not all(
        col in data.columns.to_list() 
        for col in ['Open', 'High', 'Low', 'Close']): 
        
        raise exception.DataError(
            utils.text_fix("""
            Some columns are missing columns: 
            ['Open', 'High', 'Low', 'Close']
            """, newline_exclude=True))

    days_op = int(days_op)
    if days_op > 365 or days_op < 1:
        raise exception.DataError(f"'days_op' cant be: '{days_op}'.")

    if progress:
        utils.load_bar(size=1, step=0)
        t = time()

    if not 'Volume' in data.columns:
        data['Volume'] = 0

    if progress: 
        utils.load_bar(size=1, step=1)
        print('| DataTimer:',utils.num_align(round(time()-t,2)))

    _cm.__data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    _cm.__data.index.name = 'Date'
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index)

    _cm.__data_icon = icon.strip()
    _cm.__data_interval = interval.strip()
    _cm.__data_width_day = days_op
    _cm.__data_width_day = utils.calc_day(interval, _cm.__data_width)

    if statistics: stats_icon(prnt=True)

def run(cls:type, initial_funds:int = 10000, commission:tuple = 0, 
        spread:tuple = 0, slippage:tuple = 0,
        prnt:bool = True, progress:bool = True) -> str:
    """
    Run Your Strategy.

    Executes your trading strategy.

    Info:
        CostsValue format:
            (maker, taker) may have an additional tuple indicating 
            that it may be a random number between two numbers.

        For commissions, spreads and slippage, the `CostsValue` format will be followed.

    Args:
        cls (type): A class inherited from `StrategyClass` where the strategy is 
                    implemented.
        initial_funds (int, optional): Initial amount of funds to start with. Used for 
                            statistics. Default is 10,000.
        commission (float, optional): The commission will be charged for each purchase/sale execution.
        spread (float, optional): The spread is the separation between the bid and ask 
            price and is used to mark the order book limits.
            There is no variation between maker and taker.
        slippage (float, optional): It will be calculated at each entry and exit.
            There is no variation between maker and taker.
        prnt (bool, optional): If True, prints trade statistics. If False, returns a string 
                    with the statistics. Default is True.
        progress (bool, optional): If True, shows a progress bar and timer. Default is True.

    Note:
        If your function prints to the console, the loading bar may not 
        function as expected.

    Returns:
        str: statistics.
    """
    # Exceptions.
    if _cm.__data is None: 
        raise exception.RunError('Data not loaded.')
    elif initial_funds < 0: 
        raise exception.RunError("'initial_funds' cannot be less than 0.")
    elif not issubclass(cls, strategy.StrategyClass):
        raise exception.RunError(
            f"'{cls.__name__}' is not a subclass of 'strategy.StrategyClass'.")
    elif cls.__abstractmethods__:
        raise exception.RunError(
            "The implementation of the 'next' abstract method is missing.")

    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)
    _cm._init_funds = initial_funds

    # Costs
    commission_cv = flx.CostsValue(commission, supp_double=True, 
                                   cust_error="Error of 'commission'.")
    slippage_cv = flx.CostsValue(slippage, cust_error="Error of 'slippage'.")
    spread_cv = flx.CostsValue(spread, cust_error="Error of 'spread'.")

    instance = cls(spread_pct=spread_cv, commission=commission_cv, 
                   slippage_pct=slippage_cv, init_funds=initial_funds)
    t = time()
    
    step_t = time()
    step_history = np.zeros(10)
    steph_index = 0

    skip = max(1, _cm.__data.shape[0] // _cm.max_bar_updates)

    for f in range(1, _cm.__data.shape[0]+2):
        if (progress and (f % skip == 0 or f >= _cm.__data.shape[0]) 
            and _cm.__data.shape[0] >= f):

            step_time = time()-step_t
            step_history[steph_index % 10] = step_time
            steph_index += 1
 
            run_timer_text = (
                f"| RunTimer: {utils.num_align(time()-t)} \n"
                f"| TimerPredict: " + utils.num_align(
                    time()-t + (((md:=np.median(step_history))
                    + (time()-t - md*f)/f) *
                    (_cm.__data.shape[0]-f))) + " \n"
            ) if _cm.run_timer else ""

            text = utils.text_fix(f"""
                | StepTime: {utils.num_align(step_time)} 
                {run_timer_text}
                """)

            utils.load_bar(size=_cm.__data.shape[0], step=f, text=text) 
        step_t = time()

        instance._StrategyClass__before(data=_cm.__data.iloc[:f])
    if progress or _cm.run_timer:
        print(
            f'RunTimer: {utils.num_align(time()-t)}'
            if _cm.run_timer and not progress else '') 

    act_trades = instance._StrategyClass__trades_ac
    _cm.__trades = instance._StrategyClass__trades_cl
    
    if not act_trades.empty: _cm.__trades = pd.concat([
        _cm.__trades, act_trades.dropna(axis=1, how='all')
        ], ignore_index=True)

    try: 
        return stats_trades(prnt=prnt)
    except: pass
    
def plot(log:bool = False, progress:bool = True, 
         position:str = 'complex', block:bool = True) -> None:
    """
    Plot Graph with Trades.

    Plots your data, highlighting the trades made.

    Color Guide:
        - Gold: 'x' = Position close.
        - Green, '^' = Buy position.
        - Red, 'v' = Sell position.

    Args:
        log (bool, optional): If True, plots data using a logarithmic scale. 
            Default is False.
        progress (bool, optional): If True, shows a progress bar and timer. 
            Default is True.
        position (str, optional): Specifies how positions are drawn. Options 
            are 'complex' or 'simple'. If None or 'none', positions will not 
            be drawn. Default is 'complex'. The "complex" option may take longer 
            to process.
        block (bool, optional): If True, pauses script execution until all 
            figure windows are closed. If False, the script continues running 
            after displaying the figures. Default is True.
    """

    # Exceptions.
    if _cm.__data is None or not type(_cm.__data) is pd.DataFrame or _cm.__data.empty: 
        raise exception.PlotError('Data not loaded.')
    elif position and not position.lower() in ('complex', 'simple', 'none'):
        raise exception.PlotError(
            f"'{position}' Not a valid option for: 'position'.")
    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)
    
    if progress: 
        t = time()
        text = f'| PlotTimer: {utils.num_align(0)} '
        utils.load_bar(size=4, step=0, text=text)

    mpl.pyplot.style.use('ggplot')
    fig = mpl.pyplot.figure(figsize=(16,8))
    ax1 = mpl.pyplot.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = mpl.pyplot.subplot2grid((6,1), (5,0), rowspan=1, 
                                  colspan=1, sharex=ax1)
    ax2.set_yticks([])
    
    if log: 
        ax1.semilogy(_cm.__data['Close'], alpha=0); ax2.semilogy(alpha=0)

    fig.tight_layout(); fig.subplots_adjust(hspace=0)

    if progress: 
        text = f'| PlotTimer: {utils.num_align(time()-t)} '
        utils.load_bar(size=4, step=1, text=text)

    utils.plot_candles(ax1, _cm.__data, _cm.__data_width*0.9)

    if progress: 
        text = f'| PlotTimer: {utils.num_align(time()-t)} '
        utils.load_bar(size=4, step=2, text=text)

    if _cm.__data['Volume'].max() > 0:
      ax2.fill_between(_cm.__data.index, _cm.__data['Volume'], step='mid')
      ax2.set_ylim(None, _cm.__data['Volume'].max()*1.5)

    if position and position.lower() != 'none' and not _cm.__trades.empty:
        utils.plot_position(_cm.__trades, ax1, 
                          all=True if position.lower() == 'complex' else False,
                          alpha=0.3, alpha_arrow=0.8, 
                          width_exit=lambda x: _cm.__data.index[-1]-x['Date'])

    if progress: 
        text = f'| PlotTimer: {utils.num_align(time()-t)} '
        utils.load_bar(size=4, step=3, text=text)

    date_format = mpl.dates.DateFormatter('%H:%M %d-%m-%Y')
    ax1.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    ix_date = mpl.dates.num2date(_cm.__data.index)

    s_date = ".".join(str(val) for val in 
                    [ix_date[0].day, ix_date[0].month, 
                    ix_date[0].year])
    
    e_date = ".".join(str(val) for val in 
                    [ix_date[-1].day, ix_date[-1].month, 
                    ix_date[-1].year])
    
    mpl.pyplot.gcf().canvas.manager.set_window_title(
        f"Back testing: '{_cm.__data_icon}' {s_date}~{e_date}")

    if progress: 
        text = f'| PlotTimer: {utils.num_align(time()-t)} \n'
        utils.load_bar(size=4, step=4, text=text)

    mpl.pyplot.show(block=block)

def plot_strategy(log:bool = False, view:str = 'p/w/r/e', 
                  custom_graph:dict = {}, block:bool = True) -> None:
    """
    Plot Strategy Statistics.

    Plots statistics for your strategy.

    Available Graphics:
        - 'e' = Equity graph.
        - 'p' = Profit graph.
        - 'r' = Return graph.
        - 'w' = Winnings graph.

    Args:
        log (bool, optional): If True, plots data using a logarithmic scale. 
            Default is False.
        view (str, optional): Specifies which graphics to display. 
            Each option occupies the entire window. Default is 'p/w/r/e'.
        custom_graph (dict, optional): Custom graph, a dictionary with 
            'name':'function' where the function will 
            be passed: 'ax', '_cm.__trades', '_cm.__data', 'log'.
            To avoid visual problems, I suggest using 
            'trades.index' as the x-axis or normalizing the axis.
        block (bool, optional): If True, pauses script execution until all figure 
            windows are closed. If False, the script continues running after 
            displaying the figures. Default is True.
    """
    for i in custom_graph: plot_strategy_add(custom_graph[i], i)

    view = view.lower().strip().split('/')
    view = [i for i in view if i in ('p','w','r','e') | _cm.__custom_plot.keys()]

    # Exceptions.
    if _cm.__trades.empty: 
        return 'Trades not loaded.'
    elif not 'Profit' in _cm.__trades.columns:  
        return 'There is no data to see.'
    elif len(view) > 4 or len(view) < 1: 
        raise exception.StatsError(utils.text_fix(f"""
            'view' allowed format: 's/s/s/s' where s is the name of the graph.
            Available graphics: 'p','w','r','e'{
                (","+",".join(
                    [f"'{i}'" for i in _cm.__custom_plot.keys()])) 
                    if _cm.__custom_plot.keys() else ''}.
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
                                             sharex=ax)
            case 3: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], rowspan=3, 
                                             colspan=2 if i==0 else 1, 
                                             sharex=ax)
            case 4: 
                ax = mpl.pyplot.subplot2grid((6,2), loc[i], 
                                             rowspan=3, colspan=1, 
                                             sharex=ax)

        match v:
            case 'p':
                ax.plot(_cm.__trades.index,_cm.__trades['Profit'].cumsum(), 
                        c='black', label='Profit.')
                
                if log: ax.set_yscale('symlog')
            case 'w':
                ax.plot(_cm.__trades.index,
                        (_cm.__trades['ProfitPer'].apply(
                            lambda row: 1 if row>0 else -1)).cumsum(), 
                        c='black', label='Winnings.')
            case 'e':
                ax.plot(_cm.__trades.index, 
                        np.cumprod(1 + _cm.__trades['ProfitPer'] / 100), 
                        c='black', label='Equity.')

                if log: ax.set_yscale('symlog')
            case 'r':
                ax.plot(_cm.__trades.index,_cm.__trades['ProfitPer'].cumsum(), 
                        c='black', label='Return.')

                if log: ax.set_yscale('symlog')
            case key if key in _cm.__custom_plot.keys():
                _cm.__custom_plot[v](ax, _cm.__trades, 
                                    _cm.__data, log)
            case _: pass

        ax.legend(loc='upper left')

    mpl.pyplot.xticks([])
    mpl.pyplot.gcf().canvas.manager.set_window_title(f'Strategy statistics')
    mpl.pyplot.show(block=block)

def plot_strategy_decorator(name:str) -> callable:
    """
    Add statistics for plot decorator.

    Use a decorator to add the function to 
    'custom_plot' so you can run it in 'plot_strategy'.

    To avoid visual problems, I suggest using 
        'trades.index' as the x-axis or normalizing the axis.

    Args:
        name (str, optional): Name with which it will be called.

    Returns:
        callable: 'plot_strategy_add'.
    """

    return lambda x: plot_strategy_add(x, name)

def plot_strategy_add(func, name:str) -> callable:
    """
    Add statistics for plot.

    Add functions and then see them graphed with 'plot_strategy'.

    Args:
        func: Function, to which this will be passed in order: 
            'ax', '_cm.__trades', '_cm.__data', 'log'.
            To avoid visual problems, I suggest using 
            'trades.index' as the x-axis or normalizing the axis.
        name (str, optional): Name with which it will be called.

    Returns:
        callable: 'func' param.
    """

    if not name or name in _cm.__custom_plot.keys() or not callable(func):
        raise exception.StatsError("Error assigning value to '__custom_plot'.")
    _cm.__custom_plot[name.strip()] = func
    return func

def stats_icon(prnt:bool = True, data:pd.DataFrame = None, 
               data_icon:str = None, data_interval:str = None) -> str:
    """
    Icon Statistics.

    Displays statistics of the uploaded data.

    Args:
        prnt (bool, optional): If True, prints the statistics. If False, returns
            the statistics as a string. Default is True.
        data (pd.DataFrame, optional): The data with which the statistics 
            are calculated, if left to None the loaded data will be used.
            The DataFrame must contain the following columns: 
            ('Close', 'Open', 'High', 'Low', 'Volume').
        data_icon (str, optional): Icon shown in the statistics, 
            if you leave it at None the loaded data will be the one used.
        data_interval (str, optional): Interval shown in the statistics, 
            if you leave it at None the loaded data will be the one used.

    Returns:
        str: statistics.
    """

    data_interval = __data_interval if data_interval is None else data_interval
    data_icon = __data_icon if data_icon is None else data_icon
    data = __data if data is None else data
    
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
        
        e_date = ".".join(str(val) for val in 
                        [data.index[-1].day, data.index[-1].month, 
                        data.index[-1].year]
                        ) if isinstance(data.index[0], pd.Timestamp) else ""
        
        r_date = f"{s_date}~{e_date}"
    else: r_date = ""

    text = utils.statistics_format({
        'Last price':[utils.round_r(data['Close'].iloc[-1],2), 
                      _cm.__COLORS['BOLD']],
        'Maximum price':[utils.round_r(data['High'].max(),2),
                         _cm.__COLORS['GREEN']],
        'Minimum price':[utils.round_r(data['Low'].min(),2),
                         _cm.__COLORS['RED']],
        'Maximum volume':[utils.round_r(data['Volume'].max(), 2),
                          _cm.__COLORS['CYAN']],
        'Sample size':[len(data.index)],
        'Standard deviation':[utils.round_r(
            np.std(data['Close'].dropna(), ddof=1),2)],
        'Average price':[utils.round_r(data['Close'].mean(),2),
                         _cm.__COLORS['YELLOW']],
        'Average volume':[utils.round_r(data['Volume'].mean(),2),
                          _cm.__COLORS['YELLOW']],
        f"'{data_icon}'":[f'{r_date} ~ {data_interval}',
                          _cm.__COLORS['CYAN']],
    }, f"---Statistics of '{data_icon}'---")

    text = text if _cm.dots else text.replace('.', ',')
    if prnt:print(text) 
    else: return text

def stats_trades(data:bool = False, prnt:bool = True) -> str:
    """
    Trades Statistics.

    Statistics of the results.

    Args:
        data (bool, optional): If True, `stats_icon` is also returned.
        prnt (bool, optional): If True, prints the statistics. If False, returns 
            the statistics as a string. Default is True.

    Info:
        - Trades: The number of operations performed.
        - Op years: Years operated from the first to the last.
        - Return: The total equity earned.
        - Profit: The total amount earned.
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
                calculated as (Win rate × Average win) - (Loss rate × Average loss).
        - Math hope r: The relative mathematical expectation, 
                calculated as (Win rate × Average ratio) - (Loss rate × 1).
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
                (Win rate × Average win) - (Loss rate × Average loss).
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
                values ​​mean high frequency and low values ​​mean the opposite.
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
        str: statistics.
    """

    # Exceptions.
    if _cm.__trades.empty: 
        raise exception.StatsError('Trades not loaded.')
    elif not 'ProfitPer' in _cm.__trades.columns:  
        raise exception.StatsError('There is no data to see.')
    elif np.isnan(_cm.__trades['ProfitPer'].mean()): 
        raise exception.StatsError('There is no data to see.') 

    # Number of years operated.
    op_years = abs(
        (_cm.__trades['Date'].iloc[-1] - _cm.__trades['Date'].iloc[0])/
        (_cm.__data_width_day*_cm.__data_year_days))

    # Annualized trades calc.
    trades_calc = _cm.__trades.copy()
    trades_calc['Year'] = ((trades_calc['Date'] - trades_calc['Date'].iloc[0]) / 
                  (trades_calc['Date'].iloc[-1] - trades_calc['Date'].iloc[0]) * 
                  op_years).astype(int)

    trades_calc['Diary'] = ((trades_calc['Date'] - trades_calc['Date'].iloc[0]) / 
                (trades_calc['Date'].iloc[-1] - trades_calc['Date'].iloc[0]) * 
                op_years*_cm.__data_year_days).astype(int)

    trades_calc['Duration'] = (trades_calc['PositionDate']-trades_calc['Date'])/_cm.__data_width_day
    trades_calc['Multiplier'] = 1 + trades_calc['ProfitPer'] / 100

    ann_return = trades_calc.groupby('Year')['Multiplier'].prod()
    ann_profit = trades_calc.groupby('Year')['Profit'].sum()

    diary_return = trades_calc.groupby('Diary')['Multiplier'].prod()
    diary_profit = trades_calc.groupby('Diary')['Profit'].sum()

    # Consecutive trades calc.
    trades_count_cs = _cm.__trades['Profit'].apply(
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

    text = utils.statistics_format({
        'Trades':[len(_cm.__trades.index),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['CYAN']],

        'Op years':[utils.round_r(op_years, 2), _cm.__COLORS['CYAN']],

        'Return':[str(_return:=utils.round_r((trades_calc['Multiplier'].prod()-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return) > 0 else _cm.__COLORS['RED'],],

        'Profit':[str(_profit:=utils.round_r(_cm.__trades['Profit'].sum(),2)),
                _cm.__COLORS['GREEN'] if float(_profit) > 0 else _cm.__COLORS['RED'],],

        'Max return':[str(utils.round_r((
            np.cumprod(trades_calc['Multiplier'].dropna()).max()-1)*100,2))+'%'],

        'Return from max':[str(utils.round_r(
            -((np.cumprod(trades_calc['Multiplier'].dropna()).max()-1)
            - (trades_calc['Multiplier'].prod()-1))*100,2))+'%'],

        'Days from max':[str(utils.round_r(
            (trades_calc['Date'].dropna().iloc[-1]
                - trades_calc['Date'].dropna().loc[
                np.argmax(np.cumprod(trades_calc['Multiplier'].dropna()))])
            / _cm.__data_width_day, 2)),
            _cm.__COLORS['CYAN']],

        'Return ann':[str(_return_ann:=utils.round_r((ann_return.prod()**(1/op_years)-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return_ann) > 0 else _cm.__COLORS['RED'],],

        'Profit ann':[str(_profit_ann:=utils.round_r(ann_profit.mean(),2)),
                  _cm.__COLORS['GREEN'] if float(_profit_ann) > 0 else _cm.__COLORS['RED'],],

        'Return ann vol':[utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1)
                                        *np.sqrt(_cm.__data_year_days), 2),
                          _cm.__COLORS['YELLOW']],

        'Profit ann vol':[utils.round_r(np.std(diary_profit.dropna(),ddof=1)
                                    *np.sqrt(_cm.__data_year_days), 2),
                        _cm.__COLORS['YELLOW']],

        'Average ratio':[utils.round_r(stats.average_ratio(_cm.__trades), 2),
                        _cm.__COLORS['YELLOW'],],

        'Average return':[str(utils.round_r((
                trades_calc['Multiplier'].dropna().mean()-1)*100,2))+'%',
            _cm.__COLORS['YELLOW'],],

        'Average profit':[str(utils.round_r(_cm.__trades['Profit'].mean(),2))+'%',
                    _cm.__COLORS['YELLOW'],],

        'Profit fact':[_profit_fact:=utils.round_r(stats.profit_fact(_cm.__trades['Profit']), 2),
                _cm.__COLORS['GREEN'] if float(_profit_fact) > 1 else _cm.__COLORS['RED'],],

        'Return diary std':[(_return_std:=utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1), 2)),
                    _cm.__COLORS['YELLOW'] if float(_return_std) > 1 else _cm.__COLORS['GREEN'],],

        'Profit diary std':[(_profit_std:=utils.round_r(np.std(diary_profit.dropna(),ddof=1), 2)),
                      _cm.__COLORS['YELLOW'] if float(_profit_std) > 1 else _cm.__COLORS['GREEN'],],

        'Math hope':[_math_hope:=utils.round_r(stats.math_hope(_cm.__trades['Profit']), 2),
            _cm.__COLORS['GREEN'] if float(_math_hope) > 0 else _cm.__COLORS['RED'],],

        'Math hope r':[_math_hope_r:=utils.round_r(
                stats.math_hope_relative(_cm.__trades, _cm.__trades['ProfitPer']), 2),
            _cm.__COLORS['GREEN'] if float(_math_hope_r) > 0 else _cm.__COLORS['RED'],],

        'Historical var':[0 if _cm.__trades['Profit'].dropna().empty else utils.round_r(
                            stats.var_historical(_cm.__trades['Profit'].dropna()), 2)],

        'Parametric var':[0 if _cm.__trades['Profit'].dropna().empty else utils.round_r(
                            stats.var_parametric(_cm.__trades['Profit'].dropna()), 2)],

        'Sharpe ratio':[utils.round_r(stats.sharpe_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            _cm.__data_year_days,
            (diary_return.dropna()-1)*100), 2)],

        'Sharpe ratio$':[utils.round_r(stats.sharpe_ratio(
            np.average(ann_profit),
            _cm.__data_year_days,
            diary_profit), 2)],

        'Sortino ratio':[utils.round_r(stats.sortino_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            _cm.__data_year_days,
            (diary_return.dropna()-1)*100), 2)],

        'Sortino ratio$':[utils.round_r(stats.sortino_ratio(
            np.average(ann_profit),
            _cm.__data_year_days,
            diary_profit), 2)],

        'Duration ratio':[utils.round_r(
            trades_calc['Duration'].sum()/len(_cm.__trades.index), 2),
            _cm.__COLORS['CYAN']],

        'Payoff ratio':[utils.round_r(stats.payoff_ratio(_cm.__trades['ProfitPer']))],

        'Expectation':[utils.round_r(stats.expectation(_cm.__trades['ProfitPer']))],

        'Skewness':[utils.round_r((diary_return.dropna()-1).skew(), 2)],

        'Kurtosis':[utils.round_r((diary_return.dropna()-1).kurt(), 2)],

        'Average winning op':[str(utils.round_r(_cm.__trades['ProfitPer'][
                _cm.__trades['ProfitPer'] > 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['GREEN']],

        'Average losing op':[str(utils.round_r(_cm.__trades['ProfitPer'][
                _cm.__trades['ProfitPer'] < 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['RED']],

        'Average duration winn':[str(utils.round_r(trades_calc['Duration'][
                trades_calc['ProfitPer'] > 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Average duration loss':[str(utils.round_r(trades_calc['Duration'][
                trades_calc['ProfitPer'] < 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Daily frequency op':[utils.round_r(
            len(_cm.__trades.index) / (op_years*_cm.__data_year_days), 2),
            _cm.__COLORS['CYAN']],

        'Max consecutive winn':[trades_csct.max(),
                                _cm.__COLORS['GREEN']],

        'Max consecutive loss':[abs(trades_csct.min()),
                                _cm.__COLORS['RED']],

        'Max losing streak':[abs(trades_streak.min())],

        'Max drawdown':[str(round(
            stats.max_drawdown(np.cumprod(
                trades_calc['Multiplier'].dropna()))*100,1)) + '%'],

        'Average drawdown':[str(-round(np.mean(
            stats.get_drawdowns(np.cumprod(
                trades_calc['Multiplier'].dropna())))*100, 1)) + '%'],

        'Max drawdown$':[str(round(
            stats.max_drawdown(_cm.__trades['Profit'].dropna().cumsum()+
                               _cm._init_funds)*100,1)) + '%'],

        'Average drawdown$':[str(-round(np.mean(
            stats.get_drawdowns(_cm.__trades['Profit'].dropna().cumsum()+
                                _cm._init_funds))*100, 1)) + '%'],

        'Long exposure':[str(round(
            stats.long_exposure(_cm.__trades['Type'])*100)) + '%',
            _cm.__COLORS['GREEN']],

        'Winnings':[str(round(stats.winnings(_cm.__trades['ProfitPer'])*100)) + '%'],

    }, "---Statistics of strategy---")

    text = text if _cm.dots else text.replace('.', ',')
    if data: text += stats_icon(False)
    
    if prnt: print(text)
    else: return text
