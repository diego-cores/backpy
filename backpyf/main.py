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
    stats_icon: Shows statistics related to the financial icon.
    stats_trades: Statistics of the trades.
"""

from datetime import datetime

import matplotlib.pyplot
import matplotlib as mpl

import pandas as pd
import numpy as np

from time import time

from . import _commons as _cm
from . import exception
from . import strategy
from . import utils

def load_binance_data(symbol:str = 'BTCUSDT', interval:str = '1d', 
                      start_time:str = None, end_time:str = None,
                      statistics:bool = True, progress:bool = True,
                      data_extract:bool = False) -> tuple:
    """
    Load Binance Data.

    Loads data using the binance-connector module.

    Args:
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str): Start date for load data in YYYY-MM-DD format.
        end_time (str): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
                        the module variables will not be assigned with them.
    """
    # Exceptions.
    if start_time is None or end_time is None:
        raise exception.BinanceError('Binance parameters error.')
    
    try:
        from binance.spot import Spot as Client

        if progress:
            t = time()
            step = 0
        
        def __loop_def (st_t):
            dt = client.klines(symbol=symbol, 
                               interval=interval, 
                               startTime=st_t, 
                               endTime=int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000), 
                               limit=1000)
            
            if progress:
                nonlocal step

                utils.load_bar(size=step+1, step=step, count=False)
                print(
                    '0 of 1 completed | DataTimer:',utils.num_align(time()-t), end='')
                step += 1

            return dt
            
        client = Client()
        data = utils._loop_data(
            function=__loop_def,
            bpoint=lambda x, y=None: y == int(x[0].iloc[-1]) if y else int(x[0].iloc[-1]),
            init = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000),
            timeout = _cm.__binance_timeout
            ).astype(float)
        
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
        
        if progress: 
            utils.load_bar(size=1, step=1)
            print('| DataTimer:',utils.num_align(round(time()-t,2)))
        
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

    except ModuleNotFoundError: 
        raise exception.BinanceError('Binance connector is not installed.')
    except: 
        raise exception.BinanceError('Binance parameters error.')

def load_yfinance_data(tickers:str = any, 
                       start:str = None, end:str = None, interval:str = '1d', 
                       statistics:bool = True, progress:bool = True,
                       data_extract:bool = False) -> tuple:
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
        statistics (bool, optional): If True, prints statistics of the downloaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
                        the module variables will not be assigned with them.
    """
    try:
        import yfinance as yf

        t = time() if progress else None

        yf.set_tz_cache_location('.\yfinance_cache')
        
        data = yf.download(tickers, start=start, end=end, 
                             interval=interval, progress=progress)
        
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
        
        _cm.__data = data
        _cm.__data_width = data_width
        _cm.__data_icon = tickers.strip()
        _cm.__data_interval = interval.strip()

    except ModuleNotFoundError: 
        raise exception.YfinanceError('Yfinance is not installed.')
    except: 
        raise exception.YfinanceError('Yfinance parameters error.')

def load_data(data:pd.DataFrame = any, icon:str = None, 
              interval:str = None, statistics:bool = True, 
              progress:bool = True) -> None: 
    """
    Load Any Data.

    Loads data into the system.

    Args:
        data (pd.DataFrame): DataFrame containing the data to load. Must have the 
                            following columns: ['Open', 'High', 'Low', 'Close', 
                            'Volume'].
        icon (str, optional): String representing the data icon.
        interval (str, optional): String representing the data interval.
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

    if statistics: stats_icon(prnt=True)

def run(cls:type, initial_funds:int = 10000, 
        commission:float = 0, spread:float = 0, 
        prnt:bool = True, progress:bool = True) -> str:
    """
    Run Your Strategy.

    Executes your trading strategy.

    Args:
        cls (type): A class inherited from `StrategyClass` where the strategy is 
                    implemented.
        initial_funds (int, optional): Initial amount of funds to start with. Used for 
                            statistics. Default is 10,000.
        commission (float, optional): Commission percentage for each trade. Used for 
                            statistics. Default is 0.
        spread (float, optional): Spread percentage for each trade. It is calculated 
                            at the closing and opening of each trade.
        prnt (bool, optional): If True, prints trade statistics. If False, returns a string 
                    with the statistics. Default is True.
        progress (bool, optional): If True, shows a progress bar and timer. Default is True.

    Note:
        If your function prints to the console, the loading bar may not 
        function as expected.
    """
    # Exceptions.
    if _cm.__data is None: 
        raise exception.RunError('Data not loaded.')
    elif initial_funds < 0: 
        raise exception.RunError("'initial_funds' cannot be less than 0.")
    elif commission < 0: 
        raise exception.RunError("'commission' cannot be less than 0.")
    elif not issubclass(cls, strategy.StrategyClass):
        raise exception.RunError(
            f"'{cls.__name__}' is not a subclass of 'strategy.StrategyClass'")
    elif cls.__abstractmethods__:
        raise exception.RunError(
            "The implementation of the 'next' abstract method is missing.")
    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)

    _cm._init_funds = initial_funds
    instance = cls(spread_pct=spread , commission=commission, 
                   init_funds=initial_funds)
    t = time()
    
    step_t = time()
    step_history = np.array([])

    for f in range(1, _cm.__data.shape[0]+2):
        if progress and _cm.__data.shape[0] >= f:
            utils.load_bar(size=_cm.__data.shape[0], step=f) 
            step_history = np.append(step_history, time()-step_t)

            run_timer_text = (
                f"| RunTimer: {utils.num_align(time()-t)} \n"
                f"| TimerPredict: " + utils.num_align(
                    step_history.sum() + 
                    (step_history.mean()+step_history.std()) * 
                    (_cm.__data.shape[0]-step_history.size)) + " \n"
            ) if _cm.run_timer else ""

            print(utils.text_fix(f"""
                | StepTime: {utils.num_align(step_history[-1])} 
                {run_timer_text}
                """), 
                end='')
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

    #try: 
    return stats_trades(prnt=prnt)
    #except: pass
    
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
        utils.load_bar(size=4, step=0)
        print('| PlotTimer:',utils.num_align(time()-t), end='')

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
        utils.load_bar(size=4, step=1)
        print('| PlotTimer:',utils.num_align(time()-t), end='')

    utils.plot_candles(ax1, _cm.__data, _cm.__data_width*0.9)

    if progress: 
        utils.load_bar(size=4, step=2)
        print('| PlotTimer:',utils.num_align(time()-t), end='')

    if _cm.__data['Volume'].max() > 0:
      ax2.fill_between(_cm.__data.index, _cm.__data['Volume'], step='mid')
      ax2.set_ylim(None, _cm.__data['Volume'].max()*1.5)

    if position and position.lower() != 'none' and not _cm.__trades.empty:
        utils.plot_position(_cm.__trades, ax1, 
                          all=True if position.lower() == 'complex' else False,
                          alpha=0.3, alpha_arrow=0.8, 
                          width_exit=lambda x: _cm.__data.index[-1]-x['Date'])

    if progress: 
        utils.load_bar(size=4, step=3)
        print('| PlotTimer:',utils.num_align(time()-t), end='')

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
        utils.load_bar(size=4, step=4)
        print('| PlotTimer:',utils.num_align(time()-t))

    mpl.pyplot.show(block=block)

def plot_strategy(log:bool = False, view:str = 'p/w/r/n', 
                  block:bool = True) -> None:
    """
    Plot Strategy Statistics.

    Plots statistics for your strategy.

    Available Graphics:
        - 'p' = Profit graph.
        - 'r' = Return graph.
        - 'w' = Winnings graph.

    Args:
        log (bool, optional): If True, plots data using a logarithmic scale. 
            Default is False.
        view (str, optional): Specifies which graphics to display. Options are 
            'p', 'r', 'w', or 'n'. Each option occupies the entire window. 
            Default is 'p/w/r/n'.
        block (bool, optional): If True, pauses script execution until all figure 
            windows are closed. If False, the script continues running after 
            displaying the figures. Default is True.
    """

    view = view.lower().strip().split('/')
    view = [i for i in view if i in ('p','w','r')]

    # Exceptions.
    if _cm.__trades.empty: 
        return 'Trades not loaded.'
    elif not 'Profit' in _cm.__trades.columns:  
        return 'There is no data to see.'
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
                ax.plot(_cm.__trades.index,_cm.__trades['Profit'].cumsum(), 
                        c='black', label='Profit.')
                
                if log: ax.set_yscale('symlog')
            case 'w':
                ax.plot(_cm.__trades.index,
                        (_cm.__trades['ProfitPer'].apply(
                            lambda row: 1 if row>0 else -1)).cumsum(), 
                        c='black', label='Winnings.')
            case 'r':
                ax.plot(_cm.__trades.index,_cm.__trades['ProfitPer'].cumsum(), 
                        c='black', label='Return.')

                if log: ax.set_yscale('symlog')
            case _: pass
        ax.legend(loc='upper left')

    mpl.pyplot.xticks([])
    mpl.pyplot.gcf().canvas.manager.set_window_title(f'Strategy statistics.')
    mpl.pyplot.show(block=block)

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
    }, f"Statistics of '{data_icon}'")

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
        - Return: The total percentage earned.
        - Average return: The average percentage earned.
        - Average ratio: The average ratio.
        - Profit: The total amount earned.
        - Profit fact: The profit factor is calculated by dividing total profits by total 
                losses.
        - Profit std: The standard deviation of profits, indicating the variability in performance.
        - Return std: The standard deviation of return, indicating the variability in performance.
        - Math hope: The mathematical expectation (or expected value) of returns, 
                calculated as (Win rate × Average win) - (Loss rate × Average loss).
        - Historical var: The Value at Risk (VaR) estimated using historical data, 
                calculated as the profit at the (100 - confidence level) percentile.
        - Parametric var: The Value at Risk (VaR) calculated assuming a normal distribution, 
                defined as the mean profit minus z-alpha times the standard deviation.
        - Sharpe ratio: The risk-adjusted return, calculated as the average 
                profit divided by the standard deviation of profits.
        - Sharpe ratio%: The risk-adjusted return, calculated as the average 
                profit divided by the standard deviation of return.
        - Max drawdown: The biggest drawdown the 'profit' has ever had.
        - Average drawdown: The average of all drawdowns, 
                indicating the typical loss experienced before recovery.
        - Long exposure: What percentage of traders are long.
        - Winnings: Percentage of operations won.
    """

    # Exceptions.
    if _cm.__trades.empty: 
        raise exception.StatsError('Trades not loaded.')
    elif not 'ProfitPer' in _cm.__trades.columns:  
        raise exception.StatsError('There is no data to see.')
    elif np.isnan(_cm.__trades['ProfitPer'].mean()): 
        raise exception.StatsError('There is no data to see.') 

    text = utils.statistics_format({
        'Trades':[len(_cm.__trades.index),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['CYAN']],

        'Return':[str(_return:=utils.round_r(_cm.__trades['ProfitPer'].sum(),2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return) > 0 else _cm.__COLORS['RED'],],

        'Average return':[utils.round_r(_cm.__trades['ProfitPer'].mean(),2)+'%',
                          _cm.__COLORS['YELLOW'],],

        'Average ratio':[utils.round_r(
            (abs(_cm.__trades['Close']-_cm.__trades['TakeProfit']) / 
                abs(_cm.__trades['Close']-_cm.__trades['StopLoss'])).mean() 
            if not _cm.__trades['TakeProfit'].apply(
                    lambda x: x is None or x <= 0).all() and 
                not _cm.__trades['StopLoss'].apply(
                    lambda x: x is None or x <= 0).all() else 0, 2),
                _cm.__COLORS['YELLOW'],],

        'Profit':[(_profit:=utils.round_r(_cm.__trades['Profit'].sum(),2)),
                  _cm.__COLORS['GREEN'] if float(_profit) > 0 else _cm.__COLORS['RED'],],

        'Profit fact':[_profit_fact:=(utils.round_r(
            _cm.__trades[_cm.__trades['Profit']>0]['Profit'].sum()/
            abs(_cm.__trades[_cm.__trades['Profit']<=0]['Profit'].sum()),2) 
            if not pd.isna(_cm.__trades['Profit']).all() and
                (_cm.__trades['Profit']>0).sum() > 0 and
                (_cm.__trades['Profit']<=0).sum() > 0 else 0),
                _cm.__COLORS['GREEN'] if float(_profit_fact) > 1 else _cm.__COLORS['RED'],],

        'Profit std':[(_profit_std:=utils.round_r(np.std(_cm.__trades['Profit'],ddof=1), 2)),
                      _cm.__COLORS['YELLOW'] if float(_profit_std) > 1 else _cm.__COLORS['GREEN'],],

        'Return std':[(_return_std:=utils.round_r(np.std(_cm.__trades['ProfitPer'],ddof=1), 2)),
                    _cm.__COLORS['YELLOW'] if float(_return_std) > 1 else _cm.__COLORS['GREEN'],],

        'Math hope':[_math_hope:=utils.round_r((
                (_cm.__trades['Profit'] > 0).sum()/len(_cm.__trades.index)*
                    _cm.__trades['Profit'][_cm.__trades['Profit'] > 0].mean())-
                ((_cm.__trades['Profit'] < 0).sum()/len(_cm.__trades.index)*
                    -_cm.__trades['Profit'][_cm.__trades['Profit'] < 0].mean()), 2),
            _cm.__COLORS['GREEN'] if float(_math_hope) > 0 else _cm.__COLORS['RED'],],

        'Historical var':[utils.round_r(
                            utils.var_historical(_cm.__trades['Profit']), 2)],

        'Parametric var':[utils.round_r(
                            utils.var_parametric(_cm.__trades['Profit']), 2)],

        'Sharpe ratio':[utils.round_r(np.average(
                _cm.__trades['Profit'].dropna())
                    /np.std(_cm.__trades['Profit'].dropna(),ddof=1), 2)],

        'Sharpe ratio%':[utils.round_r(np.average(
                _cm.__trades['ProfitPer'].dropna())
                    /np.std(_cm.__trades['ProfitPer'].dropna(),ddof=1), 2)],

        'Max drawdown':[str(round(
            utils.max_drawdown(_cm.__trades['Profit'].dropna().cumsum()+
                               _cm._init_funds)*100,1)) + '%'],

        'Average drawdown':[str(-round(np.mean(
            utils.get_drawdowns(_cm.__trades['Profit'].dropna().cumsum()+
                                _cm._init_funds))*100, 1)) + '%'],

        'Long exposure':[str(round((
                _cm.__trades['Type']==1).sum()/
                    _cm.__trades['Type'].count()*100,1)) + '%',
            _cm.__COLORS['CYAN'],],

        'Winnings':[str(round(
            (_cm.__trades['ProfitPer']>0).sum()/
                _cm.__trades['ProfitPer'].count()*100,1) 
            if not ((_cm.__trades['ProfitPer']>0).sum() == 0 or 
                _cm.__trades['ProfitPer'].count() == 0) else 0) + '%'],

    }, "Statistics of strategy.")

    text = text if _cm.dots else text.replace('.', ',')
    if data: text += stats_icon(False)
    
    if prnt: print(text)
    else: return text
