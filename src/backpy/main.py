"""
Main module

This module contains the main functions of BackPy, including data loading, 
strategy processing, and graph display.

Variables:
    logger (Logger): Logger variable.

Functions:
    default_logging: Configure logging.
    load_binance_data_spot: Loads data using the binance-connector module.
    load_binance_data_futures: Loads data using the binance-futures-connector module.
    load_yfinance_data: Loads data using the yfinance module.
    load_data: Loads user-provided data.
    load_data_bpd: Load data from a '.bpd' file and save it to the module.
    save_data_bpd: Saves 'data' to a '.bpd' file.
    run_config: Configure the module to execute the strategy as you prefer.
    run: Executes the backtesting process.
    run_animation: Run an animation with your strategy and data.
    plot: Plots your data, highlighting the trades made.
    plot_strategy: Plots statistics for your strategy.
    plot_strategy_decorator: Decorator function for the 'plot_strategy_add' function.
    plot_strategy_add: Add functions and then see them graphed with 'plot_strategy'.
    stats_icon: Shows statistics related to the financial icon.
    stats_trades: Statistics of the trades.

Hidden Functions:
    __load_binance_data: Load data from Binance using a client.
"""

from typing import Callable, Sequence, cast
from datetime import datetime
import logging

from matplotlib.collections import LineCollection, PatchCollection, PathCollection
from matplotlib.dates import DateFormatter, date2num, num2date
from matplotlib.animation import FuncAnimation
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import matplotlib as mpl

import random as rd
import pickle as pk
import pandas as pd
import numpy as np

from time import time

from . import custom_plt as cpl
from . import flex_data as flx
from . import _commons as _cm
from . import exception
from . import strategy
from . import utils
from . import stats

logger:logging.Logger = logging.getLogger(__name__)

def default_logging(level:int = logging.WARNING) -> None:
    """
    Default logging

    Configure logging.

    Args:
        level (int, optional): Logging level.
    """

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

def __load_binance_data(client:Callable, symbol:str = 'BTCUSDT', 
                        interval:str = '1d', start_time:str | None = None, 
                        end_time:str | None = None, statistics:bool = True, 
                        progress:bool = True, data_extract:bool = False
                        ) -> tuple[pd.DataFrame, float] | None:
    """
    Load Binance data.

    Loads data using the Binance client.

    Args:
        client (Callable): Bianance client.
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str | None): Start date for load data in YYYY-MM-DD format.
        end_time (str | None): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
            the module variables will not be assigned with them.

    Returns:
        tuple[DataFrame,float]|None: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """

    # Exceptions.
    if start_time is None or end_time is None:
        raise exception.BinanceError(
            "'start_time' and 'end_time' cannot be None.")

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    ini_time = 0

    def __loop_def(st_t):
        dt = client.klines(symbol=symbol, 
                        interval=interval, 
                        startTime=st_t, 
                        endTime=end, 
                        limit=1000)

        if progress:
            nonlocal ini_time

            if not load_prgs.size:
                ini_time = dt[0][0]
                load_prgs.size = (end-ini_time)//(dt[-1][0]-dt[0][0])
            load_prgs.next()

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

    data.columns = ['timestamp', 
                    'open', 
                    'high', 
                    'low', 
                    'close', 
                    'volume', 
                    'close_time', 
                    'quote_asset_volume', 
                    'number_of_trades', 
                    'taker_buy_base', 
                    'taker_buy_quote', 
                    '_']

    data.index = data['timestamp'].values
    data = data[['open', 'high', 'low', 'close', 'volume']]

    if data.empty or isinstance(data, pd.Series): 
        raise exception.BinanceError('Data empty error.')

    data.index = date2num(pd.to_datetime(data.index, unit='ms', utc=True)) # type: ignore[arg-type]
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
                            start_time:str | None = None, 
                            end_time:str | None = None, statistics:bool = True, 
                            progress:bool = True, data_extract:bool = False
                            ) -> tuple[pd.DataFrame, float] | None:
    """
    Load Binance data from futures.

    Loads data using the binance-connector module from futures.

    Why this differentiation?
        Binance futures data is different from spot data, 
        so it's up to you to decide which one to use based on how you plan to trade.

    Args:
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str | None): Start date for load data in YYYY-MM-DD format.
        end_time (str | None): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
            the module variables will not be assigned with them.

    Returns:
        tuple[DataFrame,float]|None: If 'data_extract' is true, 
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
                            start_time:str | None = None, end_time:str | None = None,
                            statistics:bool = True, progress:bool = True,
                            data_extract:bool = False
                            ) -> tuple[pd.DataFrame, float] | None:
    """
    Load Binance data from spot.

    Loads data using the binance-connector module from spot.

    Why this differentiation?
        Binance spot data is different from futures data, 
        so it's up to you to decide which one to use based on how you plan to trade.

    Args:
        symbol (str, optional): The trading pair.
        interval (str, optional): Data interval, e.g 1s, 1m, 5m, 1h, 1d, etc.
        start_time (str | None): Start date for load data in YYYY-MM-DD format.
        end_time (str | None): End date for load data in YYYY-MM-DD format.
        statistics (bool, optional): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
                        the module variables will not be assigned with them.

    Returns:
        tuple[DataFrame,float]|None: If 'data_extract' is true, 
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

def load_yfinance_data(ticker:str, start:str | None = None, 
                       end:str | None = None, interval:str = '1d', 
                       days_op:int = 365, statistics:bool = True, 
                       progress:bool = True, data_extract:bool = False
                       ) -> tuple[pd.DataFrame, float] | None:
    """
    Load yfinance Data.

    Loads data using the yfinance module.

    Args:
        ticker (str): String of ticker symbols to download.
        start (str | None, optional): Start date for download in YYYY-MM-DD format. 
                              Default is 99 years ago.
        end (str | None, optional): End date for download in YYYY-MM-DD format. 
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
        tuple[DataFrame,float]|None: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """
    days_op = int(days_op)
    if days_op > 365 or days_op < 1:
        raise exception.YfinanceError(f"'days_op' cant be: '{days_op}'.")

    try:
        import yfinance as yf

        yf.set_tz_cache_location('.\\yfinance_cache')

        t = time()
        load_prgs = utils.ProgressBar()
        load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
        if progress:
            load_prgs.reset_size(1)

        data:pd.DataFrame = yf.download(ticker, start=start, 
                           end=end, interval=interval, 
                           progress=False, auto_adjust=False)

        if data.empty: 
            raise exception.YfinanceError('The symbol does not exist.')
        
        data.columns = data.columns.droplevel(1).str.lower()
        data.index = date2num(data.index) # type: ignore[arg-type]
        data_width = utils.calc_width(data.index)

        load_prgs.next()

        if statistics: stats_icon(prnt=True, 
                                  data=data, 
                                  data_icon=ticker.strip(),
                                  data_interval=interval.strip())

        data = data[['open', 'high', 'low', 'close', 'volume']]

        if data_extract:
            return data, data_width

        _cm.__data = data
        _cm.__data_width = data_width
        _cm.__data_icon = ticker.strip()
        _cm.__data_interval = interval.strip()
        _cm.__data_year_days = days_op
        _cm.__data_width_day = utils.calc_day(interval, data_width)

    except ModuleNotFoundError: 
        raise exception.YfinanceError('Yfinance is not installed.')
    except: 
        raise exception.YfinanceError('Yfinance parameters error.')

def load_data(data:pd.DataFrame, icon:str | None = None, 
              interval:str | None = None, days_op:int = 365, 
              statistics:bool = True, progress:bool = True) -> None: 
    """
    Load Any Data.

    Loads data into the system.

    Args:
        data (pd.DataFrame): DataFrame containing the data to load. Must have the 
            following columns: ['open', 'high', 'low', 'close', 
            'volume'].
        icon (str | None, optional): String representing the data icon.
        interval (str | None, optional): String representing the data interval.
        days_op (int, optional): Number of operable days in 1 year. This will be 
            stored to calculate some statistics. Normal values: 365, 252.
        statistics (bool): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
    """

    data.columns = data.columns.str.lower()
    # Exceptions.
    if not all(
        col in data.columns.to_list() 
        for col in ['open', 'high', 'low', 'close']): 
        
        raise exception.DataError(
            utils.text_fix("""
            Some columns are missing columns: 
            ['open', 'high', 'low', 'close']
            """, newline_exclude=True))

    days_op = int(days_op)
    if days_op > 365 or days_op < 1:
        raise exception.DataError(f"'days_op' cant be: '{days_op}'.")

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    if progress:
        load_prgs.reset_size(1)

    if not 'volume' in data.columns:
        data['volume'] = 0

    load_prgs.next()

    data_df = data[['open', 'high', 'low', 'close', 'volume']]
    if data_df.empty or isinstance(data_df, pd.Series): 
        raise exception.DataError('Data empty error.')

    _cm.__data = data_df
    _cm.__data.index.name = 'date'
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index)

    icon = icon or 'None'
    interval = interval or 'None'

    _cm.__data_icon = icon.strip()
    _cm.__data_interval = interval.strip()
    _cm.__data_year_days = days_op
    _cm.__data_width_day = utils.calc_day(interval, _cm.__data_width)

    if statistics: stats_icon(prnt=True)

def load_data_bpd(path:str = 'data.bpd', start:int | None = None, 
                  end:int | None = None, days_op:int | None = None, 
                  statistics:bool = True, progress:bool = True, 
                  data_extract:bool = False
                  ) -> tuple[pd.DataFrame, float] | None:
    """
    Load data from .bpd file

    Load data from a '.bpd' file and save it to the module.

    Info:
        To save a .pbd file use 'save_data_bpd'.

    Args:
        path (str, optional): Path address to the file to be loaded (.bpd).
        start (int  | None, optional): Cut the saved data [start:end].
        end (int | None, optional): Cut the saved data [start:end].
        days_op (int | None, optional): Number of operable days in 1 year. This will be 
            stored to calculate some statistics. Normal values: 365, 252.
            If you want to use the one saved just leave it as 'None'
        statistics (bool): If True, prints statistics of the loaded data.
        progress (bool, optional): If True, shows a progress bar and timer.
        data_extract (bool, optional): If True, the data will be returned and 
            the module variables will not be assigned with them.

    Returns:
        tuple[DataFrame,float]|None: If 'data_extract' is true, 
            a tuple containing the data will be returned (data, data_width).
    """

    if (start and end 
        and ((start > 0 and start < end) 
        or (start < 0 and start > end))):

        raise exception.DataError(
            "The resulting 'data' is empty. Bad: 'start' and 'end'.")

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'DataTimer':lambda x=t: utils.num_align(time()-x)})
    if progress:
        load_prgs.reset_size(1)

    with open(path, "rb") as file:
        data, icon, interval, days_op_load = pk.load(file)

    data.columns = data.columns.str.lower()
    days_op = days_op or days_op_load
    days_op = int(days_op)

    if (not isinstance(data, pd.DataFrame)
        or not all(
        col in data.columns.to_list() 
            for col in ['open', 'high', 'low', 'close'])): 

        raise exception.DataError("Bad data: 'data'.")
    elif not isinstance(icon, str): 
        raise exception.DataError("Bad data: 'icon'.")
    elif not isinstance(interval, str): 
        raise exception.DataError("Bad data: 'interval'.")
    elif days_op > 365 or days_op < 1:
        raise exception.DataError("Bad data: 'days_op'.")

    load_prgs.next()

    data = data.iloc[start:end]

    if statistics: 
        stats_icon(prnt=True, 
                    data=data, 
                    data_icon=icon,
                    data_interval=interval)

    data_width = utils.calc_width(data.index)

    if data_extract:
        return data, data_width

    _cm.__data = data
    _cm.__data_width = data_width
    _cm.__data_icon = icon 
    _cm.__data_interval = interval
    _cm.__data_year_days = days_op
    _cm.__data_width_day = utils.calc_day(interval, _cm.__data_width)

def save_data_bpd(file_name:str = 'data') -> None:
    """
    Save data on .bpd file

    Saves 'data' to a '.bpd' file.

    Info:
        To load a .pbd file use 'load_data_bpd'.

    Args:
        file_name (str, optional): The name or address and name 
            that the file will have (without extension).
    """

    if _cm.__data is None: 
        raise exception.RunError('Data not loaded.')
    elif _cm.__data_icon is None:
        raise exception.RunError('Icon not loaded.')
    elif _cm.__data_interval is None:
        raise exception.RunError('Interval not loaded.')
    elif _cm.__data_year_days is None:
        raise exception.RunError('Year days not loaded.')

    _cm.__data.columns = _cm.__data.columns.str.lower()
    with open(f"{file_name}.bpd", "wb") as file:
        pk.dump((_cm.__data, 
                _cm.__data_icon, 
                _cm.__data_interval, 
                _cm.__data_year_days), file)

def run_config(initial_funds:int = 10000, commission:tuple | float = 0, 
        spread:tuple | float = 0, slippage:tuple | float = 0, 
        gaps:bool = True, ord_closer:bool = True, 
        order_ord:dict | None = None, on_limit:bool = True,
        chunk_size:int | None = None) -> None:
    """
    Run Config

    Configure the module to execute the strategy as you prefer.

    Info:
        CostsValue format:
            (maker, taker) may have an additional tuple indicating 
            that it may be a random number between two numbers.

        For commissions, spreads and slippage, the `CostsValue` format will be followed.

    Args:
        initial_funds (int, optional): Initial amount of funds to start with. Used for 
            statistics. Default is 10,000.
        commission (tuple | float, optional): The commission will be charged 
            for each purchase/sale execution.
        spread (tuple | float, optional): The spread is the separation between 
            the bid and ask price and is used to mark the order book limits.
            There is no variation between maker and taker.
        slippage (tuple | float, optional): It will be calculated at each entry and exit.
            There is no variation between maker and taker.
        gaps (bool, optional): If True, gaps are calculated at the entry price 
            in 'taker' orders.
        ord_closer (bool, optional): If True, orders are executed based on 
            the one closest to the close.
        order_ord (dict | None, optional): Order of orders, original: 
            {'op': 0, 'rd': 1, 'stopLimit': 2, 'stopLoss': 3, 'takeLimit': 4, 
            'takeProfit': 5}, you cannot enter a value equal to or less 
            than 0 or greater than 99. 'op' and 'rd' cannot be modified.
            0 will execute before 99.
        on_limit (bool, optional): If True, the 'stopLimit' and 
            'takeLimit' orders are executed on the same candle if there is price.
        chunk_size (int, optional): BackPy loads the variable space before 
            executing your strategy in chunks, the size of these chunks 
            can take up more memory or make the backtest faster, do not 
            modify this variable if you do not need to, each value represents
            a space saved for position logs. The default value is 10,000.
    """
    # Exceptions
    if initial_funds < 0: 
        raise exception.RunError("'initial_funds' cannot be less than 0.")
    elif (isinstance(order_ord, dict) and any(
            [k not in ('takeProfit', 'takeLimit', 'stopLoss', 'stopLimit')
             or order_ord[k] <= 0 or order_ord[k] > 99 for k in order_ord])):
        raise exception.RunError("'order_ord' bad value or format.")
    elif chunk_size and (not isinstance(chunk_size, int) or chunk_size <= 0):
        raise exception.RunError("'chunk_size' can only be 'int' greater than 0.")

    # Config
    _cm.__init_funds = initial_funds
    _cm.__min_gap = not gaps
    _cm.__orders_nclose = not ord_closer
    _cm.__limit_ig = not on_limit

    if isinstance(order_ord, dict):
        _cm.__orders_order = {k:order_ord[k] for k in order_ord if k in 
                            ('takeProfit', 'takeLimit', 'stopLoss', 'stopLimit')}

    # Costs config
    _cm.__commission = flx.CostsValue(commission, supp_double=True, 
                                      cust_error="Error of 'commission'.")
    _cm.__slippage_pct = flx.CostsValue(slippage, cust_error="Error of 'slippage'.")
    _cm.__spread_pct = flx.CostsValue(spread, cust_error="Error of 'spread'.")
    _cm.__chunk_size = chunk_size or None

def run(cls:type|list[type]|tuple[type], name:str|None = None, prnt:bool = True, 
        progress:bool = True, trades_r:bool = False) -> dict|str|None:
    """
    Run

    Executes your trading strategy.

    Note:
        If your function prints to the console, the loading bar may not 
        function as expected.
        To delete a backtest use the function: 'backpy._commons.del_backtest'.

    Args:
        cls (type|list[type]|tuple[type]): A class inherited from `StrategyClass` where the strategy is implemented.
        name (str|None, optional): Backtest name, None = cls.__name__, 
            if the name is duplicated a number will be added at the end.
        prnt (bool, optional): If True, prints trade statistics. If False, returns a string 
                    with the statistics. Default is True.
        progress (bool, optional): If True, shows a progress bar and timer. Default is True.
        trades_r (bool, optional): If True, the dictionary with the backtest 
            data is returned and will not be saved.

    Returns:
        dict|str|None: Statistics or backtest.
    """
    # Exceptions.
    if _cm.__data is None: 
        raise exception.RunError('Data not loaded.')
    elif not isinstance(cls, (tuple, list)):
        cls = [cls]

    instances = []
    for st in cls:
        if not issubclass(st, strategy.StrategyClass):
            raise exception.RunError(
                f"'{st.__name__}' is not a subclass of 'strategy.StrategyClass'.")
        elif getattr(st, '__abstractmethods__'):
            raise exception.RunError(
                "The implementation of the 'next' abstract method is missing.")

        instances.append(st(data=_cm.__data))

    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)

    # Progress bar variables
    t = time()
    step_t = time()
    steph_index = 0
    step_history = np.zeros(10)
    skip = max(1, _cm.__data.shape[0] // _cm.max_bar_updates)

    load_prgs = utils.ProgressBar()

    def prediction_calc() -> float|np.floating:
        """
        Prediction calc

        Calculate the prediction so you can add it to the progress bar.

        Return:
            float|np.floating: Prediction.
        """
        nonlocal steph_index

        if _cm.__data is None: 
            return 0.

        step_history[steph_index % 10] = time()-step_t
        steph_index += 1
        return time()-t + (
            ((md:=np.median(step_history)) + (time()-t - md*f)/f) 
            * (_cm.__data.shape[0]-f))

    # Progress bar config
    if _cm.run_timer:
        load_prgs.adder_add({
            'RunTimer':lambda x=t: utils.num_align(time()-x),
            'TimerPredict':lambda: utils.num_align(prediction_calc())
        })
    load_prgs.adder_add({'StepTime':lambda: utils.num_align(time()-step_t)})

    if progress:
        load_prgs.reset_size(_cm.__data.shape[0])

    # Run strategy
    balance_rec = []
    for f, _ in enumerate(_cm.__data.index):
        f += 1

        # Progress bar
        if (progress and (f % skip == 0 or f >= _cm.__data.shape[0]) 
            and _cm.__data.shape[0] >= f):
            load_prgs._step = f-1
            load_prgs.next()
        step_t = time()

        for i in instances:
            i._StrategyClass__before(index=f, balance=balance_rec)
            if len(balance_rec) == f:
                balance_rec[f-1] = i._StrategyClass__balance
            else:
                balance_rec.append(i._StrategyClass__balance)
    if _cm.run_timer and not progress:
        print(f'RunTimer: {utils.num_align(time()-t)}') 

    # Save variables
    positions_list:np.ndarray|None = None
    positions_open_list = []
    for i in instances:
        positions_open_list.extend(i._StrategyClass__positions)

        if not positions_list is None:
            positions_list = np.concatenate((positions_list, i._StrategyClass__pos_record[
                :i._StrategyClass__pos_record._pos]))
        else:
            positions_list = np.array(i._StrategyClass__pos_record[
                :i._StrategyClass__pos_record._pos])

    act_trades = pd.DataFrame(positions_open_list).dropna(axis=1, how='all')
    trades = pd.DataFrame(positions_list).dropna(axis=1, how='all')

    if not act_trades.empty: 
        trades = pd.concat([trades, act_trades], ignore_index=True)

    backtest = {
        'name':_cm.__gen_fname(name or cls[0].__name__, _cm.__backtests), 
        'trades':trades, 
        'balance_rec':pd.Series(balance_rec, index=_cm.__data.index),
        'init_funds':_cm.__init_funds,
        'd_year_days':_cm.__data_year_days,
        'd_width_day':_cm.__data_width_day,
        'd_width':_cm.__data_width
    }

    if trades_r:
        return backtest
    elif not trades.empty:
        _cm.__backtests.append(backtest)

    try: 
        return stats_trades(prnt=prnt)
    except: pass

def run_animation(cls:type, candles:int = 100, interval:int = 100, 
                  operation_route:bool | None = True, pad:bool = False,
                  panel:str = 'new', style:str | None = 'last', 
                  style_c:dict | None = None, block:bool = True) -> None:
    """
    Run animation

    Run an animation with your strategy and data.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Args:
        cls (type): A class inherited from `StrategyClass` where the strategy is implemented.
        candles (int, optional): Number of candles the animation will have.
        interval (int, optional): Delay between frames in milliseconds. Only greater or equal than 100.
        operation_route (bool | None, optional): True draws the entire operation, 
            False only draws open/close, and None draws the entire position 
            except the colored rectangle.
        pad (bool, optional): If it is True, a pad is added to the right 
            and the end of the candles is more in the center.
        panel (str, optional): To create a new window or add a panel, 
            only 'new' or 'add' are possible.
        style (str|None, optional): Color style. 
            If you leave it as 'last' the last one will be used.
        style_c (dict|None, optional): Customize the defined style by 
            modifying the dictionary. To know what to modify, 
            read the docstring of 'def_style'.
        block (bool, optional): If True, pauses script execution until all 
            figure windows are closed. If False, the script continues running 
            after displaying the figures. Default is True.
    """
    # Exceptions.
    panel = panel.lower()
    valid_styles = {'random', 'last'} | set(_cm.__plt_styles.keys())

    if _cm.__data is None: 
        raise exception.RunError('Data not loaded.')
    elif panel not in ('new', 'add'):
        raise exception.RunError(
            f"'{panel}' Not a valid option for: 'panel'.")
    elif not issubclass(cls, strategy.StrategyClass):
        raise exception.RunError(
            f"'{cls.__name__}' is not a subclass of 'strategy.StrategyClass'.")
    elif getattr(cls, '__abstractmethods__'):
        raise exception.RunError(
            "The implementation of the 'next' abstract method is missing.")
    elif (not style is None and (style:=style.lower()) not in valid_styles):
        raise exception.PlotError(f"'{style}' Not a style.")
    elif interval < 100:
        raise exception.PlotError(f"'interval' it can only be greater or equal than 100.")

    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)

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

    instance = cls(data=_cm.__data)
    fig = plt.figure(figsize=(16,8))
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, 
                                  colspan=1, sharex=ax1)

    gdir = plt_colors.get('gdir', False)
    cpl.custom_ax(ax1, plt_colors['bg'], edge=gdir)
    cpl.custom_ax(ax2, plt_colors['bg'], edge=gdir)

    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    # Init.
    f = 0
    utils.plot_candles(ax1, _cm.__data.iloc[[f]], _cm.__data_width*0.9,
                       color_up=market_colors.get('u', 'g'),
                       color_down=market_colors.get('d', 'r'))
    utils.plot_volume(ax2, _cm.__data.iloc[[f]]['volume'], _cm.__data_width, 
                      color=plt_colors.get('vol', 'tab:orange'))
    instance._StrategyClass__before(index=f)
    trades_last = pd.DataFrame(
        instance._StrategyClass__positions).dropna(axis=1, how='all')

    def update(_):
        """
        Update

        Update animation.
        """

        nonlocal f, trades_last, pad

        if (_cm.__data is None 
            or not isinstance(_cm.__data_width, float) 
            or f+1 >= len(_cm.__data)):
            return ()

        f += 1
        l = f-candles
        if l <= 0: l = 0

        utils.plot_candles(ax1, _cm.__data.iloc[[f]], _cm.__data_width*0.9,
                        color_up=market_colors.get('u', 'g'),
                        color_down=market_colors.get('d', 'r'))
        utils.plot_volume(ax2, _cm.__data.iloc[[f]]['volume'], _cm.__data_width, 
                      color=plt_colors.get('vol', 'tab:orange'))
        instance._StrategyClass__before(index=f)

        act_trades = pd.DataFrame(
            instance._StrategyClass__positions).dropna(axis=1, how='all')
        trades = pd.DataFrame(
            instance._StrategyClass__pos_record[
                :instance._StrategyClass__pos_record._pos]
        ).dropna(axis=1, how='all')
        
        if not act_trades.empty: 
            trades = pd.concat([trades, act_trades], ignore_index=True)


        dupl_trades = pd.concat([trades, trades_last]).drop_duplicates(keep=False)
        trades_mask = trades.apply(tuple, axis=1).isin(dupl_trades.apply(tuple, axis=1))
        trades_uniq = trades.loc[trades_mask].copy()
        trades_last = pd.concat([trades, trades_last]).drop_duplicates()

        if not trades_uniq.empty:
            utils.plot_position(trades_uniq, ax1, 
                            color_take=market_colors.get('u', 'g'),
                            color_stop=market_colors.get('d', 'r'),
                            operation_route=operation_route, alpha=0.3, 
                            alpha_arrow=0.8)

        pad_l = (abs(_cm.__data.index[l:f][-1]-_cm.__data.index[l:f][0])/2 if pad else 0)
        ax1.set_ylim(_cm.__data.iloc[l:f]['low'].to_numpy(dtype=float).min()*0.99,
                     _cm.__data.iloc[l:f]['high'].to_numpy(dtype=float).max()*1.01,)
        ax1.set_xlim(_cm.__data.index.values[l:f][0]-_cm.__data_width*(candles*0.03), 
                    _cm.__data.index.values[l:f][-1]+_cm.__data_width*2*(candles*0.03)+pad_l)
        ax2.set_ylim(top=_cm.__data.iloc[l:f]['volume'].to_numpy(dtype=float).max()*1.1 or 1)

        def axes_xlim(ax:Axes):
            """
            Axes xlim

            Delete: 'LineCollection', 'PatchCollection' and 'PathCollection'
                if x value is less than 'data.index[l:f][0]-data_width/2'.

            Args:
                ax (Axes): Axis.
            """

            if _cm.__data is None or _cm.__data_width is None:
                return

            for coll in ax.collections:
                if isinstance(coll, LineCollection):
                    segs = coll.get_segments()[0]
                    xs = segs[0][0]
                elif isinstance(coll, PatchCollection):
                    paths = coll.get_paths()[0]
                    if not isinstance(paths.vertices, np.ndarray):
                        continue

                    xs = np.max(paths.vertices[:, 0])

                elif isinstance(coll, PathCollection):
                    offsets = coll.get_offsets()
                    if not isinstance(offsets, np.ndarray):
                        continue

                    xs = offsets[:, 0][0]
                else:
                    continue

                if xs < _cm.__data.index[l:f][0]-_cm.__data_width/2:
                    coll.remove()

        axes_xlim(ax1); axes_xlim(ax2)
        return ()

    date_format = DateFormatter('%H:%M %d-%m-%Y')

    ax2.yaxis.set_major_formatter(lambda y, _: str(y.real))
    ax1.yaxis.set_major_formatter(lambda y, _: str(y.real))
    ax1.xaxis.set_major_formatter(date_format)

    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelleft=False)

    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='y', labelleft=False)

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    anim = FuncAnimation(fig, update, frames=len(_cm.__data),
                        interval=interval, blit=False)

    cpl.add_window(
        fig=fig,
        title=f'Run animation - {style}',
        block=block,
        anim=anim,
        interval=interval,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='limited'
    )

def plot(log:bool = False, progress:bool = True, name:list[str|int|None]|str|int|None = None,
         position:str = 'complex', panel:str = 'new', style:str | None = 'last', 
         draw_style:str | None = None, draw_style_vol:str | None = None, 
         style_c:dict | None = None, block:bool = True) -> None:
    """
    Plot Graph with Trades.

    Plots your data, highlighting the trades made.

    Simbol guide:
        - 'x': Position close.
        - '^': Buy position.
        - 'v': Sell position.

    All color styles:
        'lightmode', 'darkmode', 'sunrise', 'mintfresh', 'skyday', 
        'emberday', 'lavenderblush', 'peachpuff', 'sunrisedusk', 
        'embernight', 'obsidian', 'neonforge', 'carbonfire', 
        'datamatrix', 'terminalblood', 'plasmacore'.

    Draw styles:
        'candle': Typical Japanese candle. 
        'line': A line traces the outline of the closures.
        'none': The data is not drawn.

    Volume draw styles:
        'bar': Typical bar. 
        'none': The volume is not drawn.

    Args:
        log (bool, optional): If True, plots data using a logarithmic scale. 
            Default is False.
        progress (bool, optional): If True, shows a progress bar and timer. 
            Default is True.
        name (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract trades from, None = -1, 
            you can add multiple by passing an list.
        position (str, optional): Specifies how positions are drawn. Options 
            are 'complex' or 'simple'. If None or 'none', positions will not 
            be drawn. Default is 'complex'. The "complex" option may take longer 
            to process.
        panel (str, optional): To create a new window or add a panel, 
            only 'new' or 'add' are possible.
        style (str | None, optional): Color style. 
            If you leave it as 'last' the last one will be used.
        draw_style (str | None, optional): Change the drawing style of the data.
            Current types: 'candle', 'line', 'none'. None = 'line'.
        draw_style_vol (str | None, optional): Change the drawing style of the
            volumen. Current types: 'bar', 'none'. None = 'bar'.
        style_c (dict | None, optional): Customize the defined style by 
            modifying the dictionary. To know what to modify, 
            read the docstring of 'def_style'.
        block (bool, optional): If True, pauses script execution until all 
            figure windows are closed. If False, the script continues running 
            after displaying the figures. Default is True.
    """

    # Exceptions.
    panel = panel.lower()
    valid_style = {'random', 'last'} | set(_cm.__plt_styles.keys())
    draw_style = draw_style or 'line'
    draw_style_vol = draw_style_vol or 'bar'

    if _cm.__data is None or not type(_cm.__data) is pd.DataFrame or _cm.__data.empty: 
        raise exception.PlotError('Data not loaded.')
    elif position and not position.lower() in ('complex', 'simple', 'none'):
        raise exception.PlotError(
            f"'{position}' Not a valid option for: 'position'.")
    elif panel not in ('new', 'add'):
        raise exception.PlotError(
            f"'{panel}' Not a valid option for: 'panel'.")
    elif (not style is None and not (style:=style.lower()) in valid_style):
        raise exception.PlotError(f"'{style}' Not a style.")
    elif not draw_style in {'none', 'line', 'candle'}:
        raise exception.PlotError(f"'{draw_style}' Not a draw style.")
    elif not draw_style_vol in {'none', 'bar'}:
        raise exception.PlotError(f"'{draw_style_vol}' Not a draw style.")

    # Corrections.
    _cm.__data.index = utils.correct_index(_cm.__data.index)
    _cm.__data_width = utils.calc_width(_cm.__data.index, True)

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

    t = time()
    load_prgs = utils.ProgressBar()
    load_prgs.adder_add({'PlotTimer':lambda x=t: utils.num_align(time()-x)})
    if progress: 
        load_prgs.reset_size(5)

    fig = plt.figure(figsize=(16,8))
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, 
                                  colspan=1, sharex=ax1)

    gdir = plt_colors.get('gdir', False)
    cpl.custom_ax(ax1, plt_colors['bg'], edge=gdir)
    cpl.custom_ax(ax2, plt_colors['bg'], edge=gdir)

    if log: 
        ax1.semilogy(); ax2.semilogy()

    load_prgs.next()
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    # Draw style
    match draw_style:
        case 'candle':
            utils.plot_candles(ax1, _cm.__data, _cm.__data_width*0.9,
                            color_up=market_colors.get('u', 'g'),
                            color_down=market_colors.get('d', 'r'))
        case 'line':
            utils.plot_line(ax1, _cm.__data['close'], _cm.__data_width,
                            color_up=market_colors.get('u', 'g'),
                            color_down=market_colors.get('d', 'r'))
        case 'none' | _:
            pass
    load_prgs.next()

    # Draw style volume
    match draw_style_vol:
        case 'bar':
            utils.plot_volume(ax2, _cm.__data.loc[:, 'volume'], _cm.__data_width, 
                            color=plt_colors.get('vol', 'tab:orange'))
        case 'none' | _:
            pass
    load_prgs.next()

    if position and position.lower() != 'none' and len(_cm.get_backtest_names()) > 0:

        trades = _cm.__get_trades(name)
        if not trades.empty:
            utils.plot_position(trades, ax1, 
                            color_take=market_colors.get('u', 'green'),
                            color_stop=market_colors.get('d', 'red'),
                            operation_route=position.lower() == 'complex',
                            alpha=0.3, alpha_arrow=0.8)

    load_prgs.next()
    date_format = DateFormatter('%H:%M %d-%m-%Y')

    ax2.yaxis.set_major_formatter(lambda y, _: str(y.real))
    ax1.yaxis.set_major_formatter(lambda y, _: str(y.real))
    ax1.xaxis.set_major_formatter(date_format)

    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelleft=False)

    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='y', labelleft=False)

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    ix_date:Sequence = num2date(_cm.__data.index) # type: ignore[arg-type]

    s_date = ".".join(str(val) for val in 
                    [ix_date[0].day, ix_date[0].month, 
                    ix_date[0].year])
    
    e_date = ".".join(str(val) for val in 
                    [ix_date[-1].day, ix_date[-1].month, 
                    ix_date[-1].year])

    load_prgs.next()
    cpl.add_window(
        fig=fig,
        title=f"Back testing: '{_cm.__data_icon}' {s_date}~{e_date} - {style}",
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total',
    )

def plot_strategy(name:list[str|int|None]|str|int|None = None, 
                  log:bool = False, view:str = 'b/w/r/e',  
                  custom_graph:dict = {}, panel:str = 'new',
                  style:str | None = 'last', style_c:dict | None = None, 
                  block:bool = True) -> None:
    """
    Plot Strategy Statistics.

    Plots statistics for your strategy.

    Available Graphics:
        - 'b' = Balance graph.
        - 'e' = Equity graph.
        - 'p' = Profit graph.
        - 'r' = Return graph.
        - 'w' = Winnings graph.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Args:
        name (list[str|int|None]|str|int|None, optional): 
            Backtest names to extract data from, None = -1, 
            you can add multiple by passing an list.
        log (bool, optional): If True, plots data using a logarithmic scale. 
            Default is False.
        view (str, optional): Specifies which graphics to display. 
            Default is 'b/w/r/e'. Maximum 8.
        custom_graph (dict, optional): Custom graph, a dictionary with 
            'name':'function' where the function will 
            be passed: 'ax', '_cm.__trades', '_cm.__data', 'log'.
            To avoid visual problems, I suggest using 
            'trades.index' as the x-axis or normalizing the axis.
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

    for i in custom_graph: plot_strategy_add(custom_graph[i], i)

    trades_data = _cm.__get_strategy(name)
    trades = _cm.__get_trades(name)

    # Exceptions.
    panel = panel.lower()
    valid_style = {'random', 'last'} | set(_cm.__plt_styles.keys())

    if trades.empty: 
        logger.warning('Trades not loaded')
        return
    elif not 'profit' in trades.columns:  
        logger.warning('No data to see')
        return
    elif (not style is None and not (style:=style.lower()) in valid_style):
        raise exception.StatsError(f"'{style}' Not a style.")
    elif panel not in ('new', 'add'):
        raise exception.StatsError(
            f"'{panel}' Not a valid option for: 'panel'.")

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

    fig = plt.figure(figsize=(16,8))

    init_data:dict = {col: 0 for col in trades.columns}
    init_data['positionDate'] = trades['positionDate'].iloc[0]
    init_p = pd.DataFrame([init_data])

    trades = pd.concat([
        init_p, trades], ignore_index=True)

    gdir = plt_colors.get('gdir', False)
    market_colors = plt_colors.get('mk', {'u':'g', 'd':'r'})

    graphics = ['p','w','r','e','b']
    graphics.extend(list(_cm.__custom_plot.keys()))

    axes, r_view = cpl.ax_view(view=view, graphics=graphics)

    for i,v in enumerate(r_view):
        ax = axes[i]

        cpl.custom_ax(ax, plt_colors['bg'], edge=gdir)
        ax.tick_params('x', which='both', bottom=False, 
                       top=False, labelbottom=False)
        ax.tick_params('y', which='both', left=False, 
                       right=False, labelleft=False)

        ax.yaxis.set_major_formatter(lambda y, _: str(y.real))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M %d-%m-%Y'))

        pos_date = trades['positionDate']
        ax.set_xlim(pos_date.dropna().iloc[0]-(trades_data['d_width']*len(pos_date)/10), 
                    pos_date.dropna().iloc[-1]+(trades_data['d_width']*len(pos_date)/10))

        y_limit:Callable = lambda values, comp=1: (np.min(values)-comp, 
                                                    np.max(values)*1.05+comp)

        match v:
            case 'b':
                values = trades_data['balance_rec'].values
                color = (market_colors.get('u', 'g') 
                         if values[-1] > trades_data['init_funds']
                         else market_colors.get('d', 'r'))
                ax.plot(trades_data['balance_rec'].index, values, c=color, 
                        label='Balance.', ds='steps-post')

                ax.set_ylim(y_limit(values))
                if log: ax.set_yscale('symlog')
            case 'p':
                values = trades['profit'].cumsum()
                color = (market_colors.get('u', 'g') if trades['profit'].to_numpy().sum() > 0 
                         else market_colors.get('d', 'r'))

                ax.plot(pos_date, values, c=color, 
                        label='Profit.', ds='steps-post')
   
                ax.set_ylim(y_limit(values))

                if log: ax.set_yscale('symlog')
            case 'w':
                values = (trades['profitPer'].apply(
                            lambda row: 1 if row>0 else -1)).cumsum()
                color = market_colors.get('u', 'g')

                ax.plot(pos_date, values, c=color, 
                        label='Winnings.', ds='steps-post')
                ax.set_ylim(y_limit(values))
            case 'e':
                with np.errstate(over='ignore'):
                    values = np.cumprod(1 + trades['profitPer'] / 100)
                if np.isinf(values).any():
                    values = np.zeros_like(values)

                mltp:pd.Series[float] = (1 + trades.loc[:, 'profitPer'] / 100)
                color = (market_colors.get('u', 'g')
                         if _cm.c_tf(mltp.prod()) > 1 
                         else market_colors.get('d', 'r'))

                ax.plot(pos_date, values, c=color, 
                        label='Equity.', ds='steps-post')

                values = y_limit(values, 0.01)
                ax.set_ylim(y_limit(values, 0.01))

                if log: ax.set_yscale('symlog')
            case 'r':
                values = trades['profitPer'].cumsum()

                profit_per:pd.Series[float] = trades['profitPer']
                color = (market_colors.get('u', 'g') if profit_per.sum() > 0 
                         else market_colors.get('d', 'r'))

                ax.plot(pos_date, values, c=color, 
                        label='Return.', ds='steps-post')
                ax.set_ylim(y_limit(values))

                if log: ax.set_yscale('symlog')
            case key if key in _cm.__custom_plot.keys():
                _cm.__custom_plot[v](ax, trades, 
                                    _cm.__data, log)
            case _: pass
        ax.legend(loc='upper left')

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0, right=1, top=1, 
                        bottom=0, wspace=0, hspace=0)

    plt.xticks([])

    import random
    cpl.add_window(
        fig=fig,
        title=f'Strategy statistics - {style}',
        block=block,
        style=plt_colors,
        new=True if panel == 'new' else False,
        toolbar='total'
    )

def plot_strategy_decorator(name:str) -> Callable:
    """
    Add statistics for plot decorator.

    Use a decorator to add the function to 
        'custom_plot' so you can run it in 'plot_strategy'.

    The function will be passed: 
        'ax', 'trades', '_cm.__data', 'log' in order.

    To avoid visual problems, I suggest using 
        'trades.index' as the x-axis or normalizing the axis.

    Args:
        name (str): Name with which it will be called.

    Returns:
        Callable: 'plot_strategy_add'.
    """

    return lambda x: plot_strategy_add(x, name)

def plot_strategy_add(func:Callable, name:str) -> Callable:
    """
    Add statistics for plot.

    Add functions and then see them graphed with 'plot_strategy'.

    Args:
        func (Callable): Function, to which this will be passed in order: 
            'ax', 'trades', '_cm.__data', 'log'.
            To avoid visual problems, I suggest using 
            'trades.index' as the x-axis or normalizing the axis.
        name (str): Name with which it will be called.

    Returns:
        Callable: 'func' param.
    """

    if not name or name in _cm.__custom_plot.keys() or not callable(func):
        raise exception.StatsError("Error assigning value to '__custom_plot'.")
    _cm.__custom_plot[name.strip()] = func
    return func

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
        'Last price':[utils.round_r(_cm.c_tf(data['close'].iloc[-1]),2),
                      _cm.__COLORS['BOLD']],
        'Maximum price':[utils.round_r(_cm.c_tf(data['high'].max()),2),
                         _cm.__COLORS['GREEN']],
        'Minimum price':[utils.round_r(_cm.c_tf(data['low'].min()),2),
                         _cm.__COLORS['RED']],
        'Maximum volume':[utils.round_r(_cm.c_tf(data['volume'].max()), 2),
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
    op_years = abs(
        (_cm.c_tf(trades['date'].iloc[-1]) - trades['date'].iloc[0])/
        (trades_data['d_width_day']*trades_data['d_year_days']))

    # Annualized trades calc.
    trades_calc = trades.copy()
    trades_calc['year'] = ((_cm.c_tf(trades_calc['date']) - trades_calc['date'].iloc[0]) / 
                  (_cm.c_tf(trades_calc['date'].iloc[-1]) - trades_calc['date'].iloc[0]) * 
                  op_years).astype(int)

    trades_calc['diary'] = ((_cm.c_tf(trades_calc['date']) - trades_calc['date'].iloc[0]) / 
                (_cm.c_tf(trades_calc['date'].iloc[-1]) - trades_calc['date'].iloc[0]) * 
                op_years*trades_data['d_year_days']).astype(int)

    trades_calc['duration'] = (trades_calc['positionDate']
                               -trades_calc['date'])/trades_data['d_width_day']

    ann_profit = trades_calc.groupby('year')['profit'].sum()
    diary_profit:pd.Series = pd.Series(trades_calc.groupby('diary')['profit'].sum())

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
        trades_calc['multiplier'] = 1 + trades_calc['profitPer'] / 100

        nan_inf = lambda x: x.where(~np.isinf(x), np.nan)
        multiplier_cumprod = nan_inf(trades_calc.loc[:, 'multiplier'].cumprod().dropna())

        ann_return = nan_inf(trades_calc.groupby('year')['multiplier'].prod())
        diary_return = nan_inf(trades_calc.groupby('diary')['multiplier'].prod())

        text = utils.statistics_format({
        'Trades':[len(trades.index),
                  _cm.__COLORS['BOLD']+_cm.__COLORS['CYAN']],

        'Op years':[utils.round_r(op_years, 2), _cm.__COLORS['CYAN']],

        'Return':[str(_return:=utils.round_r((_cm.c_tf(trades_calc.loc[:, 'multiplier'].prod())-1)*100,2))+'%',
                  _cm.__COLORS['GREEN'] if float(_return) > 0 else _cm.__COLORS['RED'],],

        'Profit':[str(_profit:=utils.round_r(trades['profit'].to_numpy().sum(),2)),
                _cm.__COLORS['GREEN'] if float(_profit) > 0 else _cm.__COLORS['RED'],],

        'Gross earnings':[utils.round_r((trades['profit'][_cm.c_tf(trades['profit'])>0].sum()
                           if not pd.isna(trades['profit']).all() else 0), 4),
                        _cm.__COLORS['GREEN']],

        'Gross losses':[utils.round_r(abs(trades['profit'][_cm.c_tf(trades['profit'])<=0].sum())
                           if not pd.isna(trades['profit']).all() else 0, 4),
                        _cm.__COLORS['RED']],

        'Max return':[str(utils.round_r((multiplier_cumprod.max()-1)*100,2))+'%'],

        'Return from max':[str(utils.round_r(
            -((multiplier_cumprod.max()-1)
            - (_cm.c_tf(trades_calc.loc[:, 'multiplier'].prod())-1))*100,2))+'%'],

        'Days from max':[str(utils.round_r(
            (_cm.c_tf(trades_calc['date'].dropna().iloc[-1])
                - trades_calc['date'].dropna().loc[
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

        'Average ratio':[utils.round_r(stats.average_ratio(trades), 2),
                        _cm.__COLORS['YELLOW'],],

        'Average return':[str(utils.round_r((
                trades_calc.loc[:, 'multiplier'].dropna().to_numpy().mean()-1)*100,2))+'%',
            _cm.__COLORS['YELLOW'],],

        'Average profit':[str(utils.round_r(trades.loc[:, 'profit'].mean(),2))+'%',
                    _cm.__COLORS['YELLOW'],],

        'Profit fact':[_profit_fact:=utils.round_r(stats.profit_fact(trades.loc[:, 'profit']), 3),
                _cm.__COLORS['GREEN'] if float(_profit_fact) > 1 else _cm.__COLORS['RED'],],

        'Return diary std':[(_return_std:=utils.round_r(np.std((diary_return.dropna()-1)*100,ddof=1), 2)),
                    _cm.__COLORS['YELLOW'] if float(_return_std) > 1 else _cm.__COLORS['GREEN'],],

        'Profit diary std':[(_profit_std:=utils.round_r(np.std(diary_profit.dropna(),ddof=1), 2)),
                      _cm.__COLORS['YELLOW'] if float(_profit_std) > 1 else _cm.__COLORS['GREEN'],],

        'Math hope':[_math_hope:=utils.round_r(stats.math_hope(trades.loc[:, 'profit']), 2),
            _cm.__COLORS['GREEN'] if float(_math_hope) > 0 else _cm.__COLORS['RED'],],

        'Math hope r':[_math_hope_r:=utils.round_r(
                stats.math_hope_relative(trades, trades.loc[:, 'profitPer']), 2),
            _cm.__COLORS['GREEN'] if float(_math_hope_r) > 0 else _cm.__COLORS['RED'],],

        'Historical var':[0 if trades['profit'].dropna().empty else utils.round_r(
                            stats.var_historical(trades.loc[:, 'profit'].dropna()), 2)],

        'Parametric var':[0 if trades['profit'].dropna().empty else utils.round_r(
                            stats.var_parametric(trades.loc[:, 'profit'].dropna()), 2)],

        'Sharpe ratio':[utils.round_r(stats.sharpe_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            trades_data['d_year_days'],
            (diary_return.dropna()-1)*100), 2)],

        'Sharpe ratio$':[utils.round_r(stats.sharpe_ratio(
            np.average(ann_profit),
            trades_data['d_year_days'],
            diary_profit), 2)],

        'Sortino ratio':[utils.round_r(stats.sortino_ratio(
            (ann_return.prod()**(1/op_years)-1)*100,
            trades_data['d_year_days'],
            (diary_return.dropna()-1)*100), 2)],

        'Sortino ratio$':[utils.round_r(stats.sortino_ratio(
            np.average(ann_profit),
            trades_data['d_year_days'],
            diary_profit), 2)],

        'Duration ratio':[utils.round_r(
            _cm.c_tf(trades_calc['duration'].to_numpy().sum())/len(trades.index), 2),
            _cm.__COLORS['CYAN']],

        'Payoff ratio':[utils.round_r(stats.payoff_ratio(trades.loc[:, 'profitPer']), 3)],

        'Expectation':[utils.round_r(stats.expectation(trades.loc[:, 'profitPer']))],

        'Skewness':[utils.round_r((diary_return.dropna()-1).skew(), 2)],

        'Kurtosis':[utils.round_r((diary_return.dropna()-1).kurt(), 2)],

        'Average winning op':[str(utils.round_r(trades.loc[:, 'profitPer'][
                _cm.c_tf(trades['profitPer']) > 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['GREEN']],

        'Average losing op':[str(utils.round_r(trades.loc[:, 'profitPer'][
                _cm.c_tf(trades['profitPer']) < 0].dropna().mean(), 2))+'%',
            _cm.__COLORS['RED']],

        'Average duration winn':[str(utils.round_r(trades_calc.loc[:, 'duration'][
                _cm.c_tf(trades_calc['profitPer']) > 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Average duration loss':[str(utils.round_r(trades_calc.loc[:, 'duration'][
                _cm.c_tf(trades_calc['profitPer']) < 0].dropna().mean()))+'d',
                _cm.__COLORS['CYAN']],

        'Daily frequency op':[utils.round_r(
            len(trades.index) / (op_years*trades_data['d_year_days']), 2),
            _cm.__COLORS['CYAN']],

        'Max consecutive winn':[trades_csct.max(),
                                _cm.__COLORS['GREEN']],

        'Max consecutive loss':[abs(_cm.c_tf(trades_csct.min())),
                                _cm.__COLORS['RED']],

        'Max losing streak':[abs(trades_streak.min())],

        'Max drawdown':[str(round(
            stats.max_drawdown(multiplier_cumprod)*100,1)) + '%'],

        'Average drawdown':[str(-round(np.mean(
            stats.get_drawdowns(multiplier_cumprod))*100, 1)) + '%'],

        'Max drawdown$':[str(round(
            stats.max_drawdown(trades['profit'].dropna().cumsum()+
                               trades_data['init_funds'])*100,1)) + '%'],

        'Average drawdown$':[str(-round(np.mean(
            stats.get_drawdowns(trades['profit'].dropna().cumsum()+
                                trades_data['init_funds']))*100, 1)) + '%'],

        'Long exposure':[str(round(
            stats.long_exposure(trades.loc[:, 'typeSide'])*100)) + '%',
            _cm.__COLORS['GREEN']],

        'Winnings':[str(round(stats.winnings(trades.loc[:, 'profitPer'])*100)) + '%'],

        }, f"---Statistics of '{trades_data['name']}'---")

    text = text if _cm.dots else text.replace('.', ',')
    if data: 
        text += (stats_icon(False) or '')

    if prnt: print(text)
    else: return text
