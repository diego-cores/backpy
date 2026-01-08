"""
Strategy module

This module contains the main class that must be inherited to create your 
own strategy.

Variables:
    logger (Logger): Logger variable.

Classes:
    StrategyClass: This is the class you have to inherit to create your strategy.

Functions:
    idc_decorator: Create your own indicator.

Hidden Functions:
    _data_info: Gathers information about the dataset.
"""

from bs4.element import ResultSet
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Any, cast
from functools import wraps
from uuid import uuid4
from time import time
import logging

from . import flex_data as flx
from . import _commons as cm_
from . import exception
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

def idc_decorator(func:Callable) -> Callable[..., flx.DataWrapper]:
    """
    Indicator decorator

    Create your own indicator.

    Decorate a function with this to mark it as an indicator. The function:
        - It must be defined in the class that inherits StrategyClass.
        - Must accept at least one argument (usually 'data').
        - Will be called automatically with the full dataset.
        - Will not receive 'self' (it's treated as a static method).
        - Must return a value accepted by 'DataWrapper'.
        - The returned sequence must match the length of 'data'.

    Usage:
        @idc_decorator
        def idc_test(data):
            ema = data['close'].ewm(span=20, adjust=False).mean()
            return ema # Will be wrapped in a DataWrapper automatically

    Important:
        - The indicator will be calculated **only once**, and stored.
        - Access it from `next()` using `self.idc_test`.
        - You are responsible for ensuring the indicator logic does not use future data.
        (e.g. avoid calculating max of all data at step 1, which would invalidate the backtest).

    Args:
        func (Callable): Function.

    Returns:
        Callable[...,flx.DataWrapper]: Function.
    """

    setattr(func, '_uidc', True)
    return func

def _data_info() -> tuple[str|None, str|None, float|None]:
    """
    Data Info

    Returns all 'data' variables except `__data`.

    Returns:
        tuple[str|None,str|None,float|None]: 
            A tuple containing the following variables in order:
            - __data_interval (str): Data interval.
            - __data_icon (str): Data icon.
            - __data_width (int): Data index width.
    """

    return cm_.__data_interval, cm_.__data_icon, cm_.__data_width

class StrategyClass(ABC):
    """
    StrategyClass

    This is the class you have to inherit to create your strategy.

    To use the functions, use the `self` instance. Create your strategy 
    within the `StrategyClass.next()` structure.

    Attributes:
        open: 'open' values from `data`.
        high: 'high' values from `data`.
        low: 'low' values from `data`.
        close: 'close' values from `data`.
        volume: 'volume' values from `data`.
        date: Index list from `data`.
        hour: Hour from 0 to 23 float value, based on '__day_width'.
        width: Data width from `__data_width`.
        icon: Data icon from `__data_icon`.
        interval: Data interval from `__data_interval`.

    Private Attributes:
        __balance: Balance based on initial funds.
        __balance_rec: List with all balances by date.
        __data: DataWrapper containing cuted data.
        __data_adf: DataFrame containing all data.
        __data_awr: DataWrapper containing all data.
        __day_width: 1-day width of the index, calculated on an 
            interval basis, global variable '__data_width_day'.
        __data_index: Index to cut data.
        __data_dates: DataWrapper with index of __data.
        __init_funds: Initial funds for the strategy.
        __spread_pct: Closing and opening spread.
        __slippage_pct: Closing and opening slippage.
        __commission: Commission per trade.
        __orders: Active orders.
        __positions: Active positions.
        __pos_record: Position history.
        __orders_order: Order type priority.
        __ngap: If True, gaps are not calculated when adjusting the price.
        __limit_ig: If left at True, the 'takeLimit' and 
            'stopLimit' orders are sent to the next candle.
        __orders_nclose: If set True, the orders are not ordered so that 
            the closest ones are executed first.
        __idc_data: Saved data from the indicators.
        __buffer: Buffer to concatenate DataFrame.
        __to_delate: Lists of indexes to remove from lists.

    Methods:
        tz_london: Return True if its London time zone.
        tz_tokyo: Return True if its Tokyo time zone.
        tz_sydney: Return True if its Sydney time zone.
        tz_new_york: Return True if its New York time zone.
        unique_id: Generates a random id quickly.
        act_taker: Open a position.
        act_limit: Open a limit order.
        act_close: Close an position.
        ord_put: Place a new order.
        ord_rmv: Remove the order you want.
        ord_mod: Modify the order you want.
        get_spread: Get __spread_pct.
        get_balance: Get actual balance.
        get_slippage: Get __slippage_pct.
        get_balance_rec: Get balance record.
        get_commission: Returns the commission per trade.
        get_init_funds: Returns the initial funds for the strategy.
        prev_orders: Return '__orders' values.
        prev_positions: Return '__positions' values.
        prev_positions_rec: Return '__pos_record' values.
        idc_fibonacci: Calculates Fibonacci retracement levels.
        idc_ema: Calculates the Exponential Moving Average (EMA) indicator.
        idc_sma: Calculates the Simple Moving Average (SMA) indicator.
        idc_wma: Calculates the Weighted Moving Average (WMA) indicator.
        idc_smma: Calculates the Smoothed Moving Average (SMMA) indicator.
        idc_sema: Calculates the Smoothed Exponential Moving Average (SEMA) indicator.
        idc_bb: Calculates the Bollinger Bands indicator (BB).
        idc_rsi: Calculates the Relative Strength Index (RSI).
        idc_stochastic: Calculates the Stochastic Oscillator indicator.
        idc_adx: Calculates the Average Directional Index (ADX).
        idc_macd: Calculates the Moving Average Convergence Divergence (MACD).
        idc_sqzmom: Calculates the Squeeze Momentum indicator (SQZMOM).
        idc_mom: Calculates the Momentum indicator (MOM).
        idc_ichimoku: Calculates the Ichimoku indicator.
        idc_atr: Calculates the Average True Range (ATR).

    Private Methods:
        __del: Adds items to '__to_delete'.
        __deli: Removes indexes from '__to_delate'.
        __buff: Extracts values from '__buffer'.
        __cn_insert: Insert values into a ChunkWrapper.
        __act_reduce: Reduce a position or close it.
        __put_pos: Put a position in the simulation.
        __put_ord: Place a new order.
        __view_in_orders: Returns the unionId that have ordern and position in the order list.
        __union_pos: Get all positions with the same union id.
        __word_reduce: Take an order and reduce the corresponding position.
        __order_execute: Executes an order based on the order type.
        __get_union: Find rows with the same 'unionId'.
        __price_check: Checks if 'price' is a correct value for the position.
        __uidc: Send data argument to the indicator and wraps it with '__data_store'.
        __func_idg: Generates an id for a function call.
        __store_decorator: Give '_store' attribute to a function.
        __data_store: Save the function return and, if already saved, return it from storage.
        __uidc_cut: Slices data for the user based on the current index.
        __data_cut: Slices data for the user based on the current index.
        __data_updater: Updates all data with the provided DataFrame.
        __idc_fibonacci: Calculates Fibonacci retracement levels.
        __idc_ema: Calculates the Exponential Moving Average (EMA) indicator.
        __idc_sma: Calculates the Simple Moving Average (SMA) indicator.
        __idc_wma: Calculates the Weighted Moving Average (WMA) indicator.
        __idc_smma: Calculates the Smoothed Moving Average (SMMA) indicator.
        __idc_sema: Calculates the Smoothed Exponential Moving Average (SEMA) indicator.
        __idc_bb: Calculates the Bollinger Bands indicator (BB).
        __idc_rsi: Calculates the Relative Strength Index (RSI).
        __idc_stochastic: Calculates the Stochastic Oscillator indicator.
        __idc_adx: Calculates the Average Directional Index (ADX).
        __idc_macd: Calculates the Moving Average Convergence Divergence (MACD).
        __idc_sqzmom: Calculates the Squeeze Momentum indicator (SQZMOM).
        __idc_rlinreg: This function calculates the rolling linear regression.
        __idc_mom: Calculates the Momentum indicator (MOM).
        __idc_ichimoku: Calculates the Ichimoku indicator.
        __idc_atr: Calculates the Average True Range (ATR).
        __idc_trange: This function calculates the true range.
        __before: This function is used to run trades and other operations.
    """

    open:np.ndarray
    close:np.ndarray
    low:np.ndarray
    high:np.ndarray
    volume:np.ndarray
    date:flx.DataWrapper
    hour:float

    icon:str|None
    width:float|None
    interval:str|None
    __day_width:float

    __data:flx.DataWrapper
    __data_adf:pd.DataFrame
    __data_awr:flx.DataWrapper

    __data_index:int
    __data_dates:flx.DataWrapper

    __buffer:dict
    __idc_data:dict

    __init_funds:float
    __commission:flx.CostsValue
    __spread_pct:flx.CostsValue
    __slippage_pct:flx.CostsValue
    __chunk_size:float
    __orders_order:dict

    __balance:float
    __balance_rec:list

    __ngap:bool
    __limit_ig:bool
    __orders_nclose:bool

    __orders:list
    __positions:list
    __pos_record:flx.ChunkWrapper
    __to_delate:dict

    def __init__(self, data:pd.DataFrame = pd.DataFrame()) -> None: 
        """
        __init__

        Builder for initializing the class.

        Args:
            data (DataFrame, optional): All data from the step and previous ones.
        """

        self.hour = 0.

        self.__data_adf = data
        self.__data_awr = flx.DataWrapper(data)

        self.__buffer = {}
        self.__idc_data = {}

        cmattr = lambda x: getattr(cm_, x)

        self.icon = cmattr('__data_icon')
        self.width = cmattr('__data_width')
        self.interval = cmattr('__data_interval')
        self.__day_width = cmattr('__data_width_day') or 1

        # Execution configuration
        self.__init_funds = cmattr('__init_funds') or 0.
        self.__commission = cmattr('__commission') or flx.CostsValue(0)
        self.__spread_pct  = cmattr('__spread_pct') or flx.CostsValue(0)
        self.__slippage_pct = cmattr('__slippage_pct') or flx.CostsValue(0)
        self.__chunk_size = cmattr('__chunk_size') or 10_000 
        self.__orders_order = {'op': 0, 'rd': 1, 'stopLimit': 2, 'stopLoss': 3, 
                               'takeLimit': 4, 'takeProfit': 5}

        if isinstance(cmattr('__orders_order'), dict):
            for k,v in cmattr('__orders_order').items():
                if not k in ('stopLimit', 'stopLoss', 'takeLimit', 'takeProfit'):
                    continue
                elif v > 99:
                    raise exception.StyClassError('Order ord value out of range.')

                self.__orders_order.update({k:v})

        self.__balance = self.__init_funds
        self.__balance_rec = []

        self.__ngap = bool(cmattr('__min_gap'))
        self.__limit_ig = bool(cmattr('__limit_ig'))
        self.__orders_nclose = bool(cmattr('__orders_nclose'))

        self.__orders = []
        self.__positions = []
        self.__pos_record = flx.ChunkWrapper([], chunk_size=self.__chunk_size)

        self.__to_delate = {}

        # Set decorators
        for name in dir(self):
            attr = getattr(self, name)

            if not callable(attr):
                continue
            elif getattr(attr, '_store', False):
                logger.debug("Adding __data_store decorator to '%s'", name)
                decorator = getattr(self, '_StrategyClass__data_store')(attr)
                setattr(self, name, decorator)
            elif getattr(attr, '_uidc', False):
                logger.debug("Adding __uidc decorator to '%s'", name)
                decorator = getattr(self, '_StrategyClass__uidc')(attr)
                setattr(self, name, decorator)

    @abstractmethod
    def next(self) -> Any: ...

    def tz_london(self) -> bool:
        """
        Time zone London.

        This return True if its London time zone.

        Return:
            bool: True if False otherwise.
        """

        return int(self.hour) in [6,7,8,9,10,11,12,13]

    def tz_tokyo(self) -> bool:
        """
        Time zone Tokyo.

        This return True if its Tokyo time zone.

        Return:
            bool: True if False otherwise.
        """

        return int(self.hour) in [0,1,2,3,4,5,22,23]

    def tz_sydney(self) -> bool:
        """
        Time zone Sydney.

        This return True if its Sydney time zone.

        Return:
            bool: True if False otherwise.
        """

        return int(self.hour) in [20,21,22,23,0,1,2,3]

    def tz_new_york(self) -> bool:
        """
        Time zone New York.

        This return True if its New York time zone.

        Return:
            bool: True if False otherwise.
        """

        return int(self.hour) in [11,12,13,14,15,16,17,18]

    def get_balance(self) -> float:
        """
        Get __balance

        Returns:
            float: The value of the hidden variable `__balance`.
        """

        return self.__balance

    def get_balance_rec(self) -> flx.DataWrapper:
        """
        Get __balance_rec
    
        Returns:
            DataWrapper: The value of the hidden variable `__balance_rec`.
        """

        return flx.DataWrapper(self.__balance_rec, alert=True)

    def get_spread(self) -> flx.CostsValue:
        """
        Get __spread_pct

        Info:
            To get the value use: 'get_maker' or 'get_taker'.
            In this case they will return the same.

        Returns:
            CostsValue: The value of the hidden variable `__spread_pct`.
        """

        return self.__spread_pct

    def get_slippage(self) -> flx.CostsValue:
        """
        Get __slippage_pct

        Info:
            To get the value use: 'get_maker' or 'get_taker'.
            In this case they will return the same.

        Returns:
            CostsValue: The value of the hidden variable `__slippage_pct`.
        """

        return self.__slippage_pct

    def get_commission(self) -> flx.CostsValue:
        """
        Get __commission

        Info:
            To get the value use: 'get_maker' or 'get_taker'.

        Returns:
            CostsValue: The value of the hidden variable `__commission`.
        """

        return self.__commission

    def get_init_funds(self) -> float:
        """
        Get __init_funds

        Returns:
            float: The value of the hidden variable `__init_funds`.
        """

        return self.__init_funds

    @staticmethod
    def __store_decorator(func:Callable) -> Callable:
        """
        Store decorator

        Decorate a function with this to give it 
            the attribute: '_store' and have it decorated with '__data_store'.

        Args:
            func (Callable): Function.

        Returns:
            Callable: Function.
        """

        setattr(func, '_store', True)
        return func

    def __data_store(self, func:Callable) -> Callable:
        """
        Data store

        Save the function return and, if already saved, return it from storage.

        Args:
            func (Callable): Function.

        Returns:
            Callable: Wrapper function.
        """

        def __wr_func(*args, **kwargs) -> flx.DataWrapper:
            """
            Wrapper function

            Save the function return and, if already saved, 
                return it from storage. Return in DataWrapper.

            Returns:
                DataWrapper: Function result.
            """

            id, arguments = StrategyClass.__func_idg(func, *args, **kwargs)
  
            if id in self.__idc_data:
                if arguments.get('cut', False):
                    return self.__data_cut(self.__idc_data[id],
                                           arguments.get('last', None))

                return self.__idc_data[id]

            logger.debug('Generating indicator')
            result = flx.DataWrapper(func(*args, **kwargs))

            self.__idc_data[id] = result
            if arguments.get('cut', False):
                return self.__data_cut(result, 
                                       arguments.get('last', None))

            return result
        return __wr_func

    def __uidc(self, func:Callable) -> Callable:
        """
        User indicator

        Send data argument to the indicator and save the result.

        Args:
            func (Callable): Function.

        Returns:
            Callable: Wrapper function.
        """

        if hasattr(func, "__func__"):
            func = func.__func__

        @wraps(func)
        def __wr_func(*args, **kwargs) -> flx.DataWrapper:
            """
            Wrapper function

            Save the function return and, if already saved, return it from storage.

            Sends '__data_all' to the 'data' argument.

            Returns:
                DataWrapper: Function result.
            """

            id = StrategyClass.__func_idg(func, *args, **kwargs)[0]

            if id in self.__idc_data.keys():
                return self.__uidc_cut(self.__idc_data[id])

            logger.debug('Generating user indicator')
            result = flx.DataWrapper(func(self.__data_adf, *args, **kwargs))
            self.__idc_data[id] = result

            return self.__uidc_cut(result)
        return __wr_func

    @staticmethod
    def __func_idg(func:Callable, *args, **kwargs) -> tuple[str, dict]:
        """
        Function id generator

        Generates an id for a function call 
            and returns all arguments with defaults.

        Args:
            func (Callable): Function.

        Returns:
            tuple[str,dict]: Generated id and arguments.
        """

        df = func.__defaults__ or () 
        code = func.__code__

        name_df = code.co_varnames[code.co_argcount-len(df):code.co_argcount]

        arguments = dict(zip(name_df, df))

        arguments.update({k: kwargs[k] for k in name_df if k in kwargs})
        arguments.update(zip(name_df, args))

        args_wo = arguments.copy()
        args_wo.pop('cut', None); args_wo.pop('last', None)

        return func.__name__ + ':' + '-'.join(map(str, args_wo.values())), arguments

    def __uidc_cut(self, data:flx.DataWrapper) -> flx.DataWrapper:
        """
        User indicator cut

        Slices data for the user based on the current index.

        Args:
            data (DataWrapper): Data to cut.

        Returns:
            DataWrapper: Data cut.
        """

        if len(data) != len(self.__data_adf):
            raise exception.UidcError('Length different from data.')

        result = flx.DataWrapper(data[:self.__data_index], getattr(data, '_columns'), alert=True)
        result._index = data._index

        return result

    def __data_cut(self, data:flx.DataWrapper, 
                   last:int | None = None) -> flx.DataWrapper:
        """
        Data cut

        Slices data for the user based on the current index.

        Args:
            data (DataWrapper): Data to cut.
            last (int | None, optional): You can get only the latest 'last' data.

        Returns:
            DataWrapper: Data cut.
        """

        limit = self.__data_index or 0

        return flx.DataWrapper(
            data[limit-last:limit] 
            if last is not None and last < limit 
            else data[:limit],
            columns=data._columns, 
            index=data._index, alert=True)

    def __data_updater(self, index:int, balance:list[float] | None = None) -> None:
        """
        Data updater

        Cut the data and update the variables.

        Args:
            index (int): Index at which the data must be cut.
        """

        logger.debug('Updating data')
        data = flx.DataWrapper(self.__data_awr.unwrap()[:index])
        dates = flx.DataWrapper(self.__data_adf.index.values[:index])

        self.open = data['open']
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.volume = data['volume']
        self.date = flx.DataWrapper(dates, alert=True)

        last_date = dates[-1] if dates[-1].size > 0 else 0
        self.hour = float(last_date % self.__day_width / self.__day_width * 24)

        self.__data = data
        self.__data_index = index
        self.__data_dates = dates

        self.__balance = balance[-1] if balance else self.__balance
        self.__balance_rec = balance if balance else self.__balance_rec

    def __view_in_orders(self) -> dict[str, float]:
        """
        View in orders

        Returns the unionId of current orders 
            which have a pending order as their position.

        Return:
            dict[str,float]: 
                The keys are the unionId and the 
                value is the 'orderPrice' of the position.
        """

        ids = [v['unionId'] for v in self.__orders if v['unionId'].split('/')[0] == 'w']
        result = {}

        for i in self.__orders:
            if i['order'] == 'op' and i['unionId'] and 'w/'+i['unionId'] in ids:
                result['w/'+i['unionId']] = i['orderPrice'] 

        return result

    def __before(self, index:int, balance:list[float] | None = None) -> None:
        """
        Before

        This function is used to run estrategy and calculate orders.

        Args:
            index (int): Current data index.
        """

        # Updates data to the strategy
        self.__data_updater(index=index, balance=balance)

        # Check operations
        if self.__orders:
            logger.debug('Checking orders')

            l_index = self.__view_in_orders()
            self.__orders.sort(
            key=lambda x: (
                self.__orders_order.get(x['order'], 99),
                (abs(x['orderPrice'] - (self.__data['open'][-1]
                if x['order'] == 'op' or not x['unionId'] in l_index.keys()
                else l_index[x['unionId']])) 
                 if not self.__orders_nclose else None)
            ))

            higher = self.__data['high'][-1]*(self.__spread_pct.get_taker()/100/2+1)
            lower = self.__data['low'][-1]*(1-self.__spread_pct.get_taker()/100/2)

            self.__to_delate.update({'__orders':set()})

            for i, row in enumerate(self.__orders):
                if i in self.__to_delate['__orders']:
                    continue

                # If the id contains 'w' it means that it waits for the position with the same id
                if isinstance(row['unionId'], str):
                    row_split = row['unionId'].split('/')

                    if (row_split[0] == 'w' and self.__positions
                        and row_split[-1] in [v.get('unionId') for v in self.__positions]):

                        row['unionId'] = self.__orders[i]['unionId'] = row_split[-1]
                    elif row_split[0] == 'w':
                        continue

                # Maker
                limit = (row['limit'] and higher >= row['orderPrice']
                    and lower <= row['orderPrice'])

                # Taker
                pos_cn = (not row['limit'] and higher >= row['orderPrice'] 
                            if row['typeSide'] else
                            not row['limit'] and lower <= row['orderPrice'])

                # Execute order and delete
                if limit or pos_cn:
                    self.__order_execute(row)

                    self.__del('__orders', [i])

            self.__deli('__orders')

        if (psff:=self.__buff('__pos_record')): 
            self.__pos_record = self.__cn_insert(self.__pos_record, psff)

        logger.debug("Executing 'next'")
        # Execute strategy
        self.next()

        # Concat buffer
        if (obff:=self.__buff('__orders')): self.__orders.extend(obff)
        if (psff:=self.__buff('__pos_record')): 
            self.__pos_record = self.__cn_insert(self.__pos_record, psff)

    def __cn_insert(self, vlist:flx.ChunkWrapper, 
                    values:list) -> flx.ChunkWrapper:
        """
        Chunk insert

        Insert 'values' into the 'vlist' of chunks.
        Generate a new chunk if necessary.

        Args:
            vlist (ChunkWrapper): Values with spaces.
            values (list): Values to add.

        Returns:
            ChunkWrapper: Array with the values.
        """

        types_dict = {str:'U16',
                      int:float}
        dtype = [(key, (types_dict[tp] if (tp:=type(v)) in list(types_dict.keys()) else tp)) 
                 for key, v in values[0].items()]
        vl_arr = np.array([tuple(d.values()) for d in values], dtype=dtype)

        return vlist.append(vl_arr, dtype)

    def __buff(self, name:str) -> list | None:
        """
        Buffering

        Extracts values from '__buffer'.

        Args:
            name (str): Saved value name.

        Returns:
            list|None: Returns the list of values or None if no values exist.
        """

        result = None
        if name in self.__buffer and len(self.__buffer[name]) > 0:
            result = self.__buffer[name]
            self.__buffer[name] = self.__buffer[name][:0]

        return result

    def __deli(self, name:str) -> None:
        """
        Delete index

        Removes indexes from '__to_delate'.
        Removes them from the '_StrategyClass'+name attribute.

        Args:
            name (str): Name saved in '__to_delate' and attribute name.
        """

        if (name in self.__to_delate 
            and len(self.__to_delate[name]) > 0):
            for i in sorted(self.__to_delate[name], 
                            reverse=True):

                del getattr(self, '_StrategyClass'+name)[i]

    def __del(self, name:str, index:list) -> None:
        """
        Delete

        Adds items to '__to_delete'.

        Args:
            name (str): '__to_delete' key to save, if it doesn't exist create it.
            index (list): list of indexes to save.
        """

        if name not in self.__to_delate:
            self.__to_delate.update({name:set(index)}); return

        self.__to_delate[name].update(index)

    def unique_id(self = None) -> str:
        """
        Unique id

        Generates a random id quickly.

        Args:
            self (optional): 
                You can run the function from the instance.

        Returns:
            str: An unique id.
        """

        return str(uuid4().int)

    def act_taker(self, buy:bool = True, 
                  amount:float = np.nan) -> str:
        """
        Action taker

        Open a position.

        Args:
            buy (bool, optional): Position type, True = buy.
            amount (float, optional): Position amount, np.nan equals 0.

        Returns:
            str: unionId integer.
        """

        logger.debug("Executes 'act_taker'")
        buy = bool(buy)
        ui = self.unique_id()

        self.__put_pos(
            price=self.__data["close"][-1],
            date=float(self.__data_dates[-1]),
            amount=amount,
            type_side=buy,
            union_id=ui
        )

        return ui

    def act_limit(self, price:float, buy:bool = True, 
                  amount:float = np.nan) -> str:
        """
        Action limit

        Open a limit order.

        Args:
            price (float): Price where to open it, 
                if it is within the 'spread' it will be 'taker'.
            buy (bool, optional): Position type, True = buy.
            amount (float, optional): Position amount, np.nan equals 0.

        Returns:
            str: unionId integer.
        """

        logger.debug("Executes 'act_limit'")
        buy = bool(buy)

        return self.__put_ord(
            'op', 
            price=price, 
            amount=amount, 
            buy=buy, 
            wait=False, 
            union_id=self.unique_id(),
            limit=True,
        )

    def act_close(self, index:int) -> None:
        """
        Position close

        Close an position.
        Will be applied as a 'taker'.

        Args:
            index (int): Real index of the position.
        """

        logger.debug("Executes 'act_close'")
        self.__act_reduce(index, self.__data['close'][-1], mode='taker')

    def __act_reduce(self, index:int, price:float, 
                     amount:float | None = None, mode:str = 'taker') -> None:
        """
        Action reduce

        Reduce a position or close it.

        Args:
            index (int): Real index of the position.
            price (float): Price where to reduce.
            amount (float | None, optional): Amount to reduce. None to close it.
            mode (str, optional): Order type ('taker', 'maker').
        """

        mode = mode.lower()
        if mode.lower() not in ('maker', 'taker'):
            return

        amount = amount or np.nan

        position = self.__positions[index].copy()

        pos_amount = position.get('amount')
        if np.isnan(amount):
            amount = pos_amount

        position_close = price
        position_close_spread = price
        commission = 0.

        if mode == 'maker':

            commission = self.__commission.get_maker()
            position_close_spread = price
        elif mode == 'taker':

            commission = self.__commission.get_taker()
            spread = price*(self.__spread_pct.get_taker()/100/2)
            slippage = price*(self.__slippage_pct.get_taker()/100)

            position_close_spread = (position_close-spread-slippage 
                                    if position.get('typeSide')
                                    else position_close+spread+slippage)

        # Fill data.
        position['positionClose'] = position_close_spread
        position['positionDate'] = self.__data_dates[-1]
        open = position.get('positionOpen')

        if position.get('typeSide'):
            profit_per_val = (position_close_spread-open)/open*100 
        else:
            profit_per_val = (open-position_close_spread)/open*100
        position['profitPer'] = profit_per_val

        if pos_amount > amount:
            self.__positions[index]['amount'] -= amount

            position['amount'] = amount
            pos_amount = amount

        else:
            # Close and unionId
            del self.__positions[index]

            if self.__orders:
                self.__del('__orders', [
                    i for i,d in enumerate(self.__orders) 
                    if (not position.get('unionId') is None 
                        and position.get('unionId') in (d.get('unionId').split('/')[-1], 
                                                        d.get('closeId')))
                ])

        if not np.isnan(pos_amount):
            profit_per = position.get('profitPer')

            gross_profit = pos_amount * profit_per / 100
            entry_fee = pos_amount * (position.get('commission') / 100)
            exit_fee = ((gross_profit + pos_amount) 
                        * (commission / 100))

            position['profit'] = gross_profit - entry_fee - exit_fee
            self.__balance += gross_profit - entry_fee - exit_fee
        else:
            position['profit'] = np.nan

        position['commission'] += commission

        if not '__pos_record' in self.__buffer:
            self.__buffer['__pos_record'] = []
        self.__buffer['__pos_record'].append(position)

    def __put_pos(self, price:float, date:float, 
                  amount:float, type_side:bool, union_id:str) -> None:
        """
        Put position

        Put a position in the simulation.

        Args:
            price (float): Position price.
            date (float): Date of the order.
            amount (float): Position amount.
            type_side (bool): Position type.
            union_id (str): unionId that will have.
        """

        logger.debug('Placing position')
        position_price = price
        where_date = np.where(self.__data_dates.unwrap() == date)[0][0]

        if (
            position_price > self.__data['close'][where_date]*(
                self.__spread_pct.get_taker()/100/2+1)
            or position_price < self.__data['close'][where_date]*(
                1-self.__spread_pct.get_taker()/100/2)
            ): # Maker

            position_open = position_price
            commission = self.__commission.get_maker()
        else: # Taker
            position_price = self.__data['close'][where_date]

            commission = self.__commission.get_taker()
            slippage = position_price*(self.__slippage_pct.get_taker()/100)
            spread = position_price*(self.__spread_pct.get_taker()/100/2)

            position_open = (position_price+spread+slippage
                            if type_side else position_price-spread-slippage)

        position = {
            'date':self.__data_dates[-1],
            'positionOpen':position_open,
            'commission':commission,
            'amount':amount,
            'typeSide':type_side,
            'unionId':union_id,
        }

        self.__positions.append(position) 

    def __union_pos(self, union_id:str) -> list | None:
        """
        Union positions

        Get all positions with the same union id.

        The positions are returned with a new column: 
            'rIndex' which indicates the actual index in self.__positions.

        Args:
            union_id (str): Union id to filter with.

        Returns:
            list|None: 
                Positions if none are found, None is returned.
        """

        if not self.__positions:
                return

        union_positions = [
            {**v, 'rIndex': i}
            for i, v in enumerate(self.__positions)
            if (not v.get('unionId') is None 
                and v.get('unionId') == union_id)
        ]

        return union_positions or None

    def __word_reduce(self, order:dict, mode:str = 'taker') -> bool | None:
        """
        With order reduce

        Take an order and reduce the corresponding position.

        Args:
            order (dict): Dict with the order.
            mode (str): Order type ('taker', 'maker').

        Returns:
            bool|None: 
                Returns True if the position was reduced, otherwise returns None.
        """

        logger.debug("Reducing position")
        if not (u_pos:=self.__union_pos(order['unionId'])):
            logger.debug(
                "The reduction was cancelled: No position was found with the same 'unionId'")
            return

        self.__act_reduce(
            u_pos[0]['rIndex'], order['orderPrice'], 
            amount=order['amount'], mode=mode)
        return True

    def __order_execute(self, order:dict) -> None:
        """
        Order execute

        Order execution here each order is executed 
            according to its type and a mask is applied for 'closeId'.

        Args:
            order (dict): Dict with the order.
        """

        # Hello world
        match order['order']:
            case 'op':
                self.__put_pos(
                    price=order['orderPrice'],
                    date=order['date'],
                    amount=order['amount'],
                    type_side=order['typeSideOrd'],
                    union_id=order['unionId']
                )
    
            case 'rd':
                self.__word_reduce(order, mode='maker')

            case 'takeProfit' | 'stopLoss':
                # Gap case
                if not self.__ngap:
                    min_gap = min(self.__data['open'][-1], 
                                self.__data['close'][-2])
                    max_gap = max(self.__data['open'][-1], 
                                self.__data['close'][-2])

                    if order['orderPrice'] > min_gap and order['orderPrice'] < max_gap:
                        order['orderPrice'] = self.__data['open'][-1]

                self.__word_reduce(order, mode='taker')
 
            case 'takeLimit' | 'stopLimit':
                higher = order['orderPrice']*(self.__spread_pct.get_taker()/100/2+1)
                lower = order['orderPrice']*(1-self.__spread_pct.get_taker()/100/2)

                if (higher <= order['limitPrice'] 
                    and lower >= order['limitPrice']
                    and not self.__limit_ig):

                    self.__word_reduce(order, mode='taker')
                elif (order['limitPrice'] <= self.__data['high'][-1]
                    and order['limitPrice'] >= self.__data['low'][-1]
                    and not self.__limit_ig):

                    self.__word_reduce(order, mode='maker')
                else:
                    self.__put_ord(
                        'rd', 
                        price=order['limitPrice'], 
                        amount=order['amount'],
                        buy=order['typeSide'],
                        union_id=order['unionId'],
                        close_id=order['closeId'],
                        limit=True,
                    )

        # Del closeId orders
        self.__del('__orders', [
            i for i, d in enumerate(self.__orders)
            if (not d.get('closeId') is None 
                and d.get('closeId') in (order['id'], order['closeId']))
        ])

    def __get_union(self, data:pd.DataFrame, union_id:str) -> pd.DataFrame|None:
        """
        Get union

        Get all rows from 'data' that have 'union_id' 
            and are 'op' if it has an 'order' column.

        Args:
            data (DataFrame): Data to put the mask. 
            union_id (str): unionId to filter.

        Returns:
            DataFrame|None: Resulting dataframe with 
                the rows that meet the mask.
        """

        if data.empty:
            return None

        mask = np.asarray(data['unionId'].values == union_id)
        if 'order' in data.columns:
            mask &= data['order'].values == 'op'
        data_leak = data.loc[mask]

        return data_leak if not data_leak.empty else None

    def __price_check(self, price:float, union_id:str|None, 
                      type_side:bool, price_cn:float|None = None
                      ) -> tuple[bool|None, bool]:
        """
        Price check

        Checks if 'price' is a correct value for the position 
            type associated with 'union_id' based on 'type_side'.

        Args:
            price (float): Price to be verified. 
            union_id (str|None): Associated unionId.
            type_side (bool): Associated typeSide.
            price_cn (float | None, optional): This function, 
                if you leave this variable as None, will compare 
                the price with the price of the linked operation, 
                if you want to compare it with another 
                you can use this variable.

        Returns:
            tuple[bool|None,bool]: Condition of 'typeSide' equals and 
                True if correct False if not.
        """

        if pd.isna(union_id) or union_id is None:
            return None, True

        union_id = union_id.split('/')[-1]
        func = np.max if type_side else np.min

        def get_union_price(data, col_name:str) -> tuple[None, float|bool]:
            """
            Get union price

            Finds positions with the same 'unionId' and 
                returns the result of passing 'col_name' to 'func'.
            'func' and 'unionId' external variables.

            Args:
                data: Data where to search. 
                col_name (str): Column to pass through 'func'.

            Returns:
                tuple[bool,float|bool]: Condition of 'typeSide' equals and 
                    result of 'func' on 'col_name'.
            """

            nonlocal func
            if len(data) == 0:
                return None, False

            if type(data) is list:
                leak = {k: [d[k] for d in data if d.get('unionId') == union_id 
                            and d.get('order', 'op') == 'op']
                        for k in data[0].keys()}

                if any(map(bool, leak.values())):
                    leak_type_side = leak['typeSide'][0]
                else:
                    return None, False
            else:
                leak = self.__get_union(data=data, union_id=union_id)

                if not leak is None:
                    leak_type_side = leak['typeSide'].iloc[0]
                else:
                    return None, False

            comparision = leak_type_side == type_side

            func = np.max if comparision else np.min
            return comparision, func(leak[col_name])

        comp_ord, union_ord = get_union_price(self.__orders, 'orderPrice')
        comp_pos, union_pos = get_union_price(self.__positions, 'positionOpen')

        union_ord_bff = False
        comp_ord_bff = None

        if '__orders' in self.__buffer:
            comp_ord_bff, union_ord_bff = get_union_price(
                self.__buffer['__orders'], 'orderPrice')

        comp = next(filter(lambda x: x is not None, (comp_pos, comp_ord_bff, comp_ord)), None)

        if price_cn is None:
            union_pos_f:float = union_pos or union_ord or union_ord_bff or price
            union_ord_f:float = union_ord or union_ord_bff or union_pos_f
            union_ord_bff_f:float = union_ord_bff or union_ord_f or union_pos_f

            price_cn = func([union_pos_f, union_ord_f, union_ord_bff_f])
        return comp, price >= price_cn if func == np.max else price <= price_cn

    def __put_ord(self, order_type:str, price: float, amount:float | None = None, 
                  limit_price:float | None = None, buy:bool = True, 
                  wait:bool = True, union_id:str | int | None = None, 
                  close_id:str | int | None = None, limit:bool = False) -> str:
        """
        Put order

        Place a new order.

        Args:
            order_type (str): Order type 
                ('op', 'rd', 'stopLoss', 'takeProfit', 'takeLimit', 'stopLimit').
            price (float): Order price.
            amount (float | None, optional): Order amount, None takes the total.
            limit_price (float | None, optional): 
                Limit order price, using 'takeLimit' and 'stopLimit'.
            buy (bool, optional): Order type (only works on 'op' and 'rd' type).
            wait (bool, optional): Indicates if the order waits for an open 
                position with the same unionId.
            union_id (str | int | None, optional): unionId in charge of connecting to the 
                desired position, if left as None the last one will be used.
            close_id (str | int | None, optional): closeId responsible for closing the 
                order if an order with the same closeId or id is closed.
            limit (bool, optional): Set to True if the order is Maker.

        Returns:
            str: unionId integer.
        """

        logger.debug('Creating an order')
        amount = amount or np.nan

        close_id = str(int(close_id)) if close_id else None
        union_id = f'{"w/" if wait else ""}{int(union_id)}' if union_id else None

        match order_type:
            case 'op' | 'rd':
                buy = buy
            case 'stopLoss' | 'stopLimit':
                buy = False
            case 'takeProfit' | 'takeLimit':
                buy = True
            case _:
                raise exception.OrderError('Invalid order type.')

        pos_type_side, check = self.__price_check(
            price=price, union_id=union_id, type_side=buy)

        if (order_type != 'op' and not check):
            raise exception.OrderError("Invalid 'price' for order type.")

        order = {
            'order':order_type,
            'date':self.__data_dates[-1],
            'orderPrice':price,
            'limitPrice':limit_price if limit_price else price,
            'amount':amount,
            'typeSide':pos_type_side if not pos_type_side is None else buy,
            'typeSideOrd':buy,
            'id':self.unique_id(),
            'unionId':union_id,
            'closeId':close_id,
            'limit':limit,
        }

        if not '__orders' in self.__buffer:
            self.__buffer['__orders'] = []
        self.__buffer['__orders'].append(order)

        return union_id or ''

    def ord_put(self, order_type:str, price:float, amount:float | None = None, 
                limit_price:float | None = None, union_id:str | int | None = None, 
                close_id:str | int | None = None) -> str:
        """
        Order put

        Place a new order.

        Args:
            order_type (str): Order type 
                ('stopLoss', 'takeProfit', 'takeLimit', 'stopLimit').
            price (float): Order price.
            amount (float | None, optional): Order amount, None takes the total.
            limit_price (float | None, optional): 
                Limit order price, using 'takeLimit' and 'stopLimit'.
            union_id (str | int | None, optional): unionId in charge of connecting to the 
                desired position, if left as None the last one will be used.
            close_id (str | int | None, optional): closeId responsible for closing the 
                order if an order with the same closeId or id is closed.

        Returns:
            str: unionId integer.
        """

        logger.debug("Executes 'ord_put'")
        if not order_type in (
            'stopLoss', 'takeProfit', 'takeLimit', 'stopLimit'):
            raise ValueError('Bad op type')

        # Last unionId
        last_data = None
        if self.__positions:
            last_data = self.__positions[-1]['unionId']
        elif '__orders' in self.__buffer and self.__buffer['__orders']:
            last_data = self.__buffer['__orders'][-1]['unionId']
        elif self.__orders:
            last_data = self.__orders[-1]['unionId']
        elif '__pos_record' in self.__buffer and self.__buffer['__pos_record']:
            last_data = self.__buffer['__pos_record'][-1]['unionId']
        elif len(self.__pos_record) > 0:
            last_data = self.__pos_record[-1]['unionId']

        union_id = union_id or (last_data.split('/')[-1] if last_data else 0)

        return self.__put_ord(
            order_type, 
            price=price, 
            amount=amount,
            limit_price=limit_price,
            union_id=union_id,
            close_id=close_id,
            limit=True if order_type in ('stopLimit', 'takeLimit') else False,
        )

    def ord_rmv(self, index:int) -> None:
        """
        Order remove

        Remove an active order.
        An order is considered active if it is in '__ orders'.

        Args:
            index (int): Real index of the order to be deleted.
        """

        logger.debug("Executes 'ord_rmv'")
        if not self.__orders:
            raise exception.OrderError('There are no active orders.')
        elif not isinstance(index, int):
            logger.debug(
                "The order deletion was cancelled: 'index' is not a valid type.")
            return

        order = self.__orders[index]

        self.__del('__orders', [index]+[
            i for i,d in enumerate(self.__orders) 
            if order['id'] in (d.get('unionId').split('/')[-1], 
                                d.get('closeId'))
        ])
        self.__deli('__orders')

    def ord_mod(self, index:int, price:float | None = None,
                amount:float | None = None) -> None:
        """
        Order modify

        Modify the order you want.

        Args:
            index (int): Real index of the order to be modified.
            price (float | None, optional): New order price leave as None 
                if you do not want to change it.
            amount (float | None, optional): New order amount leave as None 
                if you do not want to change it. 
                Leave it as np.nan if you want to remove the amount.
        """

        logger.debug("Executes 'ord_mod'")
        if price is None and amount is None:
            raise exception.OrderError('Nothing to modify.')

        order = self.__orders[index]

        if not price is None:
            _, union = self.__price_check(
                price=price, 
                union_id=order['unionId'], 
                type_side=order['typeSide'],
                price_cn=self.__data['close'][-1]
            )

            if union:
                self.__orders[index]['orderPrice'] = price

        if not amount is None:
            self.__orders[index]['amount'] = amount

    def prev_positions_rec(self, label:str | None = None,
                           last:int | None = None) -> flx.DataWrapper:
        """
        Prev of closed trades.

        This function returns the values of `pos_record`.

        Args:
            label (str | None, optional): Data column to return. If None, all columns 
                are returned. If 'index' or 'rIndex', only return the real index.
            last (int | None, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `pos_record` columns.

            - date: Creation date.
            - positionOpen: Opening price.
            - commission: Position commissions.
            - amount: Position amount
            - typeSide: Position type.
            - unionId: Id linked to orders.
            - positionClose: Closing price.
            - positionDate: Closing date.
            - profitPer: Profit in percentage with out commissions.
            - profit: Profit on 'amount' with commissions.

        Returns:
            DataWrapper: DataWrapper containing the data from closed trades.
        """

        __pos_rec = self.__pos_record.values()
        if len(__pos_rec) == 0: 
            return flx.DataWrapper(alert=True)
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 

            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        if label and label in ('index', 'rIndex'):
            __pos_rec:np.ndarray = np.arange(len(__pos_rec))

        data = __pos_rec[
            len(__pos_rec) - last if last is not None and last < len(__pos_rec) else 0:]

        if (label not in (None, 'index', 'rIndex')
            and __pos_rec.dtype.names and data.size):

            data = data[label]

        return flx.DataWrapper(data, alert=True)

    def prev_positions(self, label:str | None = None, 
                       uid:int | None = None, last:int | None = None
                       ) -> flx.DataWrapper:
        """
        Prev of trades active.

        This function returns the values of `positions`.

        Args:
            label (str | None, optional): Data column to return. If None, all columns 
                are returned. If 'index', only return the 'rIndex'.
            uid (int | None, optional): Filter by unionId.
            last (int | None, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `positions` columns.

            - rIndex: Column added in 'prev_positions' 
                marks the real index of each row.
            - date: Creation date.
            - positionOpen: Opening price.
            - commission: Position commissions.
            - amount: Position amount
            - typeSide: Position type True for buy.
            - unionId: Id linked to orders.

        Returns:
            DataWrapper: DataWrapper containing the data from active trades.
        """

        __pos = self.__positions
        if not __pos or len(__pos) == 0: 
            return flx.DataWrapper(alert=True)
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 

            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data_lf = lambda x: x[
            len(x) - last if last is not None and last < len(x) else 0:]
        if label and label.lower() == 'index':
            return flx.DataWrapper(data_lf(np.arange(0, len(__pos))), alert=True)

        keys = list(__pos[0].keys())
        data_columns = keys.copy()
        data_columns.insert(0, 'rIndex')

        data:list = []
        dtype = [
            (key, float if isinstance(v, int) else 'U50' if isinstance(v, str) or v is None else type(v)) 
            for key, v in {**{'rIndex':0}, **__pos[0]}.items()]

        for i,v in enumerate(__pos):
            if uid != None and uid != v['unionId']:
                continue

            data.append((i, *[v[k] for k in keys]))

        arr_data:np.ndarray = data_lf(np.array(data, dtype=dtype))
        if label != None and data_columns and arr_data.size:
            arr_data = arr_data[label]

        return flx.DataWrapper(arr_data, columns=data_columns, alert=True)

    def prev_orders(self, label:str | None = None, or_type:str | None = None,
                    ids:dict | None = None, last:int | None = None
                    ) -> flx.DataWrapper:
        """
        Prev of active orders.

        This function returns the values of `orders`.

        Args:
            label (str | None, optional): Data column to return. If None, all columns 
                are returned. If 'index', only return the 'rIndex'.
            or_type (str | None, optional): If you want to filter only one type 
                of operation you can do so with this argument.
                Current types of orders: 'op', 'takeProfit', 'stopLoss', 'takeLimit', 'stopLimit'.
            ids (dict | None, optional): Filter by id by making a dictionary with id_name:id.
            last (int | None, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `orders` columns.

            - rIndex: Column added in 'prev_orders' 
                marks the real index of each row.
            - order: Order type ('op', 'takeProfit', 
                'stopLoss', 'takeLimit', 'stopLimit').
            - date: Creation date.
            - orderPrice: Execution price.
            - amount: Amount to be executed.
            - typeSide: Position type.
            - typeSideOrd: Order type True for positive type.
            - id: Unique order ID.
            - unionId: Linked id to position. If there is no id, 'none' is returned.
            - closeId: Close link. If there is no id, 'none' is returned.
            - limit: Indicates execution type to be performed.

        Returns:
            DataWrapper: DataWrapper containing the data from active trades.
        """

        __ord = self.__orders
        if not __ord or len(__ord) == 0: 
            return flx.DataWrapper(alert=True)
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 

            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))
        elif ((not isinstance(ids, dict) and not ids is None) 
            or (isinstance(ids, dict) 
            and not set(ids.keys()).issubset(('id','unionId','closeId')))):

            raise ValueError(utils.text_fix(
                "'ids' with bad format or incorrect.", newline_exclude=True))

        data_lf = lambda x: x[
            len(x) - last if last is not None and last < len(x) else 0:]
        if label and label.lower() == 'index':
            return flx.DataWrapper(data_lf(np.arange(0, len(__ord))), alert=True)

        keys = list(__ord[0].keys())
        data_columns = keys.copy()
        data_columns.insert(0, 'rIndex')

        data:list = []
        dtype = [
            (key, float if isinstance(v, int) else 'U50' if isinstance(v, str) or v is None else type(v)) 
            for key, v in {**{'rIndex':0}, **__ord[0]}.items()]

        for i,v in enumerate(__ord):
            if or_type != None and v['order'] != or_type:
                continue
            if ids != None and all(v[i_label] == ids[i_label] 
                                   for i_label in ids.keys()):
                continue

            data.append((i, *[v[k] for k in keys]))

        arr_data:np.ndarray = data_lf(np.array(data, dtype=dtype))
        if label != None and data_columns and arr_data.size:
            arr_data = arr_data[label]

        return flx.DataWrapper(arr_data, columns=data_columns, alert=True)

    def idc_fibonacci(self, lv0:float = 10, lv1:float = 1) -> flx.DataWrapper:
        """
        Calculate Fibonacci retracement levels.

        This function calculates the Fibonacci retracement levels.

        Args:
            lv0 (float, optional): Level 0 position.
            lv1 (float, optional): Level 1 position.

        Returns:
            DataWrapper: A DataWrapper with Fibonacci levels and their 
                corresponding values.

        Columns:
            - 'Level'
            - 'Value'
        """

        # Fibonacci calc.
        return self.__idc_fibonacci(lv0=lv0, lv1=lv1)

    @__store_decorator
    def __idc_fibonacci(self, lv0:int = 10, lv1:int = 1) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.

        This function calculates the Fibonacci retracement levels.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Returns:
            DataWrapper: A DataWrapper with Fibonacci levels and their corresponding
                values.

        Columns:
            - 'Level'
            - 'Value'
        """

        fibo_levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 
                                0.786, 1, 1.618, 2.618, 3.618, 4.236])

        return pd.DataFrame({'Level':fibo_levels,
                             'Value':lv0 - (lv0 - lv1) * fibo_levels})

    def idc_ema(self, length:int, source:str = 'close', 
                last:int | None = None) -> flx.DataWrapper:
        """
        Exponential moving average (EMA).

        This function calculates the EMA.

        Args:
            length (int): The length of the EMA.
            source (str, optional): The data source for the EMA calculation. Allowed 
                parameters are 'close', 'open', 'high', 'low', and 'volume'.
            last (int | None, optional): Number of data points to return from the 
                present backwards. If None, returns data for all time.

        Returns:
            DataWrapper: DataWrapper containing the EMA values for each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low','volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low','volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Ema calc.
        return self.__idc_ema(length=length, source=source, 
                              last=last, cut=True)

    @__store_decorator
    def __idc_ema(self, data:pd.Series | None = None, length:int = 10, 
                  source:str = 'close', last:int | None = None, 
                  cut:bool = False) -> pd.Series:
        """
        Exponential Moving Average (EMA).

        This function calculates the EMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the EMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper containing the EMA values for each step.
        """

        v_data = self.__data_adf[source] if data is None else data
        ema = v_data.ewm(span=length, adjust=False).mean()

        return ema

    def idc_sma(self, length:int, source:str = 'close', 
                last:int | None = None) -> flx.DataWrapper:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Args:
            length (int): Length of the SMA.
            source (str, optional): Data source for SMA calculation. Allowed values are 
                          ('close', 'open', 'high', 'low', 'volume').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.
        
        Returns:
            DataWrapper: DataWrapper containing the SMA values for each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low','volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low','volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Sma calc.
        return self.__idc_sma(length=length, source=source, 
                                last=last, cut=True)

    @__store_decorator
    def __idc_sma(self, data:pd.Series | None = None, length:int = 10, 
                  source:str = 'close', last:int | None = None, 
                  cut:bool = False) -> pd.Series:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the SMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper containing the SMA values for each step.
        """

        v_data = self.__data_adf[source] if data is None else data
        sma = v_data.rolling(window=length).mean()

        return sma

    def idc_wma(self, length:int, source:str = 'close', 
                invt_weight:bool = False, last:int | None = None) -> flx.DataWrapper:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Args:
            length (int): Length of the WMA.
            source (str, optional): Data source for WMA calculation. Allowed values are 
                          ('close', 'open', 'high', 'low', 'volume').
            invt_weight (bool, optional): If True, the distribution of weights is reversed.
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing the WMA values for each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low','volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low','volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Wma calc.
        return self.__idc_wma(length=length, source=source, 
                              invt_weight=invt_weight, last=last, cut=True)

    @__store_decorator
    def __idc_wma(self, data:pd.Series | None = None, 
                  length:int = 10, source:str = 'close', 
                  invt_weight:bool = False, last:int | None = None, 
                  cut:bool = False) -> pd.Series:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the WMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper containing the WMA values for each step.
        """

        v_data = self.__data_adf[source] if data is None else data

        weight = (np.arange(1, length+1)[::-1] 
                  if invt_weight else np.arange(1, length+1))
        wma = v_data.rolling(window=length).apply(
            lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return wma
    
    def idc_smma(self, length:int, source:str = 'close', 
                 last:int | None = None) -> flx.DataWrapper:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Args:
            length (int): Length of the SMMA.
            source (str, optional): Data source for SMMA calculation. Allowed values are 
                          ('close', 'open', 'high', 'low', 'volume').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing the SMMA values for each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low','volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low','volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Smma calc.
        return self.__idc_smma(length=length, source=source, 
                                last=last, cut=True)

    @__store_decorator
    def __idc_smma(self, data:pd.Series|None = None, length:int = 10, 
                   source:str = 'close', last:int|None = None, 
                   cut:bool = False) -> pd.Series:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the SMMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper containing the SMMA values for each step.
        """

        v_data = self.__data_adf[source] if data is None else data

        smma = v_data.ewm(alpha=1/length, adjust=False).mean()
        smma.shift(1)

        return smma

    def idc_sema(self, length:int = 9, method:str = 'sma', 
                  smooth:int = 5, only:bool = False, 
                  source:str = 'close', last:int | None = None) -> flx.DataWrapper:
        """
        Smoothed Exponential Moving Average (SEMA).

        This function calculates the SEMA.

        Args:
            length (int, optional): Length of the EMA.
            method (str, optional): Smoothing method. Choices include various smoothing 
                          methods.
            smooth (int, optional): Length of the smoothing method.
            only (bool, optional): If True, returns only a Series with the values of the 
                        'method'.
            source (str, optional): Data source for EMA calculation. Allowed values are 
                          ('close', 'open', 'high', 'low', 'volume').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Columns:
            - 'ema'
            - 'smoothed'

        Returns:
            DataWrapper: DataWrapper containing the 'ema' and 'smoothed' values for 
                              each step
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not method in ('sma','ema','smma','wma'): 
            raise ValueError(utils.text_fix("""
                             'method' only one of these values: 
                             ['sma','ema','smma','wma'].
                             """, newline_exclude=True))
        elif smooth > 5000 or smooth <= 0: 
            raise ValueError(utils.text_fix("""
                             'smooth' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low','volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low','volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Sema calc.
        return self.__idc_sema(length=length, method=method, smooth=smooth, 
                                only=only, source=source, last=last, cut=True)
    
    @__store_decorator
    def __idc_sema(self, data:pd.Series | None = None, length:int = 9, 
                    method:str = 'sma', smooth:int = 5, only:bool = False, 
                    source:str = 'close', last:int | None = None, 
                    cut:bool = False) -> pd.DataFrame|np.ndarray:
        """
        Smoothed Exponential Moving Average (SEMA).

        This function calculates the SEMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the SEMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'ema'
            - 'smoothed'

        Returns:
            DataWrapper: DataWrapper containing 'ema' and 'smoothed' values for 
                          each step.
        """

        v_data = self.__data_adf[source] if data is None else data
        ema = v_data.ewm(span=length, adjust=False).mean()

        match method:
            case 'sma': smema = self.__idc_sma(data=ema, length=smooth).unwrap()
            case 'ema': smema = self.__idc_ema(data=ema, length=smooth).unwrap()
            case 'smma': smema = self.__idc_smma(data=ema, length=smooth).unwrap()
            case 'wma': smema = self.__idc_wma(data=ema, length=smooth).unwrap()
            case _: smema = self.__idc_sma(data=ema, length=smooth).unwrap()

        if only: 
            smema = np.flip(smema)
            return np.flip(smema[len(smema)-last 
                                 if last != None and last < len(smema) else 0:])
        
        smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)
        return smema

    def idc_bb(self, length:int = 20, std_dev:float = 2, ma_type:str = 'sma', 
               source:str = 'close', last:int | None = None) -> flx.DataWrapper:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Args:
            length (int, optional): Window length for calculating Bollinger Bands.
            std_dev (float, optional): Number of standard deviations for the bands.
            ma_type (str, optional): Type of moving average. For example, 'sma' for simple 
                          moving average.
            source (str, optional): Data source for calculation. Allowed values are 
                          ('close', 'open', 'high', 'low').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Columns:
            - 'upper'
            - '{ma_type}'
            - 'lower'

        Returns:
            DataWrapper: DataWrapper containing 'upper', '{ma_type}', and 'lower' 
                          values for each step.
        """

        source = source.lower()
        ma_type = ma_type.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif std_dev > 50 or std_dev < 0.001: 
            raise ValueError(utils.text_fix("""
                             'std_dev' it has to be greater than 0.001 and 
                             less than 50.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low'].
                             """, newline_exclude=True))
        elif not ma_type in ('sma','ema','wma','smma'): 
            raise ValueError(utils.text_fix("""
                             'ma_type' only these values: 
                             'sma', 'ema', 'wma', 'smma'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Bb calc.
        return self.__idc_bb(length=length, std_dev=std_dev, 
                            ma_type=ma_type, source=source, last=last, cut=True)

    @__store_decorator
    def __idc_bb(self, data:pd.Series | None = None, length:int = 20, 
                 std_dev:float = 2, ma_type:str = 'sma', source:str = 'close', 
                 last:int | None = None, cut:bool = False) -> pd.DataFrame:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the Bollinger Bands 
                calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'upper'
            - '{ma_type}'
            - 'lower'

        Returns:
            DataWrapper: DataWrapper containing 'upper', '{ma_type}', and 'lower' 
                          values for each step.
        """

        v_data = self.__data_adf[source] if data is None else data

        match ma_type:
            case 'sma': ma = self.__idc_sma(data=v_data, length=length).to_series()
            case 'ema': ma = self.__idc_ema(data=v_data, length=length).to_series()
            case 'wma': ma = self.__idc_wma(data=v_data, length=length).to_series()
            case 'smma': ma = self.__idc_smma(data=v_data, length=length).to_series()
            case _: ma = self.__idc_sma(data=v_data, length=length).to_series()

        std_ = (std_dev * v_data.rolling(window=length).std())
        bb = pd.DataFrame({'upper':ma + std_,
                           ma_type:ma,
                           'lower':ma - std_}, index=ma.index)

        return bb

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, 
                rsi_ma_type:str = 'smma', base_type:str = 'sma', 
                bb_std_dev:float = 2, source:str = 'close', 
                last:int | None = None) -> flx.DataWrapper:
        """
        Relative Strength Index (RSI).

        This function calculates the RSI.

        Args:
            length_rsi (int, optional): Window length for the RSI calculation using 
                              `rsi_ma_type`. Default is 14.
            length (int, optional): Window length for the moving average applied to RSI. 
                          Default is 14.
            rsi_ma_type (str, optional): Type of moving average used for calculating RSI. 
                              For example, 'wma' for weighted moving average.
            base_type (str, optional): Type of moving average applied to RSI. For example, 
                            'sma' for simple moving average.
            bb_std_dev (float, optional): Standard deviation for Bollinger Bands. Default is 2.
            source (str, optional): Data source for calculation. Allowed values are 
                          ('close', 'open', 'high', 'low').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Columns:
            - 'rsi'
            - '{base_type}'

        Returns:
            DataWrapper: DataWrapper containing 'rsi' and '{base_type}' values for 
                          each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif bb_std_dev > 50 or bb_std_dev < 0.001: 
            raise ValueError(utils.text_fix("""
                             'bb_std_dev' it has to be greater than 0.001 and 
                             less than 50.
                             """, newline_exclude=True))
        elif length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length_rsi' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low'].
                             """, newline_exclude=True))
        elif not rsi_ma_type in ('sma','ema','wma','smma'): 
            raise ValueError(utils.text_fix("""
                             'rsi_ma_type' only these values: 
                             'sma', 'ema', 'wma','smma'.
                             """, newline_exclude=True))
        elif not base_type in ('sma','ema','wma','bb'): 
            raise ValueError(utils.text_fix("""
                             'base_type' only these values: 
                             'sma', 'ema', 'wma', 'smma', 'bb'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Rsi calc.
        return self.__idc_rsi(length_rsi=length_rsi, length=length, 
                              rsi_ma_type=rsi_ma_type, base_type=base_type, 
                              bb_std_dev=bb_std_dev, source=source, 
                              last=last, cut=True)

    @__store_decorator
    def __idc_rsi(self, data:pd.Series | None = None, length_rsi:int = 14, 
                  length:int = 14, rsi_ma_type:str = 'smma', 
                  base_type:str = 'sma', bb_std_dev:float = 2, 
                  source:str = 'close', last:int | None = None, 
                  cut:bool = False)  -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        This function calculates the RSI.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the RSI calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'rsi'
            - '{base_type}'

        Returns:
            DataWrapper: DataWrapper containing 'rsi' and '{base_type}' values for 
                          each step.
        """

        delta = self.__data_adf[source].diff() if data is None else data.diff()

        ma = self.__idc_sma
        match rsi_ma_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        ma_gain = ma(data = delta.where(delta > 0, 0), 
                     length=length_rsi, source=source).to_series()
        ma_loss = ma(data = -delta.where(delta < 0, 0), 
                     length=length_rsi, source=source).to_series()
        rsi = 100 - (100 / (1+ma_gain/ma_loss))

        match base_type:
            case 'sma': mv = self.__idc_sma(data=rsi, length=length).to_series()
            case 'ema': mv = self.__idc_ema(data=rsi, length=length).to_series()
            case 'wma': mv = self.__idc_wma(data=rsi, length=length).to_series()
            case 'smma': mv = self.__idc_smma(data=rsi, length=length).to_series()
            case 'bb': mv = self.__idc_bb(data=rsi, length=length,
                                          std_dev=bb_std_dev).to_dataframe()
            case _: mv = self.__idc_sma(data=rsi, length=length).to_series()

        if type(mv) == pd.Series: mv.name = base_type

        rsi:pd.DataFrame = pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

        return rsi

    def idc_stochastic(self, length_k:int = 14, smooth_k:int = 1, 
                       length_d:int = 3, d_type:str = 'sma', 
                       source:str = 'close', last:int | None = None) -> flx.DataWrapper:
        """
        Stochastic Oscillator.

        This function calculates the stochastic oscillator.

        Args:
            length_k (int, optional): Window length for calculating the stochastic values.
            smooth_k (int, optional): Smoothing window length for the stochastic values.
            length_d (int, optional): Window length for the moving average applied to 
                            the stochastic values.
            d_type (str, optional): Type of moving average used for the stochastic oscillator. 
                          For example, 'sma' for simple moving average.
            source (str, optional): Data source for calculation. Allowed values are 
                          ('close', 'open', 'high', 'low').
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Columns:
            - 'stoch'
            - '{d_type}'

        Returns:
            DataWrapper: DataWrapper containing 'stoch' and '{d_type}' values for each 
                          step.
        """

        source = source.lower()
        if length_k > 5000 or length_k <= 0: 
            raise ValueError(utils.text_fix("""
                             'length_k' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif smooth_k > 5000 or smooth_k <= 0: 
            raise ValueError(utils.text_fix("""
                             'smooth_k' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif length_d > 5000 or smooth_k <= 0: 
            raise ValueError(utils.text_fix("""
                             'length_d' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low'].
                             """, newline_exclude=True))
        elif not d_type in ('sma','ema','wma','smma'): 
            raise ValueError(utils.text_fix("""
                             'd_type' only these values: 
                             'sma', 'ema', 'wma', 'smma'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        # Calc stoch.
        return self.__idc_stochastic(length_k=length_k, smooth_k=smooth_k, 
                                    length_d=length_d, d_type=d_type, 
                                    source=source, last=last, cut=True)

    @__store_decorator
    def __idc_stochastic(self, data:pd.Series | None = None, length_k:int = 14, 
                         smooth_k:int = 1, length_d:int = 3, d_type:str = 'sma', 
                         source:str = 'close', last:int | None = None, 
                         cut:bool = False) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        This function calculates the stochastic oscillator.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the stochastic calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'stoch'
            - '{d_type}'

        Returns:
            DataWrapper: DataWrapper containing 'stoch' and '{d_type}' values for each 
                          step.
        """

        v_data:pd.DataFrame|pd.Series = self.__data_adf if data is None else data

        low_data = v_data.loc[:, 'low'].rolling(window=length_k).min()
        high_data = v_data.loc[:, 'high'].rolling(window=length_k).max()

        ma = self.__idc_sma
        match d_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        stoch = (((v_data[source] - low_data) / 
                  (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
        result = pd.DataFrame({'stoch':stoch, 
                               d_type:ma(data=stoch, length=length_d).to_series()})

        return result

    def idc_adx(self, smooth:int = 14, length_di:int = 14,
                only:bool = False, last:int | None = None) -> flx.DataWrapper:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Args:
            smooth (int, optional): Smoothing length. Default is 14.
            length_di (int, optional): Window length for calculating +DI and -DI. Default is 14.
            only (bool, optional): If True, returns only a Series with the ADX values.
            last (int | None, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Columns:
            - 'adx'
            - '+di'
            - '-di'

        Returns:
            DataWrapper: DataWrapper containing 'adx', '+di', and '-di' values for 
                          each step.
        """

        if smooth > 5000 or smooth <= 0: 
            raise ValueError(utils.text_fix("""
                             'smooth' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif length_di > 5000 or length_di <= 0: 
            raise ValueError(utils.text_fix("""
                             'length_di' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc adx.
        return self.__idc_adx(smooth=smooth, length_di=length_di, 
                            only=only, last=last, cut=True)

    @__store_decorator
    def __idc_adx(self, data:pd.Series | None = None, smooth:int = 14, 
                  length_di:int = 14, only:bool = False, 
                  last:int | None = None, cut:bool = False) -> pd.DataFrame:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (Series | None, optional): Series of data to perform the ADX calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'adx'
            - '+di'
            - '-di'

        Returns:
            DataWrapper: DataWrapper containing 'adx', '+di', and '-di' values for 
                          each step.
        """

        v_data = self.__data_adf if data is None else data

        atr = self.__idc_atr(length=length_di, smooth='smma').unwrap()

        dm_p_raw = v_data.loc[:, 'high'].diff()
        dm_n_raw = -v_data.loc[:, 'low'].diff()
        
        dm_p = pd.Series(
            np.where((dm_p_raw > dm_n_raw) & (dm_p_raw > 0), dm_p_raw, 0), 
            index=v_data.index)
        dm_n = pd.Series(
            np.where((dm_n_raw > dm_p_raw) & (dm_n_raw > 0), dm_n_raw, 0), 
            index=v_data.index)

        di_p = 100 * self.__idc_smma(dm_p, length=length_di).to_series() / atr
        di_n = 100 * self.__idc_smma(dm_n, length=length_di).to_series() / atr

        adx = self.__idc_smma(
            data=100 * np.abs((di_p - di_n) / (di_p + di_n).replace(0, 1)), 
            length=smooth).to_series()

        if only: 
            return adx
        adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

        return adx

    def idc_macd(self, short_len:int = 12, long_len:int = 26, 
                 signal_len:int = 9, macd_ma_type:str = 'ema', 
                 signal_ma_type:str = 'ema', histogram:bool = True, 
                 source:str = 'close', last:int | None = None) -> flx.DataWrapper:
        """
        Calculate the convergence/divergence of the moving average (MACD).

        This function calculates the MACD.

        Args:
            short_len (int, optional): Length of the short moving average used to calculate MACD.
            long_len (int, optional): Length of the long moving average used to calculate MACD.
            signal_len (int, optional): Length of the moving average for the MACD signal line.
            macd_ma_type (str, optional): Type of moving average used to calculate MACD.
            signal_ma_type (str, optional): Type of moving average used to smooth the MACD.
            histogram (bool, optional): If True, includes an additional 'histogram' column.
            source (str, optional): Data source for calculations. Allowed values: 'close', 
                'open', 'high', 'low'.
            last (int | None, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Columns:
            - 'macd'
            - 'signal'
            - 'histogram'

        Returns:
            DataWrapper: A DataWrapper with MACD values and the signal line for each step.
        """

        source = source.lower()
        if short_len > 5000 or short_len <= 0: 
            raise ValueError(utils.text_fix("""
                             'short_len' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif long_len > 5000 or long_len <= 0: 
            raise ValueError(utils.text_fix("""
                             'long_len' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif signal_len > 5000 or signal_len <= 0: 
            raise ValueError(utils.text_fix("""
                             'signal_len' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not macd_ma_type in ('ema','sma'): 
            raise ValueError(utils.text_fix("""
                             'macd_ma_type' only one of these values: 
                             ['ema','sma'].
                             """, newline_exclude=True))
        elif not signal_ma_type in ('ema','sma'): 
            raise ValueError(utils.text_fix("""
                             'signal_ma_typ' only one of these values: 
                             ['ema','sma'].
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc macd.
        return self.__idc_macd(short_len=short_len, long_len=long_len, 
                            signal_len=signal_len, macd_ma_type=macd_ma_type, 
                            signal_ma_type=signal_ma_type, histogram=histogram, 
                            source=source, last=last, cut=True)

    @__store_decorator
    def __idc_macd(self, data:pd.Series | None = None, short_len:int = 12, 
                   long_len:int = 26, signal_len:int = 9, 
                   macd_ma_type:str = 'ema', signal_ma_type:str = 'ema', 
                   histogram:bool = True, source:str = 'close', 
                   last:int | None = None, cut:bool = False) -> pd.DataFrame:
        """
        Calculate the convergence/divergence of the moving average (MACD).

        This function calculates the MACD.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (Series | None, optional): The data used for calculation of MACD.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'macd'
            - 'signal'
            - 'histogram'  

        Returns:
            DataWrapper: A DataWrapper with MACD values and signal line for each step.
        """

        v_data = self.__data_adf if data is None else data

        macd_ma = self.__idc_ema
        match macd_ma_type:
            case 'ema':
                macd_ma = self.__idc_ema
            case 'sma':
                macd_ma = self.__idc_sma

        signal_ma = self.__idc_ema
        match signal_ma_type:
            case 'ema':
                signal_ma = self.__idc_ema
            case 'sma':
                signal_ma = self.__idc_sma
        
        short_ema = macd_ma(data=v_data[source], length=short_len).to_series()
        long_ema = macd_ma(data=v_data[source], length=long_len).to_series()
        macd = short_ema - long_ema

        signal_line = signal_ma(data=macd, length=signal_len).to_series()

        result = pd.DataFrame({'macd':macd, 'signal':signal_line, 
                               'histogram':macd-signal_line} 
                               if histogram else 
                               {'macd':macd, 'signal':signal_line})

        return result

    def idc_sqzmom(self, bb_len:int = 20, bb_mult:float = 1.5, 
                   kc_len:int = 20, kc_mult:float = 1.5, 
                   use_tr:bool = True, source:str = 'close', 
                   last:int | None = None) -> flx.DataWrapper:
        """
        Calculate Squeeze Momentum (SQZMOM).

        This function calculates the Squeeze Momentum, inspired by the Squeeze 
        Momentum Indicator available on TradingView. While the concept is based 
        on the original indicator, this implementation may not fully replicate its 
        exact functionality. The concept credit goes to its original developer. 
        This function is intended for use in backtesting scenarios with real or 
        simulated data for research and educational purposes only, and should not 
        be considered financial advice.

        Args:
            bb_len (int, optional): Bollinger band length.
            bb_mult (float, optional): Bollinger band standard deviation.
            kc_len (int, optional): Keltner channel length.
            kc_mult (float, optional): Keltner channel standard deviation.
            use_tr (bool, optional): If False, ('high' - 'low') is used instead of the true 
                range.
            source (str, optional): Data source for calculations. Allowed values: 'close', 
                'open', 'high', 'low'.
            last (int | None, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Columns:
            - 'sqzmom'
            - 'histogram'

        Returns:
            DataWrapper: A DataWrapper with Squeeze Momentum values and histogram for 
                each step.
        """

        source = source.lower()
        if bb_len > 5000 or bb_len <= 0: 
            raise ValueError(utils.text_fix("""
                                            'bb_len' it has to be greater than 
                                            0 and less than 5000.
                                            """, newline_exclude=True))
        elif bb_mult > 50 or bb_mult < 0.001: 
            raise ValueError(utils.text_fix("""
                                            'bb_mult' it has to be greater than 
                                            0.001 and less than 50.
                                            """, newline_exclude=True))
        elif kc_len > 5000 or kc_len <= 0: 
            raise ValueError(utils.text_fix("""
                                            'kc_len' it has to be greater than 
                                            0 and less than 5000.
                                            """, newline_exclude=True))
        elif kc_mult > 50 or kc_mult < 0.001: 
            raise ValueError(utils.text_fix("""
                                            'bb_mult' it has to be greater than 
                                            0.001 and less than 50.
                                            """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                                            'source' only one of these values: 
                                            ['close','open','high','low'].
                                            """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc sqzmom.
        return self.__idc_sqzmom(bb_len=bb_len, bb_mult=bb_mult, 
                                kc_len=kc_len, kc_mult=kc_mult, 
                                use_tr=use_tr, source=source, 
                                last=last, cut=True)

    @__store_decorator
    def __idc_sqzmom(self, data:pd.Series | None = None, 
                     bb_len:int = 20, bb_mult:float = 1.5, 
                     kc_len:int = 20, kc_mult:float = 1.5, 
                     use_tr:bool = True, source:str = 'close', 
                     last:int | None = None, cut:bool = False) -> pd.DataFrame:
        """
        Calculate Squeeze Momentum (SQZMOM).

        This function calculates the Squeeze Momentum, inspired by the Squeeze 
        Momentum Indicator available on TradingView. While the concept is based 
        on the original indicator, this implementation may not fully replicate its 
        exact functionality. The concept credit goes to its original developer. 
        This function is intended for use in backtesting scenarios with real or 
        simulated data for research and educational purposes only, and should not 
        be considered financial advice.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (Series | None, optional): The data used for calculating the Squeeze Momentum.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'sqzmom'
            - 'histogram'

        Returns:
            DataWrapper: A DataWrapper with Squeeze Momentum values and histogram for 
                each step.
        """

        v_data = self.__data_adf if data is None else data

        basis = self.__idc_sma(length=bb_len).unwrap()
        dev = bb_mult * v_data.loc[:, source].rolling(window=bb_len).std(ddof=0)

        upper_bb = basis + dev
        lower_bb = basis - dev

        ma = self.__idc_sma(length=kc_len).unwrap()
        range_ = self.__idc_sma(data=self.__idc_trange().to_series()
                                if use_tr else v_data['high']-v_data['low'], 
                                length=kc_len).unwrap()

        upper_kc = ma + range_ * kc_mult
        lower_kc = ma - range_ * kc_mult

        sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

        d = v_data[source] - ((v_data.loc[:, 'low'].rolling(window=kc_len).min() + 
                             v_data.loc[:, 'high'].rolling(window=kc_len).max()) / 2 + 
                             self.__idc_sma(length=kc_len).unwrap()) / 2

        histogram = self.__idc_rlinreg(data=d, length=kc_len, offset=0).unwrap()

        result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=v_data.index), 
                               'histogram':histogram}, 
                               index=v_data.index)
        return result

    @__store_decorator
    def __idc_rlinreg(self, data:pd.Series | None = None, 
                      length:int = 5, offset:int = 1,
                      cut:bool = False) -> pd.Series:
        """
        Calculate rolling linear regression values.

        This function calculates the rolling linear regression.

        Note:
            This is a hidden function intended to prevent user 
            modification and does not include exception handling.

        Args:
            data (Series | None, optional): The data used for linear regression calculations.
            length (int, optional): Length of each window for the rolling regression.
            offset (int, optional): Offset used in the regression calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: Array with the linear regression values for each window.
        """

        v_data = self.__data_adf  if data is None else data

        x = np.arange(length)
        y = v_data.rolling(window=length)

        m = y.apply(lambda y: np.polyfit(x, y.values, 1)[0])
        b = y.mean() - (m * float(np.mean(x))) 

        return cast(pd.Series, m * (length - 1 - offset) + b)

    def idc_mom(self, length:int = 10, source:str = 'close', 
                last:int | None = None) -> flx.DataWrapper:
        """
        Calculate momentum values (MOM).

        This function calculates the MOM.

        Args:
            length (int, optional): Length for calculating momentum.
            source (str, optional): Data source for momentum calculation. Allowed values:
                'close', 'open', 'high', 'low'.
            last (int | None, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Returns:
            DataWrapper: DataWrapper with the momentum values for each step.
        """

        source = source.lower()
        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 
                             0 and less than 5000.
                             """, newline_exclude=True))
        elif not source in ('close','open','high','low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['close','open','high','low'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc momentum.
        return self.__idc_mom(length=length, source=source, 
                              last=last, cut=True)

    @__store_decorator
    def __idc_mom(self, data:pd.Series | None = None, length:int = 10, 
                  source:str = 'close', last:int | None = None,
                  cut:bool = False) -> pd.Series:
        """
        Calculate momentum values (MOM).

        This function calculates the MOM.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (Series | None, optional): The data used to calculate momentum.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper with the momentum values for each step.
        """

        v_data = self.__data_adf if data is None else data
        mom:pd.Series = v_data[source] - v_data.loc[:, source].shift(length)

        return mom

    def idc_ichimoku(self, tenkan_period:int = 9, kijun_period:int = 26, 
                     senkou_span_b_period:int = 52, ichimoku_lines:bool = True, 
                     last:int | None = None) -> flx.DataWrapper:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Args:
            tenkan_period (int, optional): Window length to calculate the Tenkan-sen line.
            kijun_period (int, optional): Window length to calculate the Kijun-sen line.
            senkou_span_b_period (int, optional): Window length to calculate the Senkou Span B.
            ichimoku_lines (bool, optional): If True, adds the columns 'tenkan_sen' and
                'kijun_sen' to the returned DataFrame.
            last (int | None, optional): Number of data points to return starting from the
                present backwards. If None, returns data for all available periods.

        Columns:
            - 'senkou_a'
            - 'senkou_b'
            - 'tenkan_sen'
            - 'kijun_sen'
            - 'ichimoku_lines'

        Returns:
            DataWrapper: A DataWrapper with Ichimoku cloud values and optionally
                'tenkan_sen' and 'kijun_sen' columns if `ichimoku_lines` is True.
        """

        if tenkan_period > 5000 or tenkan_period <= 0: 
            raise ValueError(utils.text_fix("""
                                            'tenkan_period' it has to be 
                                            greater than 0 and less than 5000.
                                            """, newline_exclude=True))
        elif kijun_period > 5000 or kijun_period <= 0: 
            raise ValueError(utils.text_fix("""
                                            'kijun_period' it has to be 
                                            greater than 0 and less than 5000.
                                            """, newline_exclude=True))
        elif senkou_span_b_period > 5000 or senkou_span_b_period <= 0: 
            raise ValueError(utils.text_fix("""
                                            'senkou_span_b_period' it has to be 
                                            greater than 0 and less than 5000.
                                            """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        
        # Calc ichimoku.
        return self.__idc_ichimoku(tenkan_period=tenkan_period, 
                                    kijun_period=kijun_period, 
                                    senkou_span_b_period=senkou_span_b_period, 
                                    ichimoku_lines=ichimoku_lines, 
                                    last=last, cut=True)

    @__store_decorator
    def __idc_ichimoku(self, data:pd.Series | None = None, tenkan_period:int = 9, 
                       kijun_period:int = 26, senkou_span_b_period:int = 52, 
                       ichimoku_lines:bool = True, 
                       last:int | None = None, cut:bool = False) -> pd.DataFrame:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (Series | None, optional): The data used to calculate the Ichimoku cloud values.
            cut (bool, optional): True to return the trimmed data with current index.

        Columns:
            - 'senkou_a'
            - 'senkou_b'
            - 'tenkan_sen'
            - 'kijun_sen'
            - 'ichimoku_lines'

        Returns:
            DataWrapper: A DataWrapper with Ichimoku cloud values and optionally
                'tenkan_sen' and 'kijun_sen' columns if `ichimoku_lines` is True.
        """

        v_data = self.__data_adf if data is None else data

        tenkan_sen_val = (v_data.loc[:, 'high'].rolling(window=tenkan_period).max() + 
                          v_data.loc[:, 'low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen_val = (v_data.loc[:, 'high'].rolling(window=kijun_period).max() + 
                         v_data.loc[:, 'low'].rolling(window=kijun_period).min()) / 2

        senkou_span_a_val = ((tenkan_sen_val + kijun_sen_val) / 2)
        senkou_span_b_val = ((v_data.loc[:, 'high'].rolling(
            window=senkou_span_b_period).max() + 
            v_data.loc[:, 'low'].rolling(window=senkou_span_b_period).min()) / 2)
        senkou_span = (pd.DataFrame({'senkou_a':senkou_span_a_val,
                                    'senkou_b':senkou_span_b_val, 
                                    'tenkan_sen':tenkan_sen_val,
                                    'kijun_sen':kijun_sen_val}) 
                      if ichimoku_lines else 
                        pd.DataFrame({'senkou_a':senkou_span_a_val,
                                      'senkou_b':senkou_span_b_val}))
        
        return senkou_span

    def idc_atr(self, length:int = 14, smooth:str = 'smma', 
                last:int | None = None) -> flx.DataWrapper:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Args:
            length (int, optional): Window length used to smooth the average true range (ATR).
            smooth (str, optional): Type of moving average used to smooth the ATR. 
            last (int | None, optional): Number of data points to return starting from the 
                present backward. If None, returns data for all available periods.

        Returns:
            DataWrapper: DataWrapper with the average true range values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 
                             0 and less than 5000.
                             """, newline_exclude=True))
        elif not smooth in ('smma', 'sma','ema','wma'): 
            raise ValueError(utils.text_fix("""
                             'smooth' only these values: 
                             'smma', 'sma', 'ema', 'wma'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > len(self.__data['close']))): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        # Calc atr.
        return self.__idc_atr(length=length, smooth=smooth, 
                              last=last, cut=True)

    @__store_decorator
    def __idc_atr(self, length:int = 14, smooth:str = 'smma', 
                  last:int | None = None, cut:bool = False) -> np.ndarray:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: Series with the average true range values for each step.
        """

        tr = self.__idc_trange().to_series()

        match smooth:
            case 'wma':
                atr:np.ndarray = self.__idc_wma(data=tr, length=length, 
                                                last=last).unwrap()
            case 'sma':
                atr:np.ndarray = self.__idc_sma(data=tr, length=length, 
                                                last=last).unwrap()
            case 'ema':
                atr:np.ndarray = self.__idc_ema(data=tr, length=length, 
                                                last=last).unwrap()
            case 'smma':
                atr:np.ndarray = self.__idc_smma(data=tr, length=length, 
                                                 last=last).unwrap()
            case _:
                atr:np.ndarray = self.__idc_wma(data=tr, length=length, 
                                                last=last).unwrap()

        return atr

    @__store_decorator
    def __idc_trange(self, data:pd.Series | None = None, 
                     handle_na: bool = True, last:int | None = None,
                     cut:bool = False) -> pd.Series:
        """
        Calculate the true range.

        This function calculates the true range.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (Series | None, optional): The data used to perform the calculation.
            handle_na (bool, optional): Whether to handle NaN values in 'close'.
            last (int | None, optional): Number of data points to return starting from the 
                present backward. If None, returns data for all available periods.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            DataWrapper: DataWrapper with the true range values for each step.
        """

        v_data = self.__data_adf if data is None else data

        close = v_data.loc[:, 'close'].shift(1)

        if handle_na:
                close.fillna(v_data['low'], inplace=True)
                     
        hl = v_data.loc[:, 'high'] - v_data.loc[:, 'low']
        hyc = abs(v_data['high'] - close)
        lyc = abs(v_data['low'] - close)
        tr:pd.Series[float] = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)

        if not handle_na:
            tr[close.isna()] = np.nan

        return tr
