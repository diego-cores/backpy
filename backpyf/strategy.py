"""
Strategy module

This module contains the main class that must be inherited to create your 
own strategy.

Classes:
    StrategyClass: This is the class you have to inherit to create your strategy.

Functions:
    idc_decorator: Create your own indicator.

Hidden Functions:
    _data_info: Gathers information about the dataset.
"""

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable
from uuid import uuid4
from time import time

from . import flex_data as flx
from . import _commons as _cm
from . import exception
from . import utils

def idc_decorator(func:callable) -> Callable[..., flx.DataWrapper]:
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
            ema = data['Close'].ewm(span=20, adjust=False).mean()
            return ema # Will be wrapped in a DataWrapper automatically

    Important:
        - The indicator will be calculated **only once**, and stored.
        - Access it from `next()` using `self.idc_test`.
        - You are responsible for ensuring the indicator logic does not use future data.
        (e.g. avoid calculating max of all data at step 1, which would invalidate the backtest).

    Args:
        func (callable): Function.

    Return:
        callable: Function.
    """

    func._uidc = True
    return func

def _data_info() -> tuple:
    """
    Data Info

    Returns all 'data' variables except `__data`.

    Returns:
        tuple: A tuple containing the following variables in order:
            - __data_interval (str): Data interval.
            - __data_icon (str): Data icon.
            - __data_width (int): Data index width.
    """

    return _cm.__data_interval, _cm.__data_icon, _cm.__data_width

class StrategyClass(ABC):
    """
    StrategyClass

    This is the class you have to inherit to create your strategy.

    To use the functions, use the `self` instance. Create your strategy 
    within the `StrategyClass.next()` structure.

    Attributes:
        open: Last 'Open' value from `data`.
        high: Last 'High' value from `data`.
        low: Last 'Low' value from `data`.
        close: Last 'Close' value from `data`.
        volume: Last 'Volume' value from `data`.
        date: Last index from `data`.
        width: Data width from `__data_width`.
        icon: Data icon from `__data_icon`.
        interval: Data interval from `__data_interval`.

    Private Attributes:
        __init_funds: Initial funds for the strategy.
        __spread_pct: Closing and opening spread.
        __slippage_pct: Closing and opening slippage.
        __commission: Commission per trade.
        __trade: DataFrame for new trades.
        __trades_ac: DataFrame for open trades.
        __trades_cl: DataFrame for closed trades.
        __data: DataFrame containing cuted data.
        __data_all: DataFrame containing all data.
        __data_index: Index to cut data.
        __idc_data: Saved data from the indicators.
        __buffer: Buffer to concatenate DataFrame.

    Methods:
        unique_id: .
        act_taker: .
        act_limit: .
        act_close: .
        ord_put: .
        ord_rmv: .
        ord_mod: .
        get_spread: Get __spread_pct.
        get_slippage: Get __slippage_pct.
        get_commission: Returns the commission per trade.
        get_init_funds: Returns the initial funds for the strategy.
        prev: Recovers all step data.
        prev_orders: .
        prev_positions: .
        prev_pos_rec: .
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
        __act_reduce: .
        __put_op: .
        __ord_put: .
        __order_execute: .
        __price_check: .
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

    def __init__(self, data:pd.DataFrame = pd.DataFrame(), 
                 spread_pct:flx.CostsValue = 0, 
                 commission:flx.CostsValue = 0, 
                 slippage_pct:flx.CostsValue = 0, 
                 init_funds:int = 0) -> None: 
        """
        __init__

        Builder for initializing the class.

        Args:
            data (pd.DataFrame, optional): All data from the step and previous ones.
            trades_cl (pd.DataFrame, optional): Closed trades.
            trades_ac (pd.DataFrame, optional): Open trades.
            spread_pct (CostsValue, optional): Spread per trade.
            commission (CostsValue, optional): Commission per trade.
            slippage_pct (CostsValue, optional): Slippage per trade.
            init_founds (int, optional): Initial funds for the strategy.
        """

        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.date = None

        self.__data = pd.DataFrame()
        self.__data_all = data
        self.__data_index = None

        self.__idc_data = {}
        self.__buffer = {'__orders':[]}

        self.interval, self.icon, self.width = _data_info()

        self.__spread_pct  = spread_pct
        self.__slippage_pct = slippage_pct
        self.__commission = commission
        self.__init_funds = init_funds

        self.__orders = pd.DataFrame()
        self.__positions = pd.DataFrame()

        self.__orders['order'] = pd.Categorical(
            [], categories=['op', 'stopLoss', 'takeProfit'], ordered=True)

        self.__pos_record = pd.DataFrame()

        for name in dir(self):
            attr = getattr(self, name)

            if not callable(attr):
                continue
            elif getattr(attr, '_store', False):
                decorator = self._StrategyClass__data_store(attr)
                setattr(self, name, decorator)
            elif getattr(attr, '_uidc', False):
                decorator = self._StrategyClass__uidc(attr)
                setattr(self, name, decorator)

    @abstractmethod
    def next(self) -> None: ...

    def get_spread(self) -> flx.CostsValue:
        """
        Get __spread_pct

        Info:
            To get the value use: 'get_maker' or 'get_taker'.
            In this case they will return the same.

        Returns:
            flx.CostsValue: The value of the hidden variable `__spread_pct`.
        """

        return self.__spread_pct

    def get_slippage(self) -> flx.CostsValue:
        """
        Get __slippage_pct

        Info:
            To get the value use: 'get_maker' or 'get_taker'.
            In this case they will return the same.

        Returns:
            flx.CostsValue: The value of the hidden variable `__slippage_pct`.
        """

        return self.__slippage_pct

    def get_commission(self) -> flx.CostsValue:
        """
        Get __commission

        Info:
            To get the value use: 'get_maker' or 'get_taker'.

        Returns:
            float: The value of the hidden variable `__commission`.
        """

        return self.__commission
    
    def get_init_funds(self) -> int:
        """
        Get __init_funds

        Returns:
            float: The value of the hidden variable `__init_funds`.
        """

        return self.__init_funds

    def __store_decorator(func:callable) -> callable:
        """
        Store decorator

        Decorate a function with this to give it 
            the attribute: '_store' and have it decorated with '__data_store'.

        Args:
            func (callable): Function.

        Return:
            callable: Function.
        """

        func._store = True
        return func

    def __data_store(self, func:callable) -> callable:
        """
        Data store

        Save the function return and, if already saved, return it from storage.

        Args:
            func (callable): Function.

        Return:
            callable: Wrapper function.
        """

        def __wr_func(*args, **kwargs) -> pd.DataFrame:
            """
            Wrapper function

            Save the function return and, if already saved, return it from storage.

            Return:
                pd.DataFrame: Function result.
            """

            id, arguments = StrategyClass.__func_idg(func, *args, **kwargs)
  
            if id in self.__idc_data:
                if arguments.get('cut', False):
                    return self.__data_cut(self.__idc_data[id],
                                           arguments.get('last', None))

                return self.__idc_data[id]

            result = func(*args, **kwargs)
            self.__idc_data[id] = result

            if arguments.get('cut', False):
                return self.__data_cut(result, 
                                       arguments.get('last', None))

            return result
        return __wr_func

    def __uidc(self, func:callable) -> callable:
        """
        User indicator

        Send data argument to the indicator and save the result.

        Args:
            func (callable): Function.

        Return:
            callable: Wrapper function.
        """

        func = staticmethod(func).__func__

        @wraps(func)
        def __wr_func(*args, **kwargs) -> flx.DataWrapper:
            """
            Wrapper function

            Save the function return and, if already saved, return it from storage.

            Sends '__data_all' to the 'data' argument.

            Return:
                DataWrapper: Function result.
            """

            id = StrategyClass.__func_idg(func, *args, **kwargs)[0]

            if id in self.__idc_data.keys():
                return self.__uidc_cut(self.__idc_data[id])

            result = flx.DataWrapper(func.__func__(self.__data_all, *args, **kwargs))
            self.__idc_data[id] = result

            return self.__uidc_cut(result)
        return __wr_func

    def __func_idg(func:callable, *args, **kwargs) -> tuple:
        """
        Function id generator

        Generates an id for a function call 
            and returns all arguments with defaults.

        Args:
            func (callable): Function.

        Return:
            tuple: Generated id and arguments.
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
            data (flx.DataWrapper): Data to cut.

        Return:
            flx.DataWrapper: Data cut.
        """

        if len(data) != len(self.__data_all):
            raise exception.UidcError('Length different from data.')

        result = flx.DataWrapper(data[:self.__data_index], data._columns)
        result._index = data._index

        return result

    def __data_cut(self, data:pd.DataFrame, last:int = None) -> pd.DataFrame:
        """
        Data cut

        Slices data for the user based on the current index.

        Args:
            data (pd.DataFrame): Data to cut.
            last (int, optional): You can get only the latest 'last' data.

        Return:
            pd.DataFrame: Data cut.
        """

        limit = self.__data_index
        return (data.iloc[limit-last:limit] 
                if last is not None and last < limit
                else data.iloc[:limit])

    def __data_updater(self, index:int) -> None:
        """
        Data updater.

        Updates all data with the provided DataFrame.

        Args:
            data (pd.DataFrame): All data from the step and previous ones.
        """

        data = self.__data_all.iloc[:index]
        data_ = data.values[-1]

        self.open = data_[0]
        self.high = data_[1]
        self.low = data_[2]
        self.close = data_[3]
        self.volume = data_[4]
        self.date = data.index[-1]

        self.__data = data
        self.__data_index = index

    def __before(self, index:int) -> None:
        """
        Before.

        This function is used to run trades and other operations.

        Args:
            index (int): Current data index.
        """

        self.__data_updater(index=index)

        # Check operations
        if not self.__orders.empty:
            self.__orders = self.__orders.sort_values('order')

            higher = self.__data['High'].values[-1]*(self.__spread_pct.get_taker()/100/2+1)
            lower = self.__data['Low'].values[-1]*(1-self.__spread_pct.get_taker()/100/2)

            def get_union_cn(row, data):

                leak  = self.__get_union(data, row['unionId'])
                if leak is None:
                    return

                if leak['typeSide'].values[0] == row['typeSide']:
                    return (not row['limit'] and higher >= row['orderPrice'])

                return (not row['limit'] and lower <= row['orderPrice'])

            for index, row in self.__orders.iterrows():
                if not index in self.__orders.index:
                    continue

                row_split = row['unionId'].split('/')
                if (row_split[0] == 'w' 
                    and row_split[-1] in self.__positions['unionId'].values):

                    row['unionId'] = self.__orders.loc[index, 'unionId'] = row_split[-1]
                elif row_split[0] == 'w':
                    continue

                limit = (row['limit'] and higher >= row['orderPrice']
                and lower <= row['orderPrice'])

                ord_cn = get_union_cn(row, self.__orders)
                pos_cn = get_union_cn(row, self.__positions)

                if limit or pos_cn or ord_cn:
                    self.__order_execute(row)

                    self.__orders = self.__orders.drop(index, errors='ignore')

        self.next()

        if len(self.__buffer['__orders']) > 0:
            self.__orders = pd.concat([self.__orders, 
                                    pd.DataFrame(self.__buffer['__orders'])], ignore_index=True) 
            self.__buffer['__orders'] = list()

    def unique_id(self = None):
        """
        """

        return str(uuid4().int)

    def act_taker(self, buy:bool = True, amount:float = np.nan):
        """
        """

        buy = bool(buy)
        ui = self.unique_id()

        order = pd.Series({
            'date':self.__data.index[-1],
            'orderPrice': self.__data["Close"].iloc[-1],
            'amount':amount,
            'typeSide':buy,
            'unionId':ui,
        })

        self.__put_op(order)
        return ui

    def act_limit(self, price:float, buy:bool = True, amount:float = np.nan):
        """
        """

        buy = bool(buy)

        return self.__ord_put(
            'op', 
            price=price, 
            amount=amount, 
            buy=buy, 
            wait=False, 
            union_id=self.unique_id(),
            limit=True,
        )

    def act_close(self, index):
        """
        """

        self.__act_reduce(index, self.__data['Close'].iloc[-1], mode='taker')

    def __act_reduce(self, index, price, amount = None, mode = 'taker'):
        """
        """

        position_frame = self.__positions.iloc[[index]]
        position_series = position_frame.iloc[0]

        pos_amount = position_series['amount']
        if np.isnan(amount):
            amount = pos_amount

        position_close = price

        if mode == 'maker':

            commission = self.__commission.get_maker()
            position_close_spread = price
        elif mode == 'taker':

            commission = self.__commission.get_taker()
            spread = price*(self.__spread_pct.get_taker()/100/2)
            slippage = price*(self.__slippage_pct.get_taker()/100)

            position_close_spread = (position_close-spread-slippage 
                                    if position_series['typeSide']
                                    else position_close+spread+slippage)
        else:
            return

        # Fill data.
        position_frame['positionClose'] = position_close_spread
        position_frame['positionDate'] = self.__data.index[-1]
        open = position_close_spread

        position_frame['profitPer'] = ((position_close_spread-open)/open*100 
                            if position_series['typeSide']
                            else (open-position_close_spread)/open*100)

        if pos_amount > amount:

            self.__positions.iloc[index, self.__positions.columns.get_loc('amount')] -= amount
            position_frame['amount'] = amount
        else:
            self.__positions = self.__positions.drop(position_series.name)

            self.__orders = self.__orders[
                (position_series['unionId'] != self.__orders['unionId'].str.split('/').str[-1])
                & (position_series['unionId'] != self.__orders['closeId'])]

        if not np.isnan(pos_amount):
            profit_per = position_frame['profitPer'].values[0]

            gross_profit = pos_amount * profit_per / 100
            broker_fee = pos_amount * (commission / 100)
            exit_fee = ((gross_profit + pos_amount) 
                        * (position_series['commission'] / 100))

            position_frame['profit'] = gross_profit - broker_fee - exit_fee
        else:
            position_frame['profit'] = np.nan

        position_frame['commission'] += commission

        self.__pos_record = pd.concat([self.__pos_record, position_frame], ignore_index=True) 

    def __put_op(self, order:pd.Series):
        """
        """

        position_price = order['orderPrice']

        if (
            position_price >= self.__data['Close'].loc[order['date']]*(
                self.__spread_pct.get_taker()/100/2+1)
            or position_price <= self.__data['Close'].loc[order['date']]*(
                1-self.__spread_pct.get_taker()/100/2)
            ):

            position_open = position_price
            commission = self.__commission.get_maker()
        else:
            position_price = self.__data['Close'].loc[order['date']]

            commission = self.__commission.get_taker()
            slippage = position_price*(self.__slippage_pct.get_taker()/100)
            spread = position_price*(self.__spread_pct.get_taker()/100/2)

            position_open = (position_price+spread+slippage
                            if order['typeSide'] else position_price-spread-slippage)

        position = pd.DataFrame({
            'date':self.__data.index[-1],
            'positionOpen':position_open,
            'commission':commission,
            'amount':order['amount'],
            'typeSide':order['typeSide'],
            'unionId':order['unionId'],
        }, index=[1])

        self.__positions = pd.concat([self.__positions, position], ignore_index=True) 

    def __order_execute(self, order):
        """
        """

        # Hello world
        match order['order']:
            case 'op':
                self.__put_op(order)

            case 'takeProfit' | 'stopLoss':
                if pd.isna(order['unionId']):
                    return

                union_pos = self.__positions['unionId'][
                    self.__positions['unionId'].values == order['unionId']]
                if union_pos.empty:
                    return 

                min_gap = min(self.__data['Open'].values[-1], self.__data['Close'].values[-2])
                max_gap = max(self.__data['Open'].values[-1], self.__data['Close'].values[-2])

                if order['orderPrice'] > min_gap and order['orderPrice'] < max_gap:
                    order['orderPrice'] = self.__data['Open'].values[-1]

                self.__act_reduce(
                    self.__positions.index.get_loc(union_pos.index[0]), 
                        order['orderPrice'], amount=order['amount'])

            case 'takeLimit' | 'stopLimit':
                pass

        mask = self.__orders['closeId'] != order['id']
        if not pd.isna(order['closeId']):
            mask &= self.__orders['closeId'] != order['closeId']

        self.__orders = self.__orders[mask]

    def ord_put(self, order_type:str, price:float, 
                amount:float = None, buy:bool = True, 
                union_id:int = None, close_id:int = None):
        """
        """

        if not order_type in (
            'stopLoss', 'takeProfit', 'takeLimit', 'stopLimit'):
            raise ValueError('Bad op type')

        last_data = None
        if not self.__positions.empty:
            last_data = self.__positions
        elif not self.__orders.empty:
            last_data = self.__orders
        elif not self.__pos_record.empty:
            last_data = self.__pos_record

        last_ui = last_data.iloc[-1]['unionId'] if isinstance(last_data, pd.DataFrame) else None
        union_id = union_id or (last_ui.split('/')[-1] if last_ui else 0)

        return self.__ord_put(
            order_type, 
            price=price, 
            amount=amount, 
            buy=buy, 
            union_id=union_id,
            close_id=close_id,
            limit=True if order_type in ('stopLimit', 'takeLimit') else False,
        )

    def __get_union(self, data, union_id):
        """
        """
        if data.empty:
            return None

        mask = data['unionId'].values == union_id
        if 'order' in data.columns:
            mask &= data['order'].values == 'op'
        data_leak = data[mask]

        return data_leak if not data_leak.empty else None

    def __price_check(self, price, union_id, type_side):
        """
        """

        if pd.isna(union_id):
            return

        union_id = union_id.split('/')[-1]
        func = np.max if type_side else np.min

        def get_union_price(data, col_name):
            """
            """

            nonlocal func
            leak = self.__get_union(data=data, union_id=union_id)

            if not leak is None:

                func = np.max if leak['typeSide'].iloc[0] == type_side else np.min
                return func(leak[col_name])

        union_ord = get_union_price(self.__orders, 'orderPrice')
        union_pos = get_union_price(self.__positions, 'positionOpen')

        union_pos = union_pos or union_ord or price
        union_ord = union_ord or union_pos

        price_cn = func([union_pos, union_ord])
        return price >= price_cn if func == np.max else price <= price_cn

    def __ord_put(self, order_type:str, price:float, 
                  amount:float = None, buy:bool = True, wait:bool = True,
                  union_id:int = None, close_id:int = None, limit:bool = False):
        """
        """

        amount = amount or np.nan

        close_id = str(close_id) if close_id else np.nan
        union_id = f'{"w/" if wait else ""}{union_id}' if union_id else np.nan

        match order_type:
            case 'op':
                buy = buy
            case 'stopLoss' | 'stopLimit':
                buy = False
            case 'takeProfit' | 'takeLimit':
                buy = True
            case _:
                raise exception.OrderError('Invalid order type.')

        if (order_type != 'op' 
            and ~self.__price_check(price=price, union_id=union_id, type_side=buy)):
            raise exception.OrderError("Invalid 'price' for order type.")

        order = {
            'order':order_type,
            'date':self.__data.index[-1],
            'orderPrice':price,
            'amount':amount,
            'typeSide':buy,
            'id':self.unique_id(),
            'unionId':union_id,
            'closeId':close_id,
            'limit':limit
        }

        self.__buffer['__orders'].append(order)

        return union_id

    def ord_rmv(self, index:int):
        """
        """

        order = self.__orders.iloc[[index]]
        self.__orders = self.__orders.drop(order.index)

        self.__orders = self.__orders[
            (order['id'].iloc[0] != self.__orders['unionId'].str.split('/').str[-1])
            & (order['id'].iloc[0] != self.__orders['closeId'])]

    def ord_mod(self, index:int, price:float = None, amount:float = None):
        """
        """

        order = self.__orders.iloc[[index]]

        union = self.__price_check(
            price=price, 
            union_id=order['unionId'].values[0], 
            type_side=order['typeSide'].values[0])

        if union and not price is None:
            self.__orders.loc[order.index, 'orderPrice'] = price
        if not amount is None:
            self.__orders.loc[order.index, 'amount'] = amount

    def prev(self, label:str = None, last:int = None) -> flx.DataWrapper:
        """
        Prev.

        This function returns the values of `data`.
        
        Args:
            label (str, optional): Data column to return. If None, all columns 
                are returned. If 'index', only indexes are returned, ignoring 
                the `last` parameter.
            last (int, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `data` columns.

            - Open: The 'Open' price of the step.
            - High: The 'High' price of the step.
            - Low: The 'Low' price of the step.
            - Close: The 'Close' price of the step.
            - Volume: The 'Volume' of the step.
            - index: The 'Index' of the step.

        Returns:
            DataWrapper: DataWrapper containing the data of previous steps.
        """

        __data = self.__data
        if label == 'index': 
            return flx.DataWrapper(__data.index, columns='index')
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data_columns = __data.columns
        data = __data.values[
            len(__data) - last if last is not None and last < len(__data) else 0:]

        if label != None: 
            _loc = __data.columns.get_loc(label)

            data_columns = data_columns[_loc]
            data = data[:,_loc]

        return flx.DataWrapper(data, columns=data_columns)

    def prev_pos_rec(self, label:str = None, last:int = None) -> flx.DataWrapper:
        """
        Prev of trades closed.

        This function returns the values of `trades_cl`.

        Args:
            label (str, optional): Data column to return. If None, all columns 
                are returned. If 'index', only indexes are returned, ignoring 
                the `last` parameter.
            last (int, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `__pos_record` columns.

            - date: Creation date.
            - positionOpen: Opening price.
            - commission: Position commissions.
            - amount: Position amount
            - typeSide: Position type True for buy.
            - unionId: Id linked to orders.
            - positionClose: Closing price.
            - positionDate: Closing date.
            - profitPer: Profit in percentage with out commissions.
            - profit: Profit on 'amount' with commissions.

        Returns:
            DataWrapper: DataWrapper containing the data from closed trades.
        """

        __pos_rec = self.__pos_record
        if label == 'index': 
            return flx.DataWrapper(__pos_rec.index, columns='index')
        elif __pos_rec.empty: 
            return flx.DataWrapper()
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data_columns = __pos_rec.columns
        data = __pos_rec.values[
            len(__pos_rec) - last if last is not None and last < len(__pos_rec) else 0:]
        
        if label != None: 
            _loc = __pos_rec.columns.get_loc(label)

            data_columns = data_columns[_loc]
            data = data[:,_loc]

        return flx.DataWrapper(data, columns=data_columns)

    def prev_positions(self, label:str = None, last:int = None) -> flx.DataWrapper:
        """
        Prev of trades active.

        This function returns the values of `trades_ac`.

        Args:
            label (str, optional): Data column to return. If None, all columns 
                are returned. If 'index', only indexes are returned, ignoring 
                the `last` parameter.
            last (int, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `__positions` columns.

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
        if label == 'index': 
            return flx.DataWrapper(__pos.index, columns='index')
        elif __pos.empty: 
            return flx.DataWrapper()
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data_columns = __pos.columns
        data = __pos.values[
            len(__pos) - last if last is not None and last < len(__pos) else 0:]
    
        if label != None: 
            _loc = __pos.columns.get_loc(label)

            data_columns = data_columns[_loc]
            data = data[:,_loc]

        return flx.DataWrapper(data, columns=data_columns)

    def prev_orders(self, label:str = None, or_type:str = None, 
                    ids:dict = None, last:int = None) -> flx.DataWrapper:
        """
        Prev of active orders.

        This function returns the values of `__orders`.

        Args:
            label (str, optional): Data column to return. If None, all columns 
                are returned. If 'index', only indexes are returned, ignoring 
                the `last` parameter.
            or_type (str, optional): If you want to filter only one type 
                of operation you can do so with this argument.
                Current types of orders: 'op', 'takeProfit', 'stopLoss', 'takeLimit', 'stopLimit'.
            ids (dict, optional): Filter by id by making a dictionary with id_name:id.
            last (int, optional): Number of steps to return starting from the 
                present. If None, data for all times is returned.

        Info:
            `__orders` columns.

            - order: Order type ('op', 'takeProfit', 'stopLoss', 'takeLimit', 'stopLimit').
            - date: Creation date.
            - orderPrice: Execution price.
            - amount: Amount to be executed.
            - typeSide: Order type True for buy.
            - id: Unique order ID.
            - unionId: Linked id to position.
            - closeId: Close link.
            - limit: Indicates execution type to be performed.

        Returns:
            DataWrapper: DataWrapper containing the data from active trades.
        """

        __ord = self.__orders
        if __ord.empty: 
            return flx.DataWrapper()
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 

            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))
        elif ((not isinstance(ids, dict) and not ids is None) 
            or (isinstance(ids, dict) 
            and not set(ids.keys()).issubset(('id','unionId','closeId')))):

            raise ValueError(utils.text_fix(
                "'union_id' with bad format or incorrect.", newline_exclude=True))

        data_columns = __ord.columns

        if or_type != None:
            __ord = __ord[__ord['order'].values == or_type]
        if ids != None:
            mask = True

            # Optimized code
            if 'unionId' in ids:
                mask &= __ord['unionId'].values == ids['unionId']
            if 'id' in ids:
                mask &= __ord['id'].values == ids['id']
            if 'closeId' in ids:
                mask &= __ord['closeId'].values == ids['closeId']

            __ord = __ord[mask]

        if label == 'index': 
            return flx.DataWrapper(__ord.index, columns='index')

        data = __ord.values[
            len(__ord) - last if last is not None and last < len(__ord) else 0:]
    
        if label != None: 
            _loc = __ord.columns.get_loc(label)

            data_columns = data_columns[_loc]
            data = data[:,_loc]

        return flx.DataWrapper(data, columns=data_columns)

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
        return flx.DataWrapper(
            self.__idc_fibonacci(lv0=lv0, lv1=lv1))

    @__store_decorator
    def __idc_fibonacci(self, lv0:int = 10, lv1:int = 1) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.

        This function calculates the Fibonacci retracement levels.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Returns:
            pd.DataFrame: A DataFrame with Fibonacci levels and their corresponding
                values.

        Columns:
            - 'Level'
            - 'Value'
        """

        fibo_levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 
                                0.786, 1, 1.618, 2.618, 3.618, 4.236])

        return pd.DataFrame({'Level':fibo_levels,
                             'Value':lv0 - (lv0 - lv1) * fibo_levels})

    def idc_ema(self, length:int = any, 
                source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Exponential moving average (EMA).

        This function calculates the EMA.

        Args:
            length (int): The length of the EMA.
            source (str, optional): The data source for the EMA calculation. Allowed 
                parameters are 'Close', 'Open', 'High', 'Low', and 'Volume'.
            last (int, optional): Number of data points to return from the 
                present backwards. If None, returns data for all time.

        Returns:
            DataWrapper: DataWrapper containing the EMA values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('Close','Open','High','Low','Volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low','Volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Ema calc.
        return flx.DataWrapper(self.__idc_ema(length=length, source=source, 
                                              last=last, cut=True))

    @__store_decorator
    def __idc_ema(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None, 
                  cut:bool = False) -> pd.Series:
        """
        Exponential Moving Average (EMA).

        This function calculates the EMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the EMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series containing the EMA values for each step.
        """

        data = self.__data_all[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()

        return ema

    def idc_sma(self, length:int = any, 
                source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Args:
            length (int): Length of the SMA.
            source (str, optional): Data source for SMA calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.
        
        Returns:
            DataWrapper: DataWrapper containing the SMA values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('Close','Open','High','Low','Volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low','Volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Sma calc.
        return flx.DataWrapper(self.__idc_sma(length=length, source=source, 
                                              last=last, cut=True))

    @__store_decorator
    def __idc_sma(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None, 
                  cut:bool = False) -> pd.Series:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the SMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series containing the SMA values for each step.
        """

        data = self.__data_all[source] if data is None else data
        sma = data.rolling(window=length).mean()

        return sma

    def idc_wma(self, length:int = any, source:str = 'Close', 
                invt_weight:bool = False, last:int = None) -> flx.DataWrapper:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Args:
            length (int): Length of the WMA.
            source (str, optional): Data source for WMA calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            invt_weight (bool, optional): If True, the distribution of weights is reversed.
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing the WMA values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('Close','Open','High','Low','Volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low','Volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Wma calc.
        return flx.DataWrapper(self.__idc_wma(length=length, source=source, 
                              invt_weight=invt_weight, last=last, cut=True))

    @__store_decorator
    def __idc_wma(self, data:pd.Series = None, 
                  length:int = any, source:str = 'Close', 
                  invt_weight:bool = False, last:int = None, 
                  cut:bool = False) -> pd.Series:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the WMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series containing the WMA values for each step.
        """

        data = self.__data_all[source] if data is None else data

        weight = (np.arange(1, length+1)[::-1] 
                  if invt_weight else np.arange(1, length+1))
        wma = data.rolling(window=length).apply(
            lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return wma
    
    def idc_smma(self, length:int = any, 
                 source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Args:
            length (int): Length of the SMMA.
            source (str, optional): Data source for SMMA calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing the SMMA values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 0 and 
                             less than 5000.
                             """, newline_exclude=True))
        elif not source in ('Close','Open','High','Low','Volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low','Volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Smma calc.
        return flx.DataWrapper(self.__idc_smma(length=length, source=source, 
                                               last=last, cut=True))

    @__store_decorator
    def __idc_smma(self, data:pd.Series = None, length:int = any, 
                   source:str = 'Close', last:int = None, 
                   cut:bool = False) -> pd.Series:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the SMMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series containing the SMMA values for each step.
        """

        data = self.__data_all[source] if data is None else data

        smma = data.ewm(alpha=1/length, adjust=False).mean()
        smma.shift(1)

        return smma
    
    def idc_sema(self, length:int = 9, method:str = 'sma', 
                  smooth:int = 5, only:bool = False, 
                  source:str = 'Close', last:int = None) -> flx.DataWrapper:
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
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing the 'ema' and 'smoothed' values for 
                              each step.
        
        Columns:
            - 'ema'
            - 'smoothed'
        """

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
        elif not source in ('Close','Open','High','Low','Volume'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low','Volume'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Sema calc.
        return flx.DataWrapper(self.__idc_sema(length=length, method=method, smooth=smooth, 
                                only=only, source=source, last=last, cut=True))
    
    @__store_decorator
    def __idc_sema(self, data:pd.Series = None, length:int = 9, 
                    method:str = 'sma', smooth:int = 5, only:bool = False, 
                    source:str = 'Close', last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Smoothed Exponential Moving Average (SEMA).

        This function calculates the SEMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the SEMA calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: DataFrame containing 'ema' and 'smoothed' values for 
                          each step.

        Columns:
            - 'ema'
            - 'smoothed'
        """

        data = self.__data_all[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()

        match method:
            case 'sma': smema = self.__idc_sma(data=ema, length=smooth)
            case 'ema': smema = self.__idc_ema(data=ema, length=smooth)
            case 'smma': smema = self.__idc_smma(data=ema, length=smooth)
            case 'wma': smema = self.__idc_wma(data=ema, length=smooth)

        if only: 
            smema = np.flip(smema)
            return np.flip(smema.iloc[len(smema)-last 
                                 if last != None and last < len(smema) else 0:])
        
        smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)
        return smema

    def idc_bb(self, length:int = 20, std_dev:float = 2, ma_type:str = 'sma', 
               source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Args:
            length (int, optional): Window length for calculating Bollinger Bands.
            std_dev (float, optional): Number of standard deviations for the bands.
            ma_type (str, optional): Type of moving average. For example, 'sma' for simple 
                          moving average.
            source (str, optional): Data source for calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing 'Upper', '{ma_type}', and 'Lower' 
                          values for each step.
                 
        Columns:
            - 'Upper'
            - '{ma_type}'
            - 'Lower'
        """

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
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low'].
                             """, newline_exclude=True))
        elif not ma_type in ('sma','ema','wma','smma'): 
            raise ValueError(utils.text_fix("""
                             'ma_type' only these values: 
                             'sma', 'ema', 'wma', 'smma'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Bb calc.
        return flx.DataWrapper(self.__idc_bb(length=length, std_dev=std_dev, 
                                ma_type=ma_type, source=source, last=last, cut=True))

    @__store_decorator
    def __idc_bb(self, data:pd.Series = None, length:int = 20, 
                 std_dev:float = 2, ma_type:str = 'sma', source:str = 'Close', 
                 last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the Bollinger Bands 
                calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: DataFrame containing 'Upper', '{ma_type}', and 'Lower' 
                          values for each step.
              
        Columns:
            - 'Upper'
            - '{ma_type}'
            - 'Lower'
        """

        data = self.__data_all[source] if data is None else data

        match ma_type:
            case 'sma': ma = self.__idc_sma(data=data, length=length)
            case 'ema': ma = self.__idc_ema(data=data, length=length)
            case 'wma': ma = self.__idc_wma(data=data, length=length)
            case 'smma': ma = self.__idc_smma(data=data, length=length)
        std_ = (std_dev * data.rolling(window=length).std())
        bb = pd.DataFrame({'Upper':ma + std_,
                           ma_type:ma,
                           'Lower':ma - std_}, index=ma.index)

        return bb

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, 
                rsi_ma_type:str = 'smma', base_type:str = 'sma', 
                bb_std_dev:float = 2, source:str = 'Close', 
                last:int = None) -> flx.DataWrapper:
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
                          ('Close', 'Open', 'High', 'Low').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing 'rsi' and '{base_type}' values for 
                          each step.

        Columns:
            - 'rsi'
            - '{base_type}'
        """

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
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low'].
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
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Rsi calc.
        return flx.DataWrapper(self.__idc_rsi(length_rsi=length_rsi, length=length, 
                              rsi_ma_type=rsi_ma_type, base_type=base_type, 
                              bb_std_dev=bb_std_dev, source=source, 
                              last=last, cut=True))

    @__store_decorator
    def __idc_rsi(self, data:pd.Series = None, length_rsi:int = 14, 
                  length:int = 14, rsi_ma_type:str = 'smma', 
                  base_type:str = 'sma', bb_std_dev:float = 2, 
                  source:str = 'Close', last:int = None, cut:bool = False)  -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        This function calculates the RSI.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the RSI calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: DataFrame containing 'rsi' and '{base_type}' values for 
                          each step.

        Columns:
            - 'rsi'
            - '{base_type}'
        """

        delta = self.__data_all[source].diff() if data is None else data.diff()

        match rsi_ma_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        ma_gain = ma(data = delta.where(delta > 0, 0), 
                     length=length_rsi, source=source)
        ma_loss = ma(data = -delta.where(delta < 0, 0), 
                     length=length_rsi, source=source)
        rsi = 100 - (100 / (1+ma_gain/ma_loss))

        match base_type:
            case 'sma': mv = self.__idc_sma(data=rsi, length=length)
            case 'ema': mv = self.__idc_ema(data=rsi, length=length)
            case 'wma': mv = self.__idc_wma(data=rsi, length=length)
            case 'smma': mv = self.__idc_smma(data=rsi, length=length)
            case 'bb': mv = self.__idc_bb(data=rsi, length=length, 
                                          std_dev=bb_std_dev)
        if type(mv) == pd.Series: mv.name = base_type

        rsi = pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

        return rsi

    def idc_stochastic(self, length_k:int = 14, smooth_k:int = 1, 
                       length_d:int = 3, d_type:str = 'sma', 
                       source:str = 'Close', last:int = None) -> flx.DataWrapper:
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
                          ('Close', 'Open', 'High', 'Low').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing 'stoch' and '{d_type}' values for each 
                          step.
        
        Columns:
            - 'stoch'
            - '{d_type}'
        """

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
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low'].
                             """, newline_exclude=True))
        elif not d_type in ('sma','ema','wma','smma'): 
            raise ValueError(utils.text_fix("""
                             'd_type' only these values: 
                             'sma', 'ema', 'wma', 'smma'.
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        # Calc stoch.
        return flx.DataWrapper(
            self.__idc_stochastic(length_k=length_k, smooth_k=smooth_k, 
                                length_d=length_d, d_type=d_type, 
                                source=source, last=last, cut=True))

    @__store_decorator
    def __idc_stochastic(self, data:pd.Series = None, length_k:int = 14, 
                         smooth_k:int = 1, length_d:int = 3, d_type:int = 'sma', 
                         source:str = 'Close', last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        This function calculates the stochastic oscillator.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the stochastic calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: DataFrame containing 'stoch' and '{d_type}' values for each 
                          step.

        Columns:
            - 'stoch'
            - '{d_type}'
        """

        data = self.__data_all if data is None else data

        low_data = data['Low'].rolling(window=length_k).min()
        high_data = data['High'].rolling(window=length_k).max()

        match d_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        stoch = (((data[source] - low_data) / 
                  (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
        result = pd.DataFrame({'stoch':stoch, 
                               d_type:ma(data=stoch, length=length_d)})

        return result

    def idc_adx(self, smooth:int = 14, length_di:int = 14,
                only:bool = False, last:int = None) -> flx.DataWrapper:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Args:
            smooth (int, optional): Smoothing length. Default is 14.
            length_di (int, optional): Window length for calculating +DI and -DI. Default is 14.
            only (bool, optional): If True, returns only a Series with the ADX values.
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            DataWrapper: DataWrapper containing 'adx', '+di', and '-di' values for 
                          each step.

        Columns:
            - 'adx'
            - '+di'
            - '-di'
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
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc adx.
        return flx.DataWrapper(
            self.__idc_adx(smooth=smooth, length_di=length_di, 
                            only=only, last=last, cut=True))

    @__store_decorator
    def __idc_adx(self, data:pd.Series = None, smooth:int = 14, 
                  length_di:int = 14, only:bool = False, 
                  last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series, optional): Series of data to perform the ADX calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: DataFrame containing 'adx', '+di', and '-di' values for 
                          each step.

        Columns:
            - 'adx'
            - '+di'
            - '-di'
        """

        data = self.__data_all if data is None else data

        atr = self.__idc_atr(length=length_di, smooth='smma')

        dm_p_raw = data['High'].diff()
        dm_n_raw = -data['Low'].diff()
        
        dm_p = pd.Series(
            np.where((dm_p_raw > dm_n_raw) & (dm_p_raw > 0), dm_p_raw, 0), 
            index=data.index)
        dm_n = pd.Series(
            np.where((dm_n_raw > dm_p_raw) & (dm_n_raw > 0), dm_n_raw, 0), 
            index=data.index)

        di_p = 100 * self.__idc_smma(dm_p, length=length_di) / atr
        di_n = 100 * self.__idc_smma(dm_n, length=length_di) / atr

        adx = self.__idc_smma(
            data=100 * np.abs((di_p - di_n) / (di_p + di_n).replace(0, 1)), 
            length=smooth)

        if only: 
            return adx
        adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

        return adx

    def idc_macd(self, short_len:int = 12, long_len:int = 26, 
                 signal_len:int = 9, macd_ma_type:str = 'ema', 
                 signal_ma_type:str = 'ema', histogram:bool = True, 
                 source:str = 'Close', last:int = None) -> flx.DataWrapper:
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
            source (str, optional): Data source for calculations. Allowed values: 'Close', 
                'Open', 'High', 'Low'.
            last (int, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Returns:
            DataWrapper: A DataWrapper with MACD values and the signal line for each step.

        Columns:
            - 'macd'
            - 'signal'
            - 'histogram'      
        """

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
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc macd.
        return flx.DataWrapper(
            self.__idc_macd(short_len=short_len, long_len=long_len, 
                            signal_len=signal_len, macd_ma_type=macd_ma_type, 
                            signal_ma_type=signal_ma_type, histogram=histogram, 
                            source=source, last=last, cut=True))

    @__store_decorator
    def __idc_macd(self, data:pd.Series = None, short_len:int = 12, 
                   long_len:int = 26, signal_len:int = 9, 
                   macd_ma_type:str = 'ema', signal_ma_type:str = 'ema', 
                   histogram:bool = True, source:str = 'Close', 
                   last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Calculate the convergence/divergence of the moving average (MACD).

        This function calculates the MACD.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.Series, optional): The data used for calculation of MACD.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: A DataFrame with MACD values and signal line for each step.

        Columns:
            - 'macd'
            - 'signal'
            - 'histogram'  
        """

        data = self.__data_all if data is None else data

        match macd_ma_type:
            case 'ema':
                macd_ma = self.__idc_ema
            case 'sma':
                macd_ma = self.__idc_sma

        match signal_ma_type:
            case 'ema':
                signal_ma = self.__idc_ema
            case 'sma':
                signal_ma = self.__idc_sma
        
        short_ema = macd_ma(data=data[source], length=short_len)
        long_ema = macd_ma(data=data[source], length=long_len)
        macd = short_ema - long_ema

        signal_line = signal_ma(data=macd, length=signal_len)

        result = pd.DataFrame({'macd':macd, 'signal':signal_line, 
                               'histogram':macd-signal_line} 
                               if histogram else 
                               {'macd':macd, 'signal':signal_line})

        return result

    def idc_sqzmom(self, bb_len:int = 20, bb_mult:float = 1.5, 
                   kc_len:int = 20, kc_mult:float = 1.5, 
                   use_tr:bool = True, source:str = 'Close', 
                   last:int = None) -> flx.DataWrapper:
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
            use_tr (bool, optional): If False, ('High' - 'Low') is used instead of the true 
                range.
            source (str, optional): Data source for calculations. Allowed values: 'Close', 
                'Open', 'High', 'Low'.
            last (int, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Returns:
            DataWrapper: A DataWrapper with Squeeze Momentum values and histogram for 
                each step.

        Columns:
            - 'sqzmom'
            - 'histogram'
        """

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
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                                            'source' only one of these values: 
                                            ['Close','Open','High','Low'].
                                            """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc sqzmom.
        return flx.DataWrapper(
            self.__idc_sqzmom(bb_len=bb_len, bb_mult=bb_mult, 
                            kc_len=kc_len, kc_mult=kc_mult, 
                            use_tr=use_tr, source=source, 
                            last=last, cut=True))

    @__store_decorator
    def __idc_sqzmom(self, data:pd.Series = None, 
                     bb_len:int = 20, bb_mult:float = 1.5, 
                     kc_len:int = 20, kc_mult:float = 1.5, 
                     use_tr:bool = True, source:str = 'Close', 
                     last:int = None, cut:bool = False) -> pd.DataFrame:
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
            data (pd.Series, optional): The data used for calculating the Squeeze Momentum.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: A DataFrame with Squeeze Momentum values and histogram for 
                each step.

        Columns:
            - 'sqzmom'
            - 'histogram'
        """

        data = self.__data_all if data is None else data

        basis = self.__idc_sma(length=bb_len)
        dev = bb_mult * data[source].rolling(window=bb_len).std(ddof=0)

        upper_bb = basis + dev
        lower_bb = basis - dev

        ma = self.__idc_sma(length=kc_len)
        range_ = self.__idc_sma(data=self.__idc_trange()
                                if use_tr else data['High']-data['Low'], 
                                length=kc_len)
        
        upper_kc = ma + range_ * kc_mult
        lower_kc = ma - range_ * kc_mult

        sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

        d = data[source] - ((data['Low'].rolling(window=kc_len).min() + 
                             data['High'].rolling(window=kc_len).max()) / 2 + 
                             self.__idc_sma(length=kc_len)) / 2

        histogram = self.__idc_rlinreg(data=d, length=kc_len, offset=0)

        result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index), 
                               'histogram':pd.Series(histogram)}, 
                               index=data.index)
        return result

    @__store_decorator
    def __idc_rlinreg(self, data:pd.Series = None, 
                      length:int = 5, offset:int = 1,
                      cut:bool = False) -> np.ndarray:
        """
        Calculate rolling linear regression values.

        This function calculates the rolling linear regression.

        Note:
            This function is not very efficient. It is recommended that the length
            of the data does not exceed 50. This is a hidden function intended to
            prevent user modification and does not include exception handling.

        Args:
            data (pd.Series, optional): The data used for linear regression calculations.
            length (int, optional): Length of each window for the rolling regression.
            offset (int, optional): Offset used in the regression calculation.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            np.ndarray: Array with the linear regression values for each window.
        """
        data = self.__data if data is None else data

        x = np.arange(length)
        y = data.rolling(window=length)

        m = y.apply(lambda y: np.polyfit(x, y, 1)[0])
        b = y.mean() - (m * np.mean(x)) 

        return m * (length - 1 - offset) + b

    def idc_mom(self, length:int = 10, source:str = 'Close', 
                last:int = None) -> flx.DataWrapper:
        """
        Calculate momentum values (MOM).

        This function calculates the MOM.

        Args:
            length (int, optional): Length for calculating momentum.
            source (str, optional): Data source for momentum calculation. Allowed values:
                'Close', 'Open', 'High', 'Low'.
            last (int, optional): Number of data points to return starting from the
                present backward. If None, returns data for all available periods.

        Returns:
            DataWrapper: DataWrapper with the momentum values for each step.
        """

        if length > 5000 or length <= 0: 
            raise ValueError(utils.text_fix("""
                             'length' it has to be greater than 
                             0 and less than 5000.
                             """, newline_exclude=True))
        elif not source in ('Close','Open','High','Low'): 
            raise ValueError(utils.text_fix("""
                             'source' only one of these values: 
                             ['Close','Open','High','Low'].
                             """, newline_exclude=True))
        elif (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        # Calc momentum.
        return flx.DataWrapper(self.__idc_mom(length=length, source=source, 
                                              last=last, cut=True))

    @__store_decorator
    def __idc_mom(self, data:pd.Series = None, length:int = 10, 
                  source:str = 'Close', last:int = None,
                  cut:bool = False) -> pd.Series:
        """
        Calculate momentum values (MOM).

        This function calculates the MOM.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.Series, optional): The data used to calculate momentum.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series with the momentum values for each step.
        """

        data = self.__data_all if data is None else data
        mom = data[source] - data[source].shift(length)

        return mom

    def idc_ichimoku(self, tenkan_period:int = 9, kijun_period:int = 26, 
                     senkou_span_b_period:int = 52, ichimoku_lines:bool = True, 
                     last:int = None) -> flx.DataWrapper:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Args:
            tenkan_period (int, optional): Window length to calculate the Tenkan-sen line.
            kijun_period (int, optional): Window length to calculate the Kijun-sen line.
            senkou_span_b_period (int, optional): Window length to calculate the Senkou Span B.
            ichimoku_lines (bool, optional): If True, adds the columns 'tenkan_sen' and
                'kijun_sen' to the returned DataFrame.
            last (int, optional): Number of data points to return starting from the
                present backwards. If None, returns data for all available periods.

        Returns:
            DataWrapper: A DataWrapper with Ichimoku cloud values and optionally
                'tenkan_sen' and 'kijun_sen' columns if `ichimoku_lines` is True.

        Columns:
            - 'senkou_a'
            - 'senkou_b'
            - 'tenkan_sen'
            - 'kijun_sen'
            - 'ichimoku_lines'
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
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        
        # Calc ichimoku.
        return flx.DataWrapper(
             self.__idc_ichimoku(tenkan_period=tenkan_period, 
                                kijun_period=kijun_period, 
                                senkou_span_b_period=senkou_span_b_period, 
                                ichimoku_lines=ichimoku_lines, 
                                last=last, cut=True))

    @__store_decorator
    def __idc_ichimoku(self, data:pd.Series = None, tenkan_period:int = 9, 
                       kijun_period:int = 26, senkou_span_b_period:int = 52, 
                       ichimoku_lines:bool = True, 
                       last:int = None, cut:bool = False) -> pd.DataFrame:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.Series, optional): The data used to calculate the Ichimoku cloud values.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.DataFrame: A DataFrame with Ichimoku cloud values and optionally
                'tenkan_sen' and 'kijun_sen' columns if `ichimoku_lines` is True.

        Columns:
            - 'senkou_a'
            - 'senkou_b'
            - 'tenkan_sen'
            - 'kijun_sen'
            - 'ichimoku_lines'
        """

        data = self.__data_all if data is None else data

        tenkan_sen_val = (data['High'].rolling(window=tenkan_period).max() + 
                          data['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen_val = (data['High'].rolling(window=kijun_period).max() + 
                         data['Low'].rolling(window=kijun_period).min()) / 2

        senkou_span_a_val = ((tenkan_sen_val + kijun_sen_val) / 2)
        senkou_span_b_val = ((data['High'].rolling(
            window=senkou_span_b_period).max() + 
            data['Low'].rolling(window=senkou_span_b_period).min()) / 2)
        senkou_span = (pd.DataFrame({'senkou_a':senkou_span_a_val,
                                    'senkou_b':senkou_span_b_val, 
                                    'tenkan_sen':tenkan_sen_val,
                                    'kijun_sen':kijun_sen_val}) 
                      if ichimoku_lines else 
                        pd.DataFrame({'senkou_a':senkou_span_a_val,
                                      'senkou_b':senkou_span_b_val}))
        
        return senkou_span

    def idc_atr(self, length:int = 14, smooth:str = 'smma', 
                last:int = None) -> flx.DataWrapper:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Args:
            length (int, optional): Window length used to smooth the average true range (ATR).
            smooth (str, optional): Type of moving average used to smooth the ATR. 
            last (int, optional): Number of data points to return starting from the 
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
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        # Calc atr.
        return flx.DataWrapper(self.__idc_atr(length=length, smooth=smooth, 
                                              last=last, cut=True))

    @__store_decorator
    def __idc_atr(self, length:int = 14, smooth:str = 'smma', 
                  last:int = None, cut:bool = False) -> pd.Series:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series with the average true range values for each step.
        """

        tr = self.__idc_trange()

        match smooth:
            case 'wma':
                atr = self.__idc_wma(data=tr, length=length, last=last)
            case 'sma':
                atr = self.__idc_sma(data=tr, length=length, last=last)
            case 'ema':
                atr = self.__idc_ema(data=tr, length=length, last=last)
            case 'smma':
                atr = self.__idc_smma(data=tr, length=length, last=last)
        return atr

    @__store_decorator
    def __idc_trange(self, data:pd.Series = None, 
                     handle_na: bool = True, last:int = None,
                     cut:bool = False) -> pd.Series:
        """
        Calculate the true range.

        This function calculates the true range.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.Series, optional): The data used to perform the calculation.
            handle_na (bool, optional): Whether to handle NaN values in 'Close' as 
                per TradingView's rules.
            last (int, optional): Number of data points to return starting from the 
                present backward. If None, returns data for all available periods.
            cut (bool, optional): True to return the trimmed data with current index.

        Returns:
            pd.Series: Series with the true range values for each step.
        """

        data = self.__data_all if data is None else data

        close = data['Close'].shift(1)

        if handle_na:
                close.fillna(data['Low'], inplace=True)
                     
        hl = data['High'] - data['Low']
        hyc = abs(data['High'] - close)
        lyc = abs(data['Low'] - close)
        tr = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)

        if not handle_na:
            tr[close.isna()] = np.nan

        return tr
