"""
Strategy Module.

This module contains the main class that must be inherited to create your 
own strategy.

Classes:
    StrategyClass: This is the class you have to inherit to create your strategy.

Hidden Functions:
    _data_info: Gathers information about the dataset (hidden function).
"""

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from . import _commons as _cm
from . import flexdata as flx
from . import exception
from . import utils

def _data_info() -> tuple:
    """
    Data Info.

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
    StrategyClass.

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
        width: Data width from `main.__data_width`.
        icon: Data icon from `main.__data_icon`.
        interval: Data interval from `main.__data_interval`.

    Private Attributes:
        __init_funds: Initial funds for the strategy.
        __spread_pct: Closing and opening spread.
        __commission: Commission per trade.
        __trade: DataFrame for new trades.
        __trades_ac: DataFrame for open trades.
        __trades_cl: DataFrame for closed trades.
        __data: DataFrame containing all data.

    Methods:
        get_init_funds: Returns the initial funds for the strategy.
        get_commission: Returns the commission per trade.
        act_mod: Modifies an existing trade.
        act_close: Closes an existing trade.
        act_open: Opens a new trade.
        prev_trades_ac: Retrieves previously opened trades.
        prev_trades_cl: Retrieves previously closed trades.
        prev: Recovers all step data.
        idc_hvolume: Calculates the horizontal volume.
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
        idc_sqzmom: Calculates the Squeeze Momentum indicator.
        idc_mom: Calculates the Momentum indicator (MOM).
        idc_ichimoku: Calculates the Ichimoku indicator.
        idc_fibonacci: Calculates Fibonacci retracement levels.
        idc_atr: Calculates the Average True Range (ATR).

    Private Methods:
        __act_close: Closes an existing trade.
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
        __idc_sqzmom: Calculates the Squeeze Momentum indicator.
        __idc_mom: Calculates the Momentum indicator (MOM).
        __idc_ichimoku: Calculates the Ichimoku indicator.
        __idc_fibonacci: Calculates Fibonacci retracement levels.
        __idc_atr: Calculates the Average True Range (ATR).
        __before: This function is used to run trades and other operations.
    """

    def __init__(self, data:pd.DataFrame = pd.DataFrame(), 
                 trades_cl:pd.DataFrame = pd.DataFrame(), 
                 trades_ac:pd.DataFrame = pd.DataFrame(),
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
            spread_pct (float, optional): Spread per trade.
            commission (float, optional): Commission per trade.
            slippage_pct (float, optional): Slippage per trade.
            init_founds (int, optional): Initial funds for the strategy.
        """

        if not type(data) is pd.DataFrame: 
            raise exception.StyClassError('Data is empty.')

        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.date = None

        self.__data = pd.DataFrame()
        self.__data_updater(data=data)
        
        self.interval, self.icon, self.width = _data_info()

        self.__spread_pct  = spread_pct
        self.__slippage_pct = slippage_pct
        self.__commission = commission
        self.__init_funds = init_funds

        self.__trade = pd.DataFrame()
        self.__trades_ac = trades_ac
        self.__trades_cl = trades_cl

    @abstractmethod
    def next(self) -> None: ...

    def get_commission(self) -> float:
        """
        Get __commission.

        Returns:
            float: The value of the hidden variable `__commission`.
        """

        return self.__commission
    
    def get_init_funds(self) -> int:
        """
        Get __init_funds.

        Returns:
            float: The value of the hidden variable `__init_funds`.
        """

        return self.__init_funds
    
    def __data_updater(self, data:pd.DataFrame) -> None:
        """
        Data updater.

        Updates all data with the provided DataFrame.

        Args:
            data (pd.DataFrame): All data from the step and previous ones.
        """

        if data.empty:
            return 'Data is empty.'

        data_ = data.values[-1]
        self.open = data_[0]
        self.high = data_[1]
        self.low = data_[2]
        self.close = data_[3]
        self.volume = data_[4]
        self.date = data.index[-1]

        self.__data = data
        self.__trade = self.__trade.iloc[0:0]

    def __before(self, data=pd.DataFrame()):
        """
        Before.

        This function is used to run trades and other operations.

        Args:
            data (pd.DataFrame): Data from the current and previous steps.
        """

        self.__data_updater(data=data)

        # Check if a trade needs to be closed.
        if not self.__trades_ac.empty:
            self.__trades_ac.apply(lambda row: self.__act_close(index=row.name) 
                                if (not row['Type'] and 
                                (self.__data["Low"].iloc[-1] <= row['TakeProfit'] 
                                or self.__data["High"].iloc[-1] >= row['StopLoss']))
                                or (row['Type'] and 
                                (self.__data["High"].iloc[-1] >= row['TakeProfit'] 
                                or self.__data["Low"].iloc[-1] <= row['StopLoss'])) 
                                else None, axis=1)
        
        self.next()

        # Concat new trade.
        if not self.__trade.empty: 
            self.__trades_ac = pd.concat([self.__trades_ac, self.__trade], 
                                         ignore_index=True)

        self.__trades_ac.reset_index(drop=True, inplace=True)

        return self.__trades_ac, self.__trades_cl

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
            return flx.DataWrapper(__data.index)

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data = __data.values[
            len(__data) - last if last is not None and last < len(__data) else 0:]

        if label != None: 
            data = data[:,__data.columns.get_loc(label)]

        return flx.DataWrapper(data)

    def prev_trades_cl(self, label:str = None, last:int = None) -> flx.DataWrapper:
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
            `__trades_cl` columns, the same columns you can access with 
            `prev_trades_ac`.

            - Date: The step date when the trade began.
            - Close: The close price at the trade's start.
            - Open: The open price at the trade's start.
            - Low: The lowest price at the trade's start.
            - High: The highest price at the trade's start.
            - PositionOpen: The 'Close' price at the trade's start.
            - PositionClose: Price at which the position was closed.
            - PositionCloseNoS: Price at which the position was closed without any slipage.
            - PositionDate: The step date when the trade ends.
            - StopLoss: The stop loss position.
            - TakeProfit: The take profit position.
            - Amount: Chosen amount.
            - ProfitPer: Trade profit in percentage.
            - Profit: Trade profit based on amount.
            - Type: Type of trade.

        Returns:
            DataWrapper: DataWrapper containing the data from closed trades.
        """

        __trades_cl = self.__trades_cl
        if label == 'index': 
            return flx.DataWrapper(__trades_cl.index)
        elif __trades_cl.empty: 
            return flx.DataWrapper()

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data = __trades_cl.values[
            len(__trades_cl) - last if last is not None and last < len(__trades_cl) else 0:]
        
        if label != None: 
            data = data[:,__trades_cl.columns.get_loc(label)]

        return flx.DataWrapper(data)
    
    def prev_trades_ac(self, label:str = None, last:int = None) -> flx.DataWrapper:
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
            `__trades_ac` columns, the same columns you can access with 
            `prev_trades_cl`.

            - Date: The step date when the trade began.
            - Close: The close price at the trade's start.
            - Open: The open price at the trade's start.
            - Low: The lowest price at the trade's start.
            - High: The highest price at the trade's start.
            - PositionOpen: The 'Close' price at the trade's start.
            - PositionClose: Price at which the position was closed.
            - PositionCloseNoS: Price at which the position was closed without any slipage.
            - PositionDate: The step date when the trade ends.
            - StopLoss: The stop loss position.
            - TakeProfit: The take profit position.
            - Amount: Chosen amount.
            - ProfitPer: Trade profit in percentage.
            - Profit: Trade profit based on amount.
            - Type: Type of trade.
        
        Returns:
            DataWrapper: DataWrapper containing the data from active trades.
        """

        __trades_ac = self.__trades_ac
        if label == 'index': 
            return flx.DataWrapper(__trades_ac.index)
        elif __trades_ac.empty: 
            return flx.DataWrapper()

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
            raise ValueError(utils.text_fix("""
                            Last has to be less than the length of 
                            'data' and greater than 0.
                            """, newline_exclude=True))

        data = __trades_ac.values[
            len(__trades_ac) - last if last is not None and last < len(__trades_ac) else 0:]
    
        if label != None: 
            data = data[:,__trades_ac.columns.get_loc(label)]

        return flx.DataWrapper(data)
    
    def idc_hvolume(self, start:int = 0, end:int = None, 
                    bar:int = 10) -> flx.DataWrapper:
        """
        Indicator horizontal volume.

        This function calculates the horizontal volume.

        Alert:
            Using this function with the `end` parameter set to None is not 
            recommended and may cause slowness in the program.

        Args:
            start (int, optional): Starting point from now to capture the 
                horizontal volume. Defaults to 0.
            end (int, optional): Ending point from now to capture the horizontal 
                volume. If None, data is captured from the beginning.
            bar (int): Number of horizontal volume bars. More bars increase 
                precision.

        Returns:
            DataWrapper: DataWrapper with the position of each bar and the volume.

        Columns:
            - 'Pos'
            - 'H_Volume'
        """

        if start < 0: 
            raise ValueError("'start' must be greater or equal than 0.")
        elif end != None:
            if end < 0: 
                raise ValueError("'end' must be greater or equal than 0.")
            elif start >= end: 
                raise ValueError("'start' must be less than end.")
        if bar <= 0: 
            raise ValueError("'bar' must be greater than 0.")

        data_len = self.__data.shape[0]
        data_range = self.__data.iloc[data_len-end 
                                      if end != None and end < data_len else 0:
                                      data_len-start 
                                      if start < data_len else data_len]

        if bar == 1: 
            return pd.DataFrame({"H_Volume":data_range["Volume"].sum()}, 
                                index=[1])

        bar_list = np.array([0]*bar, dtype=np.int64) 
        result = pd.DataFrame({"Pos":(data_range["High"].max() - 
                                      data_range["Low"].min())/(bar-1) * 
                                      range(bar) + data_range["Low"].min(),
                               "H_Volume":bar_list})

        def vol_calc(row) -> None: 
            bar_list[np.argmin(
                np.abs(np.subtract(result["Pos"].values, row["Low"]))):
                np.argmin(
                    np.abs(np.subtract(result["Pos"].values, row["High"])))+1
                ] += int(row["Volume"])

        data_range.apply(vol_calc, axis=1)
        result["H_Volume"] += bar_list

        return flx.DataWrapper(result)
        
    def idc_ema(self, length:int = any, 
                source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Exponential moving average (EMA).

        This function calculates the EMA.

        Args:
            length (int): The length of the EMA.
            source (str): The data source for the EMA calculation. Allowed 
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
        return flx.DataWrapper(self.__idc_ema(length=length, source=source, last=last))

    def __idc_ema(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None) -> np.ndarray:
        """
        Exponential Moving Average (EMA).

        This function calculates the EMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the EMA calculation.

        Returns:
            np.ndarray: Array containing the EMA values for each step.
        """

        data = self.__data[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()

        return ema.iloc[len(ema)-last 
                        if last != None and last < len(ema) else 0:]

    def idc_sma(self, length:int = any, 
                source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Args:
            length (int): Length of the SMA.
            source (str): Data source for SMA calculation. Allowed values are 
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
        return flx.DataWrapper(self.__idc_sma(length=length, source=source, last=last))
    
    def __idc_sma(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None) -> np.ndarray:
        """
        Simple Moving Average (SMA).

        This function calculates the SMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the SMA calculation.

        Returns:
            np.ndarray: Array containing the SMA values for each step.
        """

        data = self.__data[source] if data is None else data
        sma = data.rolling(window=length).mean()

        return sma.iloc[len(sma)-last 
                        if last != None and last < len(sma) else 0:]
    
    def idc_wma(self, length:int = any, source:str = 'Close', 
                invt_weight:bool = False, last:int = None) -> flx.DataWrapper:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Args:
            length (int): Length of the WMA.
            source (str): Data source for WMA calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            invt_weight (bool): If True, the distribution of weights is reversed.
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
                              invt_weight=invt_weight, last=last))
    
    def __idc_wma(self, data:pd.Series = None, 
                  length:int = any, source:str = 'Close', 
                  invt_weight:bool = False, last:int = None) -> np.ndarray:
        """
        Weighted Moving Average (WMA).

        This function calculates the WMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the WMA calculation.

        Returns:
            np.ndarray: Array containing the WMA values for each step.
        """

        data = self.__data[source] if data is None else data

        weight = (np.arange(1, length+1)[::-1] 
                  if invt_weight else np.arange(1, length+1))
        wma = data.rolling(window=length).apply(
            lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return wma.iloc[len(wma)-last 
                        if last != None and last < len(wma) else 0:]
    
    def idc_smma(self, length:int = any, 
                 source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Args:
            length (int): Length of the SMMA.
            source (str): Data source for SMMA calculation. Allowed values are 
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
        return flx.DataWrapper(self.__idc_smma(length=length, source=source, last=last))
    
    def __idc_smma(self, data:pd.Series = None, length:int = any, 
                   source:str = 'Close', last:int = None) -> np.ndarray:
        """
        Smoothed Moving Average (SMMA).

        This function calculates the SMMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the SMMA calculation.

        Returns:
            np.ndarray: Array containing the SMMA values for each step.
        """

        data = self.__data[source] if data is None else data

        smma = data.ewm(alpha=1/length, adjust=False).mean()
        smma.shift(1)

        return smma.iloc[len(smma)-last 
                        if last != None and last < len(smma) else 0:]
    
    def idc_sema(self, length:int = 9, method:str = 'sma', 
                  smooth:int = 5, only:bool = False, 
                  source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Smoothed Exponential Moving Average (SEMA).

        This function calculates the SEMA.

        Args:
            length (int): Length of the EMA.
            method (str): Smoothing method. Choices include various smoothing 
                          methods.
            smooth (int): Length of the smoothing method.
            only (bool): If True, returns only a Series with the values of the 
                        'method'.
            source (str): Data source for EMA calculation. Allowed values are 
                          ('Close', 'Open', 'High', 'Low', 'Volume').
            last (int, optional): Number of data points to return from the present 
                                  backwards. If None, returns data for all times.

        Returns:
            flx.DataWrapper: DataWrapper containing the 'ema' and 'smoothed' values for 
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
                                only=only, source=source, last=last))
    
    def __idc_sema(self, data:pd.Series = None, length:int = 9, 
                    method:str = 'sma', smooth:int = 5, only:bool = False, 
                    source:str = 'Close', last:int = None) -> pd.DataFrame:
        """
        Smoothed Exponential Moving Average (SEMA).

        This function calculates the SEMA.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the SEMA calculation.

        Returns:
            pd.DataFrame: DataFrame containing 'ema' and 'smoothed' values for 
                          each step.

        Columns:
            - 'ema'
            - 'smoothed'
        """

        data = self.__data[source] if data is None else data
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

        return smema.apply(
            lambda col: col.iloc[len(smema.index)-last 
                                 if last != None and 
                                 last < len(smema.index) else 0:], axis=0)

    def idc_bb(self, length:int = 20, std_dev:float = 2, ma_type:str = 'sma', 
               source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Args:
            length (int): Window length for calculating Bollinger Bands.
            std_dev (float): Number of standard deviations for the bands.
            ma_type (str): Type of moving average. For example, 'sma' for simple 
                          moving average.
            source (str): Data source for calculation. Allowed values are 
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
                                ma_type=ma_type, source=source, last=last))
    
    def __idc_bb(self, data:pd.Series = None, length:int = 20, 
                 std_dev:float = 2, ma_type:str = 'sma', 
                 source:str = 'Close', last:int = None) -> pd.DataFrame:
        """
        Bollinger Bands (BB).

        This function calculates the BB.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the Bollinger Bands 
                              calculation.

        Returns:
            pd.DataFrame: DataFrame containing 'Upper', '{ma_type}', and 'Lower' 
                          values for each step.
              
        Columns:
            - 'Upper'
            - '{ma_type}'
            - 'Lower'
        """

        data = self.__data[source] if data is None else data

        match ma_type:
            case 'sma': ma = self.__idc_sma(data=data, length=length)
            case 'ema': ma = self.__idc_ema(data=data, length=length)
            case 'wma': ma = self.__idc_wma(data=data, length=length)
            case 'smma': ma = self.__idc_smma(data=data, length=length)
        std_ = (std_dev * data.rolling(window=length).std())
        bb = pd.DataFrame({'Upper':ma + std_,
                           ma_type:ma,
                           'Lower':ma - std_}, index=ma.index)

        return bb.apply(lambda col: col.iloc[len(bb.index)-last 
                                             if last != None and 
                                             last < len(bb.index) else 0:], 
                                             axis=0)

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, 
                rsi_ma_type:str = 'smma', base_type:str = 'sma', 
                bb_std_dev:float = 2, source:str = 'Close', 
                last:int = None) -> flx.DataWrapper:
        """
        Relative Strength Index (RSI).

        This function calculates the RSI.

        Args:
            length_rsi (int): Window length for the RSI calculation using 
                              `rsi_ma_type`. Default is 14.
            length (int): Window length for the moving average applied to RSI. 
                          Default is 14.
            rsi_ma_type (str): Type of moving average used for calculating RSI. 
                              For example, 'wma' for weighted moving average.
            base_type (str): Type of moving average applied to RSI. For example, 
                            'sma' for simple moving average.
            bb_std_dev (float): Standard deviation for Bollinger Bands. Default is 2.
            source (str): Data source for calculation. Allowed values are 
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
                              bb_std_dev=bb_std_dev, source=source, last=last))

    def __idc_rsi(self, data:pd.Series = None, length_rsi:int = 14, 
                  length:int = 14, rsi_ma_type:str = 'wma', 
                  base_type:str = 'sma', bb_std_dev:float = 2, 
                  source:str = 'Close', last:int = None)  -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        This function calculates the RSI.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the RSI calculation.

        Returns:
            pd.DataFrame: DataFrame containing 'rsi' and '{base_type}' values for 
                          each step.

        Columns:
            - 'rsi'
            - '{base_type}'
        """

        delta = self.__data[source].diff() if data is None else data.diff()

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

        return rsi.apply(
            lambda col: col.iloc[len(rsi.index)-last 
                                 if last != None and 
                                 last < len(rsi.index) else 0:], axis=0)

    def idc_stochastic(self, length_k:int = 14, smooth_k:int = 1, 
                       length_d:int = 3, d_type:str = 'sma', 
                       source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Stochastic Oscillator.

        This function calculates the stochastic oscillator.

        Args:
            length_k (int): Window length for calculating the stochastic values.
            smooth_k (int): Smoothing window length for the stochastic values.
            length_d (int): Window length for the moving average applied to 
                            the stochastic values.
            d_type (str): Type of moving average used for the stochastic oscillator. 
                          For example, 'sma' for simple moving average.
            source (str): Data source for calculation. Allowed values are 
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
                                source=source, last=last))

    def __idc_stochastic(self, data:pd.Series = None, length_k:int = 14, 
                         smooth_k:int = 1, length_d:int = 3, d_type:int = 'sma', 
                         source:str = 'Close', last:int = None) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        This function calculates the stochastic oscillator.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the stochastic calculation.

        Returns:
            pd.DataFrame: DataFrame containing 'stoch' and '{d_type}' values for each 
                          step.

        Columns:
            - 'stoch'
            - '{d_type}'
        """

        data = self.__data if data is None else data

        low_data = self.__data['Low'].rolling(window=length_k).min()
        high_data = self.__data['High'].rolling(window=length_k).max()

        match d_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        stoch = (((self.__data[source] - low_data) / 
                  (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
        result = pd.DataFrame({'stoch':stoch, 
                               d_type:ma(data=stoch, length=length_d)})

        return result.apply(
            lambda col: col.iloc[len(result.index)-last 
                                 if last != None and 
                                 last < len(result.index) else 0:], axis=0) 

    def idc_adx(self, smooth:int = 14, length_di:int = 14, 
                only:bool = False, last:int = None) -> flx.DataWrapper:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Args:
            smooth (int): Smoothing length. Default is 14.
            length_di (int): Window length for calculating +DI and -DI. Default is 14.
            only (bool): If True, returns only a Series with the ADX values.
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
                            only=only, last=last))

    def __idc_adx(self, data:pd.Series = None, smooth:int = 14, 
                  length_di:int = 14, only:bool = False, 
                  last:int = None) -> pd.DataFrame:
        """
        Average Directional Index (ADX).

        This function calculates the ADX.

        Note:
            This function is hidden to prevent user modification and does not 
            include exception handling.

        Args:
            data (pd.Series): Series of data to perform the ADX calculation.

        Returns:
            pd.DataFrame: DataFrame containing 'adx', '+di', and '-di' values for 
                          each step.

        Columns:
            - 'adx'
            - '+di'
            - '-di'
        """

        data = self.__data if data is None else data

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
            return adx.iloc[len(adx)-last 
                            if last != None and last < len(adx) else 0:]
        adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

        return adx.apply(
            lambda col: col.iloc[len(adx.index)-last 
                                 if last != None and 
                                 last < len(adx.index) else 0:], axis=0) 

    def idc_macd(self, short_len:int = 12, long_len:int = 26, 
                 signal_len:int = 9, macd_ma_type:str = 'ema', 
                 signal_ma_type:str = 'ema', histogram:bool = True, 
                 source:str = 'Close', last:int = None) -> flx.DataWrapper:
        """
        Calculate the convergence/divergence of the moving average (MACD).

        This function calculates the MACD.

        Args:
            short_len (int): Length of the short moving average used to calculate MACD.
            long_len (int): Length of the long moving average used to calculate MACD.
            signal_len (int): Length of the moving average for the MACD signal line.
            macd_ma_type (str): Type of moving average used to calculate MACD.
            signal_ma_type (str): Type of moving average used to smooth the MACD.
            histogram (bool): If True, includes an additional 'histogram' column.
            source (str): Data source for calculations. Allowed values: 'Close', 
                'Open', 'High', 'Low'.
            last (int or None): Number of data points to return starting from the
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
                            source=source, last=last))

    def __idc_macd(self, data:pd.Series = None, short_len:int = 12, 
                   long_len:int = 26, signal_len:int = 9, 
                   macd_ma_type:str = 'ema', signal_ma_type:str = 'ema', 
                   histogram:bool = True, source:str = 'Close', 
                   last:int = None) -> pd.DataFrame:
        """
        Calculate the convergence/divergence of the moving average (MACD).

        This function calculates the MACD.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.DataFrame): The data used for calculation of MACD.

        Returns:
            pd.DataFrame: A DataFrame with MACD values and signal line for each step.

        Columns:
            - 'macd'
            - 'signal'
            - 'histogram'  
        """

        data = self.__data if data is None else data

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

        return result.apply(
            lambda col: col.iloc[len(result.index)-last 
                                 if last != None and 
                                 last < len(result.index) else 0:], axis=0) 

    def idc_sqzmom(self, bb_len:int = 20, bb_mult:float = 1.5, 
                   kc_len:int = 20, kc_mult:float = 1.5, 
                   use_tr:bool = True, histogram_len:int = 50, 
                   source:str = 'Close', last:int = None) -> flx.DataWrapper:
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
            bb_len (int): Bollinger band length.
            bb_mult (float): Bollinger band standard deviation.
            kc_len (int): Keltner channel length.
            kc_mult (float): Keltner channel standard deviation.
            use_tr (bool): If False, ('High' - 'Low') is used instead of the true 
                range.
            histogram_len (int): Number of steps from the present backward to calculate
                the histogram. If 0, the 'histogram' column will not be returned.
            source (str): Data source for calculations. Allowed values: 'Close', 
                'Open', 'High', 'Low'.
            last (int or None): Number of data points to return starting from the
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
        elif histogram_len < 0: 
            raise ValueError(utils.text_fix("""
                                            'histogram_len' has to be greater 
                                            or equal than 0.
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
                            use_tr=use_tr, histogram_len=histogram_len, 
                            source=source, last=last))

    def __idc_sqzmom(self, data:pd.Series = None, 
                     bb_len:int = 20, bb_mult:float = 1.5, 
                     kc_len:int = 20, kc_mult:float = 1.5, 
                     use_tr:bool = True, histogram_len:int = 50, 
                     source:str = 'Close', last:int = None) -> pd.DataFrame:
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
            data (pd.DataFrame): The data used for calculating the Squeeze Momentum.

        Returns:
            pd.DataFrame: A DataFrame with Squeeze Momentum values and histogram for 
                each step.

        Columns:
            - 'sqzmom'
            - 'histogram'
        """

        data = self.__data if data is None else data

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

        if histogram_len < 1: 
            result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index)})
            return result.apply(
                lambda col: col.iloc[len(result.index)-last 
                                     if last != None and 
                                     last < len(result.index) else 0:], axis=0)
        elif last and histogram_len > last: histogram = last
        
        histogram_len += kc_len
        d = data[source] - ((data['Low'].rolling(window=kc_len).min() + 
                             data['High'].rolling(window=kc_len).max()) / 2 + 
                             self.__idc_sma(length=kc_len)) / 2

        histogram = self.__idc_rlinreg(data=d.iloc[len(d.index)-histogram_len 
                                              if len(d.index) > histogram_len 
                                              else 0:], length=kc_len, offset=0)

        result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index), 
                               'histogram':pd.Series(histogram)}, 
                               index=data.index)
        return result.apply(
            lambda col: col.iloc[len(result.index)-last 
                                 if last != None and 
                                 last < len(result.index) else 0:], axis=0) 

    def __idc_rlinreg(self, data:pd.Series = None, 
                      length:int = 5, offset:int = 1) -> np.ndarray:
        """
        Calculate rolling linear regression values.

        This function calculates the rolling linear regression.

        Note:
            This function is not very efficient. It is recommended that the length
            of the data does not exceed 50. This is a hidden function intended to
            prevent user modification and does not include exception handling.

        Args:
            data (pd.Series): The data used for linear regression calculations.
            length (int): Length of each window for the rolling regression.
            offset (int): Offset used in the regression calculation.

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
            length (int): Length for calculating momentum.
            source (str): Data source for momentum calculation. Allowed values:
                'Close', 'Open', 'High', 'Low'.
            last (int or None): Number of data points to return starting from the
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
        return flx.DataWrapper(self.__idc_mom(length=length, source=source, last=last))

    def __idc_mom(self, data:pd.Series = None, length:int = 10, 
                  source:str = 'Close', last:int = None) -> np.array:
        """
        Calculate momentum values (MOM).

        This function calculates the MOM.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.DataFrame): The data used to calculate momentum.
            length (int): The length of the momentum calculation.

        Returns:
            np.array: Array with the momentum values for each step.
        """

        data = self.__data if data is None else data
        mom = data[source] - data[source].shift(length)

        return mom.iloc[len(mom)-last 
                        if last != None and last < len(mom) else 0:]

    def idc_ichimoku(self, tenkan_period:int = 9, kijun_period=26, 
                     senkou_span_b_period=52, ichimoku_lines:bool = True, 
                     last:int = None) -> flx.DataWrapper:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Args:
            tenkan_period (int): Window length to calculate the Tenkan-sen line.
            kijun_period (int): Window length to calculate the Kijun-sen line.
            senkou_span_b_period (int): Window length to calculate the Senkou Span B.
            ichimoku_lines (str): If set, adds the columns 'tenkan_sen' and 
                'kijun_sen' to the DataFrame. Possible values: 'ema', 'sma', etc.
            last (int or None): Number of data points to return starting from the
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
                                last=last))

    def __idc_ichimoku(self, data:pd.Series = None, tenkan_period:int = 9, 
                       kijun_period=26, senkou_span_b_period=52, 
                       ichimoku_lines:bool = True, 
                       last:int = None) -> pd.DataFrame:
        """
        Calculate Ichimoku cloud values.

        This function calculates the Ichimoku cloud.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.DataFrame): The data used to calculate the Ichimoku cloud values.
            ichimoku_lines (bool): If True, adds the columns 'tenkan_sen' and
                'kijun_sen' to the returned DataFrame.

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

        data = self.__data if data is None else data

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
        
        return senkou_span.apply(
            lambda col: col.iloc[len(senkou_span.index)-last 
                                 if last != None and 
                                 last < len(senkou_span.index) else 0:], axis=0)

    def idc_fibonacci(self, start:int = None, end:int = 30, 
                      met:bool = False, source:str = 'Low/High') -> flx.DataWrapper:
        """
        Calculate Fibonacci retracement levels.

        This function calculates the Fibonacci retracement levels.

        Args:
            start (int or None): The number of candles back to set level 0. If
                `met` is False, `start` specifies the number of candles back to
                open the Fibonacci levels. If None, level 0 is set at the most
                recent data (equivalent to 0).
            end (int): The number of candles back to set level 1. If `met` is False,
                `end` specifies the number of candles back to close the Fibonacci levels.
                If None, level 1 is set at the most recent data (equivalent to 0).
            met (bool): If False, `start` and `end` are the number of candles backward
                from the current position. If True, `start` and `end` are the values
                from which the Fibonacci levels are calculated.
            source (str): Data source for the Fibonacci levels. Format: 's/s' where 's'
                is each source. Supported values: 'Close', 'Open', 'High', 'Low'. If
                `met` is True, this parameter is ignored.

        Returns:
            DataWrapper: A DataWrapper with Fibonacci levels and their corresponding
                values.

        Columns:
            - 'Level'
            - 'Value'
        """

        if met: return self.__idc_fibonacci(start=start, end=end)

        source = source.split('/')
        if (len(source) != 2 or source[0] == '' or 
            source[1] == '' or not source[0] in ('Close', 'Open', 'High', 'Low')
            or not source[1] in ('Close', 'Open', 'High', 'Low')): 

            raise ValueError(utils.text_fix("""
                            'source' it has to be in this format: 's/s' where 
                             's' is each source.
                            Values supported in source: 
                              ('Close', 'Open', 'High', 'Low')
                             """, newline_exclude=True))
        data_start = self.__data[source[1]]; data_end = self.__data[source[0]]

        # Fibonacci calc.
        return flx.DataWrapper(
            self.__idc_fibonacci(
            start=data_start.iloc[len(data_start)-start-1 
                                  if start != None and 
                                  start < len(data_start) else 0], 
            end=data_end.iloc[len(data_end)-end-1 
                              if end != None and end < len(data_end) else 0]))

    def __idc_fibonacci(self, start:int = 10, end:int = 1) -> pd.DataFrame:
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
                             'Value':(start + (end - start) * fibo_levels)})

    def idc_atr(self, length:int = 14, smooth:str = 'smma', 
                last:int = None) -> flx.DataWrapper:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Args:
            length (int): Window length used to smooth the average true range (ATR).
            smooth (str): Type of moving average used to smooth the ATR. 
            last (int or None): Number of data points to return starting from the 
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
        return flx.DataWrapper(self.__idc_atr(length=length, smooth=smooth, last=last))
    
    def __idc_atr(self, length:int = 14, smooth:str = 'smma', 
                  last:int = None) -> np.ndarray:
        """
        Calculate the average true range (ATR).

        This function calculates the ATR.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.DataFrame): The data used to perform the calculation of true range.

        Returns:
            np.ndarray: Array with the average true range values for each step.
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

    def __idc_trange(self, data:pd.Series = None, 
                     handle_na: bool = True, last:int = None) -> np.ndarray:
        """
        Calculate the true range.

        This function calculates the true range.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.

        Args:
            data (pd.DataFrame): The data used to perform the calculation.
            handle_na (bool): Whether to handle NaN values in 'Close' as per TradingView's rules.

        Returns:
            np.ndarray: Array with the true range values for each step.
        """

        data = self.__data if data is None else data

        close = data['Close'].shift(1)

        if handle_na:
                close.fillna(data['Low'], inplace=True)
                     
        hl = data['High'] - data['Low']
        hyc = abs(data['High'] - close)
        lyc = abs(data['Low'] - close)
        tr = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)

        if not handle_na:
            tr[close.isna()] = np.nan

        return tr.iloc[len(tr)-last 
                       if last != None and last < len(tr) else 0:]

    def act_open(self, type_:bool = 1, stop_loss:int = np.nan, 
                 take_profit:int = np.nan, amount:int = np.nan) -> None:
        """
        Opens an action for trading.

        This function opens a long or short position. 

        Warning:
            If you leave your position without 'stop loss' and 'takeprofit', 
            your trade will be counted as closed, and you can't modify or close it.

        Args:
            type_ (bool): 0 for sell, 1 for buy. Other values Python evaluates 
                        as booleans are supported.
            stop_loss (int): Price for stop loss. If np.nan or None, no stop loss 
                            will be set.
            take_profit (int): Price for take profit. If np.nan or None, no take 
                              profit will be set.
            amount (int): Amount of points for the trade.
        """

        # Convert to boolean.
        type_ = int(bool(type_))

        # Check if 'stop_loss' or 'take_profit' is None.
        stop_loss = stop_loss or np.nan
        take_profit = take_profit or np.nan

        # Check exceptions.
        if not type_ in {1,0}: 
            raise exception.ActionError("'type_' only 1 or 0.")
        elif amount < 0: 
            raise exception.ActionError(
                "'amount' can only be a positive number.")
        elif ((type_ and (self.__data["Close"].iloc[-1] <= stop_loss or 
                       self.__data["Close"].iloc[-1] >= take_profit)) or 
            (not type_ and (self.__data["Close"].iloc[-1] >= stop_loss or 
                           self.__data["Close"].iloc[-1] <= take_profit))): 

            raise exception.ActionError(
                utils.text_fix("""
                               'stop_loss' or 'take_profit' 
                               incorrectly configured for the position type_.
                               """, newline_exclude=True))

        # Spread and slipage calc.
        spread = self.__data["Close"].iloc[-1]*(self.__spread_pct/100/2)
        slippage = self.__data["Close"].iloc[-1]*(self.__slippage_pct/100)
        position_open = (self.__data["Close"].iloc[-1]+spread+slippage
                         if type_ else self.__data["Close"].iloc[-1]-spread-slippage)

        # Create new trade.
        self.__trade = pd.DataFrame({'Date':self.__data.index[-1],
                                     'Close':self.__data["Close"].iloc[-1],
                                     'Open':self.__data["Open"].iloc[-1],
                                     'Low':self.__data["Low"].iloc[-1],
                                     'High':self.__data["High"].iloc[-1],
                                     'PositionOpen':position_open,
                                     'PositionClose':np.nan,
                                     'PositionCloseNoS':np.nan,
                                     'PositionDate':np.nan,
                                     'StopLoss':stop_loss,
                                     'TakeProfit':take_profit,
                                     'Amount':amount,
                                     'ProfitPer':np.nan,
                                     'Profit':np.nan,
                                     'Type':type_},index=[1])

    def act_close(self, index:int = 0) -> None:
        """
        Close an active trade.

        Args:
            index (int): The index of the active trade you want to close.
        """

        # Check exceptions.
        if self.__trades_ac.empty: 
            raise exception.ActionError('There are no active trades.')
        elif not index in self.__trades_ac.index.to_list(): 
            raise exception.ActionError('Index does not exist.')
        # Close action.
        return self.__act_close(index=index)

    def __act_close(self, index:int = 0) -> None:
        """
        Close an active trade.

        Note:
            This is a hidden function intended to prevent user modification.
            It does not include exception handling.
        """

        # Get trade to close.
        trade = self.__trades_ac.iloc[lambda x: x.index==index].copy()
        self.__trades_ac = self.__trades_ac.drop(trade.index)
        # Get PositionClose
        take = trade['TakeProfit'].iloc[0]
        stop = trade['StopLoss'].iloc[0]

        position_close = ((stop if self.__data["Low"].iloc[-1] <= stop else take
                           if self.__data["High"].iloc[-1] >= take 
                           else self.__data["Close"].iloc[-1]) 
                           if trade['Type'].iloc[0] 
                           else (stop if self.__data["High"].iloc[-1] >= stop 
                                 else take 
                                 if self.__data["Low"].iloc[-1] <= take 
                                 else self.__data["Close"].iloc[-1]))

        # Spread and slipage calc.
        spread = self.__data["Close"].iloc[-1]*(self.__spread_pct/100/2)
        slippage = self.__data["Close"].iloc[-1]*(self.__slippage_pct/100)
        position_close_spread = (position_close-spread-slippage 
                                 if trade['Type'].iloc[0] else position_close+spread+slippage)

        # Fill data.
        trade['PositionCloseNoS'] = position_close
        trade['PositionClose'] = position_close_spread
        trade['PositionDate'] = self.__data.index[-1]
        open = trade['PositionOpen'].iloc[0]
        trade['ProfitPer'] = ((position_close_spread-open)/open*100 
                              if trade['Type'].iloc[0] 
                              else (open-position_close_spread)/open*100)
        trade['Profit'] = (trade['Amount'].iloc[0]*trade['ProfitPer'].iloc[0]
                           /100-trade['Amount']*(self.__commission/100) 
                           if not np.isnan(trade['Amount'].iloc[0]) else np.nan)

        self.__trades_cl = pd.concat([self.__trades_cl,trade], 
                                     ignore_index=True) 
        self.__trades_cl.reset_index(drop=True, inplace=True)

    def act_mod(self, index:int = 0, new_stop:int = None, 
                new_take:int = None) -> None:
        """
        Modify an active trade.

        Alerts:
            If an invalid stop loss or take profit is provided, the program will
            return None and will not execute any changes.

        Args:
            index (int): The index of the active trade to modify.
            new_stop (int or None): New stop loss price. If None, stop loss will
                not be modified. If np.nan, stop loss will be removed.
            new_take (int or None): New take profit price. If None, take profit 
                will not be modified. If np.nan, take profit will be removed.
        """
        
        # Check exceptions.
        if self.__trades_ac.empty: 
            raise exception.ActionError('There are no active trades.')
        elif not (new_stop or new_take): 
            raise exception.ActionError('Nothing was changed.')
        # Get trade to modify.
        trade = self.__trades_ac.loc[index]
        # Set new stop.
        if new_stop and ((new_stop < self.__data["Close"].iloc[-1] and 
                          trade['Type']) or (not trade['Type'] and 
                                             new_stop > self.close) or 
                                             np.isnan(new_stop)): 
            self.__trades_ac.loc[index, 'StopLoss'] = new_stop 
        # Set new take.
        if new_take and ((new_take > self.__data["Close"].iloc[-1] 
                          and trade['Type']) or (not trade['Type'] and 
                                                 new_take < self.close) or 
                                                 np.isnan(new_take)): 
            self.__trades_ac.loc[index,'TakeProfit'] = new_take
