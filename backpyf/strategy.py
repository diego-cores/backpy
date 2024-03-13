"""
Strategy.
----
Here is the main class that has to be inherited in order to create your own strategy.\n
Class:
---
>>> StrategyClass
"""

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from . import exception

class StrategyClass(ABC):
    """
    StrategyClass.
    ----
    This is the class you have to inherit to create your strategy.\n
    Example:
    --
    >>> class Strategy(backpy.StrategyClass)

    To use the functions use the self instance.\n
    Create your strategy within the StrategyClass.next() structure.\n
    Functions:
    ---
    >>> next
    >>> prev
    >>> prev_trades_cl
    >>> prev_trades_ac
    >>> idc_ema
    >>> idc_hvolume
    >>> act_open
    >>> act_close
    >>> act_mod

    Hidden Functions:
    --
    >>> __before
    >>> __idc_ema
    >>> __idc_sma
    >>> __idc_wma
    >>> __act_close
    """ 
    def __init__(self, data:pd.DataFrame = any, trades_cl:pd.DataFrame = pd.DataFrame(), trades_ac:pd.DataFrame = pd.DataFrame(), commission:int = 0, init_funds:int = 0) -> None: 
        """
        __init__
        ----
        Builder.\n
        Parameters:
        --
        >>> data:pd.DataFrame = any
        >>> trades_cl:pd.DataFrame = pd.DataFrame()
        >>> trades_ac:pd.DataFrame = pd.DataFrame()
        \n
        data: \n
        \tAll data from the step and previous ones.\n
        trades_cl: \n
        \tClosed trades.\n
        trades_ac: \n
        \tOpen trades.\n
        commission: \n
        \Commission per trade.\n
        Variables:
        --
        >>> self.open = data["Open"].iloc[-1]
        >>> self.high = data["High"].iloc[-1]
        >>> self.low = data["Low"].iloc[-1]
        >>> self.close = data["Close"].iloc[-1]
        >>> self.volume = data["Volume"].iloc[-1]
        >>> self.date = data.index[-1]

        Hidden variables:
        --
        >>> self.__init_funds = init_funds
        >>> self.__commission = commission
        >>> self.__trade = pd.DataFrame() # New trade
        >>> self.__trades_ac = trades_ac
        >>> self.__trades_cl = trades_cl
        >>> self.__data = data
        """
        if data.empty: raise exception.StyClassError('Data is empty.')

        self.open = data["Open"].iloc[-1]
        self.high = data["High"].iloc[-1]
        self.low = data["Low"].iloc[-1]
        self.close = data["Close"].iloc[-1]
        self.volume = data["Volume"].iloc[-1]
        self.date = data.index[-1]

        self.__commission = commission
        self.__init_funds = init_funds

        self.__trade = pd.DataFrame()
        self.__trades_ac = trades_ac
        self.__trades_cl = trades_cl

        self.__data = data

    @abstractmethod
    def next(self) -> None: pass

    def __before(self):
        """
        Before.
        ----
        This function is used to run trades and other things.
        """
        self.next()

        # Check if a trade needs to be closed.
        self.__trades_ac.apply(lambda row: self.__act_close(index=row.name) if self.__data["High"].iloc[-1] >= max(row['TakeProfit'],row['StopLoss']) or self.__data["Low"].iloc[-1] <= min(row['TakeProfit'],row['StopLoss']) else None, axis=1) 

        # Concat new trade.
        if not self.__trade.empty and np.isnan(self.__trade['StopLoss'].iloc[0]) and np.isnan(self.__trade['TakeProfit'].iloc[0]): 
            self.__trades_cl = pd.concat([self.__trades_cl, self.__trade], ignore_index=True); self.__trades_cl.reset_index(drop=True, inplace=True)
        elif not self.__trade.empty: self.__trades_ac = pd.concat([self.__trades_ac, self.__trade], ignore_index=True)

        self.__trades_ac.reset_index(drop=True, inplace=True)
        return self.__trades_ac, self.__trades_cl

    def prev(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev.
        ----
        Returns the data from the previous steps.\n
        Columns: 'Open','High','Low','Close','Volume'.\n
        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        \n
        label: \n
        \tData column, if you leave it at None all columns will be returned.\n
        \tIf you leave 'index', all indexes will be returned, ignoring the last parameter.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        __data = self.__data
        if label == 'index': return __data.index
        elif label != None: __data = __data[label]

        if last != None:
            if last <= 0 or last > __data.shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")
        
        return __data.iloc[len(__data)-last if last != None and last < len(__data) else 0:]
    
    def prev_trades_cl(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades closed.
        ----
        Returns the data from the closed trades.\n
        Columns: 'Date','Close','Low','High','StopLoss','TakeProfit','PositionClose','PositionDate','Amount','ProfitPer','Profit','Type'.\n
        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        \n
        label: \n
        \tData column, if you leave it at None all columns will be returned.\n
        \tIf you leave 'index', all indexes will be returned, ignoring the last parameter.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        __trades_cl = self.__trades_cl

        if label == 'index': return __trades_cl.index
        elif __trades_cl.empty: return pd.DataFrame()
        elif label != None: __trades_cl = __trades_cl[label]

        if last != None: 
            if last <= 0 or last > __trades_cl.shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        return __trades_cl.iloc[len(__trades_cl)-last if last != None and last < len(__trades_cl) else 0:] 
    
    def prev_trades_ac(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades active.
        ----
        Returns the data from the active trades.\n
        Columns: 'Date','Close','Low','High','StopLoss','TakeProfit','Amount','Type'.\n
        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        \n
        label: \n
        \tData column, if you leave it at None all columns will be returned.\n
        \tIf you leave 'index', all indexes will be returned, ignoring the 'last' parameter.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        __trades_ac = self.__trades_ac
        if label == 'index': return __trades_ac.index
        elif __trades_ac.empty: return pd.DataFrame()
        elif label != None: __trades_ac = __trades_ac[label]

        if last != None:
            if last <= 0 or last > __trades_ac.shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        return __trades_ac.iloc[len(__trades_ac)-last if last != None and last < len(__trades_ac) else 0:]
    
    def idc_hvolume(self, start:int = 0, end:int = None, bar:int = 10) -> pd.DataFrame:
        """
        Indicator horizontal volume.
        ----
        Return a pd.DataFrame with the position of each bar and the volume.\n
        Columns: 'Pos','H_Volume'.\n
        Alert:
        --
        Using this function with the 'end' parameter set to None\n
        is not recommended and may cause slowness in the program.\n
        Parameters:
        --
        >>> start:int = 0
        >>> end:int = None
        >>> bar:int = 10
        \n
        start: \n
        \tCounting from now onwards, when you want the data capture to start to return the horizontal volume.\n
        end: \n
        \tCounting from now onwards, when you want the data capture to end to return the horizontal volume.\n
        \tIf left at None the data will be captured from the beginning.\n
        bar: \n
        \tThe number of horizontal volume bars (the more bars, the more precise).\n
        """
        if start < 0: raise ValueError("'start' must be greater or equal than 0.")
        elif end != None:
            if end < 0: raise ValueError("'end' must be greater or equal than 0.")
            elif start >= end: raise ValueError("'start' must be less than end.")
        if bar <= 0: raise ValueError("'bar' must be greater than 0.")

        data_len = self.__data.shape[0]; data_range = self.__data.iloc[data_len-end if end != None and end < data_len else 0:data_len-start if start < data_len else data_len]

        if bar == 1: return pd.DataFrame({"H_Volume":data_range["Volume"].sum()}, index=[1])

        bar_list = np.array([0]*bar, dtype=np.int64)
        result = pd.DataFrame({"Pos":(data_range["High"].max()-data_range["Low"].min())/(bar-1) * range(bar) + data_range["Low"].min(),"H_Volume":bar_list})

        def vol_calc(row) -> None: bar_list[np.argmin(np.abs(np.subtract(result["Pos"].values, row["Low"]))):np.argmin(np.abs(np.subtract(result["Pos"].values, row["High"])))+1] += int(row["Volume"])
        data_range.apply(vol_calc, axis=1); result["H_Volume"] += bar_list

        return result
        
    def idc_ema(self, length:int = any, source:str = 'Close', last:int = None) -> np.array:
        """
        Exponential moving average.
        ----
        Returns an pd.Series with all the steps of an ema with the length you indicate.\n
        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tEma length.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Ema calc.
        return self.__idc_ema(length=length, source=source, last=last)

    def __idc_ema(self, data:pd.Series = None, length:int = any, source:str = 'Close', last:int = None) -> np.array:
        """
        Exponential moving average.
        ----
        Returns an pd.Series with all the steps of an ema with the length you indicate.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ma with your own data.\n
        """
        data = self.__data[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()
    
        return np.flip(ema[len(ema)-last if last != None and last < len(ema) else 0:])
    
    def idc_sma(self, length:int = any, source:str = 'Close', last:int = None):
        """
        Simple moving average.
        ----
        Return an pd.Series with all the steps of an sma with the length you indicate.\n
        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tSma length.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Sma calc.
        return self.__idc_sma(length=length, source=source, last=last)
    
    def __idc_sma(self, data:pd.Series = None, length:int = any, source:str = 'Close', last:int = None):
        """
        Simple moving average.
        ----
        Return an pd.Series with all the steps of an sma with the length you indicate.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ma with your own data.\n
        """
        data = self.__data[source] if data is None else data
        sma = data.rolling(window=length).mean()

        return np.flip(sma[len(sma)-last if last != None and last < len(sma) else 0:])
    
    def idc_wma(self, length:int = any, source:str = 'Close', invt_weight:bool = False, last:int = None):
        """
        Linear weighted moving average.
        ----
        Return an pd.Series with all the steps of an wma with the length you indicate.\n
        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> invt_weight:bool = False
        >>> last:int = None
        \n
        length: \n
        \Wma length.\n
        source: \n
        \tData.\n
        invt_weight: \n
        \t\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Wma calc.
        return self.__idc_wma(length=length, source=source, invt_weight=invt_weight, last=last)
    
    def __idc_wma(self, data:pd.Series = None, length:int = any, source:str = 'Close', invt_weight:bool = False, last:int = None):
        """
        Linear weighted moving average.
        ----
        Return an pd.Series with all the steps of an wma with the length you indicate.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ma with your own data.\n
        """
        data = self.__data[source] if data is None else data

        weight = np.arange(1, length+1)[::-1] if invt_weight else np.arange(1, length+1)
        wma = data.rolling(window=length).apply(lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return np.flip(wma[len(wma)-last if last != None and last < len(wma) else 0:])
    
    def idc_bb(self, length:int = 20, std_dev:float = 2, ma_type:str = 'sma', source:str = 'Close', last:int = None):
        """
        Bollinger bands.
        ----
        Return an pd.DataFrame with the value of the upper band, the base ma and the position of the lower band for each step.\n
        Columns: 'Upper','{ma_type}','Lower'.\n
        Parameters:
        --
        >>> length:int = any
        >>> std_dev:float = 2
        >>> ma_type:str = 'sma'
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tWindow length.\n
        std_dev: \n
        \tStandard deviation.\n
        ma_type: \n
        \tMa type.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif std_dev > 50 or std_dev < 0.001: raise ValueError("'std_dev' it has to be greater than 0.001 and less than 50.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif not ma_type in ('sma','ema','wma'): raise ValueError("'ma_type' only these values: 'sma', 'ema', 'wma'.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Bb calc.
        return self.__idc_bb(length=length, std_dev=std_dev, ma_type=ma_type, source=source, last=last)
    
    def __idc_bb(self, data:pd.Series = None, length:int = 20, std_dev:float = 2, ma_type:str = 'sma', source:str = 'Close', last:int = None):
        """
        Bollinger bands.
        ----
        Return an pd.DataFrame with the value of the upper band, the base ma and the position of the lower band for each step.\n
        Columns: 'Upper','Base','Lower'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ma with your own data.\n
        """
        data = self.__data[source] if data is None else data

        match ma_type:
            case 'sma': ma = self.__idc_sma(data=data, length=length)
            case 'ema': ma = self.__idc_ema(data=data, length=length)
            case 'wma': ma = self.__idc_wma(data=data, length=length)
        ma = np.flip(ma)
        std_ = (std_dev * data.rolling(window=length).std())
        bb = pd.DataFrame({'Upper':ma + std_,
                           ma_type:ma,
                           'Lower':ma - std_}, index=ma.index)

        return bb.apply(lambda col: col.iloc[len(bb.index)-last if last != None and last < len(bb.index) else 0:], axis=0)

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, rsi_ma_type:str = 'wma', base_type:str = 'sma', bb_std_dev:float = 2, source:str = 'Close', last:int = None):
        """
        Relative strength index.
        ----
        Return an pd.DataFrame with the value of rsi and 'base_type' for each step.\n
        Columns: 'rsi',('base_type').\n
        Parameters:
        --
        >>> length_rsi:int 14
        >>> length:int = 14
        >>> rsi_ma_type:str = 'wma'
        >>> base_type:str = 'sma'
        >>> bb_std_dev:float = 2
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length_rsi: \n
        \tWindow length of 'rsi_ma_type'.\n
        length: \n
        \tWindow length of 'base_type'.\n
        rsi_ma_type: \n
        \tType of ma used for calculating rsi.\n
        base_type: \n
        \tType of ma base used applied to rsi.\n
        bb_std_dev: \n
        \tStandard deviation for bb.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif bb_std_dev > 50 or bb_std_dev < 0.001: raise ValueError("'bb_std_dev' it has to be greater than 0.001 and less than 50.")
        elif length > 5000 or length <= 0: raise ValueError("'length_rsi' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif not rsi_ma_type in ('sma','ema','wma'): raise ValueError("'rsi_ma_type' only these values: 'sma', 'ema', 'wma'.")
        elif not base_type in ('sma','ema','wma','bb'): raise ValueError("'base_type' only these values: 'sma', 'ema', 'wma', 'bb'.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Rsi calc.
        return self.__idc_rsi(length_rsi=length_rsi, length=length, base_type=base_type, source=source, last=last)

    def __idc_rsi(self, data:pd.Series = None, length_rsi:int = 14, length:int = 14, rsi_ma_type:str = 'wma', base_type:str = 'sma', bb_std_dev:float = 2, source:str = 'Close', last:int = None):
        """
        Relative strength index.
        ----
        Return an pd.DataFrame with the value of rsi and 'base_type' for each step.\n
        Columns: 'rsi',('base_type').\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ma with your own data.\n
        """
        delta = self.__data[source].diff() if data is None else data.diff()

        match rsi_ma_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma

        wma_gain = ma(data = delta.where(delta > 0, 0), length=length_rsi, source=source)
        wma_loss = ma(data = -delta.where(delta < 0, 0), length=length_rsi, source=source)
        rsi = np.flip(100 - (100 / (1+wma_gain/wma_loss)))

        match base_type:
            case 'sma': mv = self.__idc_sma(data=rsi, length=length, source=source)
            case 'ema': mv = self.__idc_ema(data=rsi, length=length, source=source)
            case 'wma': mv = self.__idc_wma(data=rsi, length=length, source=source)
            case 'bb': mv = self.__idc_bb(data=rsi, length=length, std_dev=bb_std_dev, source=source)
        if type(mv) == pd.Series: mv.name = base_type

        rsi = pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

        return rsi.apply(lambda col: col.iloc[len(rsi.index)-last if last != None and last < len(rsi.index) else 0:], axis=0)

    def idc_stochastic():
        """
        Stochastic.
        ----
        """
        pass

    def idc_adx():
        """
        Average directional index.
        ----
        """
        pass

    def idc_macd():
        """
        Convergence/divergence of the moving average.
        ----
        """
        pass

    def idc_sqzmom():
        """
        Squeeze momentum.
        ----
        """
        pass

    def idc_ichimoku():
        """
        Ichimoku cloud.
        ----
        """
        pass

    def idc_fibonacci():
        """
        Fibonacci retracement.
        ----
        """
        pass

    def idc_atr():
        """
        Average true range.
        ----
        """
        pass
    
    def act_open(self, type:int = 1, stop_loss:int = np.nan, take_profit:int = np.nan, amount:int = np.nan) -> None:
        """
        Open action.
        ----
        Open an action.\n
        Warning:
        --
        If you leave the stop loss and take profit in np.nan your trade will be counted as closed and you can't modify it or close it.\n
        Parameters:
        --
        >>> type:int = 1
        >>> stop_loss:int = np.nan
        >>> take_profit:int = np.nan
        >>> amount:int = np.nan
        \n
        type: \n
        \t0 is a sell position, 1 is a buy position.\n
        stop_loss: \n
        \tThe price where the stop loss will go, if it is left at np.nan the position is opened without stop loss.\n
        take_profit: \n
        \tThe price where the take profit will go, if it is left at np.nan the position is opened without take profit.\n
        amount: \n
        \tThe amount of imaginary points with which you would enter the position.\n
        """
        # Check if type is 1 or 0.
        if not type in {1,0}: raise exception.ActionError("'type' only 1 or 0.")
        # Check exceptions.
        if amount <= 0: raise exception.ActionError("'amount' can only be greater than 0.")
        if (type and (self.__data["Close"].iloc[-1] <= stop_loss or self.__data["Close"].iloc[-1] >= take_profit)) or (not type and (self.__data["Close"].iloc[-1] >= stop_loss or self.__data["Close"].iloc[-1] <= take_profit)): raise exception.ActionError("'stop_loss' or 'take_profit' incorrectly configured for the position type.")
        # Create new trade.
        self.__trade = pd.DataFrame({'Date':self.__data.index[-1],'Close':self.__data["Close"].iloc[-1],'Low':self.__data["Low"].iloc[-1],'High':self.__data["High"].iloc[-1],'StopLoss':stop_loss,'TakeProfit':take_profit,'PositionClose':np.nan,'PositionDate':np.nan,'Amount':amount,'ProfitPer':np.nan,'Profit':np.nan,'Type':type},index=[1])

    def act_close(self, index:int = 0) -> None:
        """
        Close action.
        ----
        Close an action.\n
        Parameters:
        --
        >>> index:int = 0
        \n
        index: \n
        \tThe index of the active trade you want to close.\n
        """
        # Check exceptions.
        if self.__trades_ac.empty: raise exception.ActionError('There are no active trades.')
        elif not index in self.__trades_ac.index.to_list(): raise exception.ActionError('Index does not exist.')
        # Close action.
        return self.__act_close(index=index)

    def __act_close(self, index:int = 0) -> None:
        """
        Close action.
        ----
        Close an action.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        """
        # Get trade to close.
        trade = self.__trades_ac.iloc[lambda x: x.index==index].copy()
        self.__trades_ac = self.__trades_ac.drop(trade.index)
        # Get PositionClose
        take = trade['TakeProfit'].iloc[0];stop = trade['StopLoss'].iloc[0]
        position_close = (stop if self.__data["Low"].iloc[-1] <= stop else take if self.__data["High"].iloc[-1] >= take else self.__data["Close"].iloc[-1]) if trade['Type'].iloc[0] else (stop if self.__data["High"].iloc[-1] >= stop else take if self.__data["Low"].iloc[-1] <= take else self.__data["Close"].iloc[-1])
        # Fill data.
        trade['PositionClose'] = position_close
        trade['PositionDate'] = self.__data.index[-1]
        open = trade['Close'].iloc[0]
        trade['ProfitPer'] = (position_close-open)/open*100 if trade['Type'].iloc[0] else (open-position_close)/open*100
        trade['Profit'] = trade['Amount'].iloc[0]*trade['ProfitPer'].iloc[0]/100-trade['Amount']*(self.__commission/100) if not np.isnan(trade['Amount'].iloc[0]) else np.nan

        self.__trades_cl = pd.concat([self.__trades_cl,trade], ignore_index=True) ; self.__trades_cl.reset_index(drop=True, inplace=True)

    def act_mod(self, index:int = 0, new_stop:int = None, new_take:int = None) -> None:
        """
        Modify action.
        ----
        Modify an action.\n
        Alert:
        --
        If an invalid stop loss or invalid takeprofit is submitted,\n
        the program will return None and will not execute any changes.\n
        Parameters:
        --
        >>> index:int = 0
        >>> new_stop:int = None
        >>> new_take:int = None
        \n
        index: \n
        \tThe index of the active trade you want to modify.\n
        new_stop: \n
        \tThe price where the new stop loss will be,\n
        \tif it is left at None the stop loss will not be modified and if it is left at np.nan the stop loss will be removed.\n
        new_take: \n
        \tThe price where the new take profit will be,\n
        \tif it is left at None the take profit will not be modified and if it is left at np.nan the take profit will be removed.\n
        """
        # Check exceptions.
        if self.__trades_ac.empty: raise exception.ActionError('There are no active trades.')
        elif not (new_stop or new_take): raise exception.ActionError('Nothing was changed.')
        # Get trade to modify.
        trade = self.__trades_ac.loc[index]
        # Set new stop.
        if new_stop and ((new_stop < self.__data["Close"].iloc[-1] and trade['Type']) or (not trade['Type'] and new_stop > self.close) or np.isnan(new_stop)): 
            self.__trades_ac.loc[index, 'StopLoss'] = new_stop 
        # Set new take.
        if new_take and ((new_take > self.__data["Close"].iloc[-1] and trade['Type']) or (not trade['Type'] and new_take < self.close) or np.isnan(new_take)): 
            self.__trades_ac.loc[index,'TakeProfit'] = new_take