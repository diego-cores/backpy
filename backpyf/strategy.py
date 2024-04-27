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
    Get:
    >>> get_init_founds
    >>> get_commission

    Actions:
    >>> act_mod
    >>> act_close
    >>> act_open

    Hidden:
    >>> __act_close

    Prev:
    >>> prev_trades_ac
    >>> prev_trades_cl
    >>> prev

    Indicators:
    >>> idc_hvolume
    >>> idc_ema
    >>> idc_sma
    >>> idc_wma
    >>> idc_smma
    >>> idc_smema
    >>> idc_bb
    >>> idc_rsi
    >>> idc_stochastic
    >>> idc_adx
    >>> idc_macd
    >>> idc_sqzmom
    >>> idc_mom
    >>> idc_ichimoku
    >>> idc_fibonacci
    >>> idc_atr
    
    Hidden:
    >>> __idc_ema
    >>> __idc_sma
    >>> __idc_wma
    >>> __idc_smma
    >>> __idc_smema
    >>> __idc_bb
    >>> __idc_rsi
    >>> __idc_stochastic
    >>> __idc_adx
    >>> __idc_macd
    >>> __idc_sqzmom
    >>> __idc_mom
    >>> __idc_ichimoku
    >>> __idc_fibonacci
    >>> __idc_atr
    
    Utils:
    >>> __idc_rlinreg
    >>> __idc_trange

    Others:
    >>> next

    Hidden:
    >>> __before

    """ 
    def __init__(self, data:pd.DataFrame = any, trades_cl:pd.DataFrame = pd.DataFrame(), trades_ac:pd.DataFrame = pd.DataFrame(), commission:float = 0, init_funds:int = 0) -> None: 
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
    def next(self) -> None: ...

    def get_commission(self) -> float:
        """
        get_commission
        ----
        Return hidden variable '__commission'.
        """
        return self.__commission
    
    def get_init_founds(self) -> int:
        """
        get_init_founds
        ----
        Return hidden variable '__init_funds'.
        """
        return self.__init_funds

    def __before(self):
        """
        Before.
        ----
        This function is used to run trades and other things.
        """
        self.next()

        # Check if a trade needs to be closed.
        self.__trades_ac.apply(lambda row: self.__act_close(index=row.name) if (not row['Type'] and (self.__data["Low"].iloc[-1] <= row['TakeProfit'] or self.__data["High"].iloc[-1] >= row['StopLoss'])) or (row['Type'] and (self.__data["High"].iloc[-1] >= row['TakeProfit'] or self.__data["Low"].iloc[-1] <= row['StopLoss'])) else None, axis=1) 

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
        You can do the calculation of ema with your own data.\n
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
        You can do the calculation of sma with your own data.\n
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
        \tThe distribution of weights is done the other way around.\n
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
        You can do the calculation of wma with your own data.\n
        """
        data = self.__data[source] if data is None else data

        weight = np.arange(1, length+1)[::-1] if invt_weight else np.arange(1, length+1)
        wma = data.rolling(window=length).apply(lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return np.flip(wma[len(wma)-last if last != None and last < len(wma) else 0:])
    
    def idc_smma(self, length:int = any, source:str = 'Close', last:int = None):
        """
        Smoothed moving average.
        ---- 
        Returns an pd.Series with all the steps of an smma with the length you indicate.\n
        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tSmma length.\n
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

        # Smma calc.
        return self.__idc_smma(length=length, source=source, last=last)
    
    def __idc_smma(self, data:pd.Series = None, length:int = any, source:str = 'Close', last:int = None):
        """
        Smoothed moving average.
        ---- 
        Returns an pd.Series with all the steps of an smma with the length you indicate.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ema with your own data.\n
        """
        data = self.__data[source] if data is None else data
        smma = data.ewm(alpha=1/length, adjust=False).mean()

        return np.flip(smma[len(smma)-last if last != None and last < len(smma) else 0:])
    
    def idc_smema(self, length:int = 9, method:str = 'sma', smooth:int = 5, only:bool = False, source:str = 'Close', last:int = None):
        """
        Smoothed exponential moving average.
        ---- 
        Return an pd.DataFrame with the value of ema and the smoothed ema for each step.\n
        Columns: 'ema','smoothed'.\n
        Parameters:
        --
        >>> length:int = any
        >>> method:str = 'sma'
        >>> smooth:int = 5
        >>> only:bool = False
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tEma length.\n
        method: \n
        \tSmooth method.\n
        smooth: \n
        \t'method' length.\n
        only: \n
        \tIf left true, only one pd.Series will be returned with the values ​​of 'method'.\n
        ma_type: \n
        \tMa type.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif not method in ('sma','ema','smma','wma'): raise ValueError("'method' only one of these values: ['sma','ema','smma','wma'].")
        elif smooth > 5000 or smooth <= 0: raise ValueError("'smooth' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Smema calc.
        return self.__idc_smema(length=length, method=method, smooth=smooth, only=only, source=source, last=last)
    
    def __idc_smema(self, data:pd.Series = None, length:int = 9, method:str = 'sma', smooth:int = 5, only:bool = False, source:str = 'Close', last:int = None):
        """
        Smoothed exponential moving average.
        ---- 
        Return an pd.DataFrame with the value of ema and the smoothed ema for each step.\n
        Columns: 'ema','smoothed'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of ema with your own data.\n
        """
        data = self.__data[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()

        match method:
            case 'sma': smema = self.__idc_sma(data=ema, length=smooth)
            case 'ema': smema = self.__idc_ema(data=ema, length=smooth)
            case 'smma': smema = self.__idc_smma(data=ema, length=smooth)
            case 'wma': smema = self.__idc_wma(data=ema, length=smooth)

        if only: smema = np.flip(smema); return np.flip(smema[len(smema)-last if last != None and last < len(smema) else 0:])
        smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)

        return smema.apply(lambda col: col.iloc[len(smema.index)-last if last != None and last < len(smema.index) else 0:], axis=0)

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
        elif not ma_type in ('sma','ema','wma','smma'): raise ValueError("'ma_type' only these values: 'sma', 'ema', 'wma', 'smma'.")
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
        You can do the calculation of bb with your own data.\n
        """
        data = self.__data[source] if data is None else data

        match ma_type:
            case 'sma': ma = self.__idc_sma(data=data, length=length)
            case 'ema': ma = self.__idc_ema(data=data, length=length)
            case 'wma': ma = self.__idc_wma(data=data, length=length)
            case 'smma': ma = self.__idc_smma(data=data, length=length)
        ma = np.flip(ma)
        std_ = (std_dev * data.rolling(window=length).std())
        bb = pd.DataFrame({'Upper':ma + std_,
                           ma_type:ma,
                           'Lower':ma - std_}, index=ma.index)

        return bb.apply(lambda col: col.iloc[len(bb.index)-last if last != None and last < len(bb.index) else 0:], axis=0)

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, rsi_ma_type:str = 'smma', base_type:str = 'sma', bb_std_dev:float = 2, source:str = 'Close', last:int = None):
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
        elif not rsi_ma_type in ('sma','ema','wma','smma'): raise ValueError("'rsi_ma_type' only these values: 'sma', 'ema', 'wma','smma'.")
        elif not base_type in ('sma','ema','wma','bb'): raise ValueError("'base_type' only these values: 'sma', 'ema', 'wma', 'smma', 'bb'.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Rsi calc.
        return self.__idc_rsi(length_rsi=length_rsi, length=length, rsi_ma_type=rsi_ma_type, base_type=base_type, bb_std_dev=bb_std_dev, source=source, last=last)

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
        You can do the calculation of rsi with your own data.\n
        """
        delta = self.__data[source].diff() if data is None else data.diff()

        match rsi_ma_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        ma_gain = ma(data = delta.where(delta > 0, 0), length=length_rsi, source=source)
        ma_loss = ma(data = -delta.where(delta < 0, 0), length=length_rsi, source=source)
        rsi = np.flip(100 - (100 / (1+ma_gain/ma_loss)))

        match base_type:
            case 'sma': mv = self.__idc_sma(data=rsi, length=length)
            case 'ema': mv = self.__idc_ema(data=rsi, length=length)
            case 'wma': mv = self.__idc_wma(data=rsi, length=length)
            case 'smma': mv = self.__idc_smma(data=rsi, length=length)
            case 'bb': mv = self.__idc_bb(data=rsi, length=length, std_dev=bb_std_dev)
        if type(mv) == pd.Series: mv.name = base_type

        rsi = pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

        return rsi.apply(lambda col: col.iloc[len(rsi.index)-last if last != None and last < len(rsi.index) else 0:], axis=0)

    def idc_stochastic(self, length_k:int = 14, smooth_k:int = 1, length_d:int = 3, d_type:int = 'sma', source:str = 'Close', last:int = None):
        """
        Stochastic.
        ----
        Return an pd.DataFrame with the value of stochastic and 'd_type' for each step.\n
        Columns: 'stoch',('d_type').\n
        Parameters:
        --
        >>> length_k:int 14
        >>> smooth_k:int = 1
        >>> length_d:int = 3
        >>> d_type:int = 'sma'
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length_k: \n
        \tWindow length to calculate 'stoch'.\n
        smooth_k: \n
        \tWindow length of 'stoch'.\n
        length_d: \n
        \tWindow length of 'd_type'.\n
        d_type: \n
        \tType of ma base used applied to stochastic.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length_k > 5000 or length_k <= 0: raise ValueError("'length_k' it has to be greater than 0 and less than 5000.")
        elif smooth_k > 5000 or smooth_k <= 0: raise ValueError("'smooth_k' it has to be greater than 0 and less than 5000.")
        elif length_d > 5000 or smooth_k <= 0: raise ValueError("'length_d' it has to be greater than 0 and less than 5000.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif not d_type in ('sma','ema','wma','smma'): raise ValueError("'d_type' only these values: 'sma', 'ema', 'wma', 'smma'.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc stoch.
        return self.__idc_stochastic(length_k=length_k, smooth_k=smooth_k, length_d=length_d, d_type=d_type, source=source, last=last)

    def __idc_stochastic(self, data:pd.Series = None, length_k:int = 14, smooth_k:int = 1, length_d:int = 3, d_type:int = 'sma', source:str = 'Close', last:int = None):
        """
        Stochastic.
        ----
        Return an pd.DataFrame with the value of stochastic and 'd_type' for each step.\n
        Columns: 'stoch',('d_type').\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        low_data = self.__data['Low'].rolling(window=length_k).min()
        high_data = self.__data['High'].rolling(window=length_k).max()

        match d_type:
            case 'sma': ma = self.__idc_sma
            case 'ema': ma = self.__idc_ema
            case 'wma': ma = self.__idc_wma
            case 'smma': ma = self.__idc_smma

        stoch = (((self.__data[source] - low_data) / (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
        result = pd.DataFrame({'stoch':stoch, d_type:ma(data=stoch, length=length_d)})

        return result.apply(lambda col: col.iloc[len(result.index)-last if last != None and last < len(result.index) else 0:], axis=0) 

    def idc_adx(self, smooth:int = 14, length_di:int = 14, only:bool = False, last:int = None):
        """
        Average directional index.
        ----
        Return an pd.Dataframe with the value of adx, +di and -di for each step.\n
        Columns: 'adx','+di','-di'.\n
        Parameters:
        --
        >>> smooth:int = 14
        >>> length_di:int = 14
        >>> only:bool = False
        >>> last:int = None
        \n
        smooth: \n
        \tSmooth length.\n
        length_di: \n
        \tWindow length for calculate 'di'.\n
        only: \n
        \tIf left true, only one pd.Series will be returned with the values ​​of adx.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if smooth > 5000 or smooth <= 0: raise ValueError("'smooth' it has to be greater than 0 and less than 5000.")
        elif length_di > 5000 or length_di <= 0: raise ValueError("'length_di' it has to be greater than 0 and less than 5000.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc adx.
        return self.__idc_adx(smooth=smooth, length_di=length_di, only=only, last=last)

    def __idc_adx(self, data:pd.Series = None, smooth:int = 14, length_di:int = 14, only:bool = False, last:int = None):
        """
        Average directional index.
        ----
        Return an pd.Dataframe with the value of adx, +di and -di for each step.\n
        Columns: 'adx','+di','-di'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        atr = self.__idc_atr(length=length_di, smooth='smma')

        dm_p = data['High'].diff()
        dm_n = -data['Low'].diff()

        di_p = 100 * np.flip(self.__idc_smma(data=dm_p.where(dm_p >= 0, 0), length=length_di)/atr)
        di_n = 100 * np.flip(self.__idc_smma(data=dm_n.where(dm_n >= 0, 0), length=length_di)/atr)
        adx = self.__idc_smma(data=100 * np.abs((di_p - di_n) / (di_p + di_n)), length=smooth)

        if only: adx = np.flip(adx); return np.flip(adx[len(adx)-last if last != None and last < len(adx) else 0:])
        adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

        return adx.apply(lambda col: col.iloc[len(adx.index)-last if last != None and last < len(adx.index) else 0:], axis=0) 

    def idc_macd(self, short_len:int = 12, long_len:int = 26, signal_len:int = 9, macd_ma_type:str = 'ema', signal_ma_typ:str = 'ema', histogram:bool = True, source:str = 'Close', last:int = None):
        """
        Convergence/divergence of the moving average.
        ----
        Return an pd.Dataframe with the value of macd and signal for each step.\n
        Columns: 'macd','signal',('histogram' if histogram is True).\n
        Parameters:
        --
        >>> short_len:int = 12
        >>> long_len:int = 26
        >>> signal_len:int = 9
        >>> macd_ma_type:str = 'ema'
        >>> signal_ma_typ:str = 'ema'
        >>> histogram:bool = True
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        short_len: \n
        \tShort ma length.\n
        \tThe short ma is used to calculate macd.\n
        long_len: \n
        \tLong ma length.\n
        \tThe long ma is used to calculate macd.\n
        signal_len: \n
        \tSignal ma length.\n
        \tThe signal ma is the smoothed macd.\n
        macd_ma_type: \n
        \tType of ma to calculate macd.\n
        signal_ma_typ: \n
        \tType of ma to smooth macd.\n
        histogram: \n
        \tAn extra column will be returned with the histogram.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if short_len > 5000 or short_len <= 0: raise ValueError("'short_len' it has to be greater than 0 and less than 5000.")
        elif long_len > 5000 or long_len <= 0: raise ValueError("'long_len' it has to be greater than 0 and less than 5000.")
        elif signal_len > 5000 or signal_len <= 0: raise ValueError("'signal_len' it has to be greater than 0 and less than 5000.")
        elif not macd_ma_type in ('ema','sma'): raise ValueError("'macd_ma_type' only one of these values: ['ema','sma'].")
        elif not signal_ma_typ in ('ema','sma'): raise ValueError("'signal_ma_typ' only one of these values: ['ema','sma'].")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc macd.
        return self.__idc_macd(short_len=short_len, long_len=long_len, signal_len=signal_len, macd_ma_type=macd_ma_type, signal_ma_typ=signal_ma_typ, histogram=histogram, source=source, last=last)

    def __idc_macd(self, data:pd.Series = None, short_len:int = 12, long_len:int = 26, signal_len:int = 9, macd_ma_type:str = 'ema', signal_ma_typ:str = 'ema', histogram:bool = True, source:str = 'Close', last:int = None):
        """
        Convergence/divergence of the moving average.
        ----
        Return an pd.Dataframe with the value of macd and signal for each step.\n
        Columns: 'macd','signal'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        match macd_ma_type:
            case 'ema':
                macd_ma = self.__idc_ema
            case 'sma':
                macd_ma = self.__idc_sma

        match signal_ma_typ:
            case 'ema':
                signal_ma = self.__idc_ema
            case 'sma':
                signal_ma = self.__idc_sma
        
        short_ema = np.flip(macd_ma(data=data[source], length=short_len))
        long_ema = np.flip(macd_ma(data=data[source], length=long_len))
        macd = short_ema - long_ema

        signal_line = np.flip(signal_ma(data=macd, length=signal_len))
        result = pd.DataFrame({'macd':macd, 'signal':signal_line, 'histogram':macd-signal_line})

        return result.apply(lambda col: col.iloc[len(result.index)-last if last != None and last < len(result.index) else 0:], axis=0) 

    def idc_sqzmom(self, bb_len:int = 20, bb_mult:float = 1.5, kc_len:int = 20, kc_mult:float = 1.5, use_tr:bool = True, histogram_len:int = 50, source:str = 'Close', last:int = None):
        """
        Squeeze momentum.
        ----
        This function calculates the Squeeze Momentum, which is inspired by the Squeeze Momentum Indicator available on TradingView. While the concept is based on the original indicator,\n
        please note that this implementation may not fully replicate the exact functionality of the original.\n
        Credit for the concept of the Squeeze Momentum goes to its original developer.\n
        This function is an adaptation and may differ from the original version.\n
        Please note that this function is intended for use in backtesting scenarios, where it may utilize real market data or simulated random data.\n
        It should be used for research and educational purposes only, and should not be considered as financial advice.\n
        Return:
        ----
        Return an pd.Dataframe with the value of sqz and histogram for each step.\n
        Columns: 'sqzmom','histogram'.\n
        Parameters:
        --
        >>> bb_len:int = 20
        >>> bb_mult:float = 1.5
        >>> kc_len:int = 20
        >>> kc_mult:float = 1.5
        >>> use_tr:bool = True
        >>> histogram_len:int = 50
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        bb_len: \n
        \tBollinger band length.\n
        bb_mult: \n
        \tBollinger band standard deviation.\n
        kc_len: \n
        \tKc length.\n
        kc_mult: \n
        \tKc standard deviation.\n
        use_tr: \n
        \tIf left false, ('High'-'Low') will be used instead of the true range.\n
        histogram_len: \n
        \tHow many steps from the present backward do you want the histogram to be calculated.\n
        \tThe higher the number, the less efficient.\n
        \tIf you leave it at 0, the 'historiogram' column will not be returned.\n
        source: \n
        \tData.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if bb_len > 5000 or bb_len <= 0: raise ValueError("'bb_len' it has to be greater than 0 and less than 5000.")
        elif bb_mult > 50 or bb_mult < 0.001: raise ValueError("'bb_mult' it has to be greater than 0.001 and less than 50.")
        elif kc_len > 5000 or kc_len <= 0: raise ValueError("'kc_len' it has to be greater than 0 and less than 5000.")
        elif kc_mult > 50 or kc_mult < 0.001: raise ValueError("'bb_mult' it has to be greater than 0.001 and less than 50.")
        elif histogram_len < 0: raise ValueError("'histogram_len' has to be greater or equal than 0.")
        elif not source in ('Close','Open','High','Low'): raise ValueError("'source' only one of these values: ['Close','Open','High','Low'].")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc sqzmom.
        return self.__idc_sqzmom(bb_len=bb_len, bb_mult=bb_mult, kc_len=kc_len, kc_mult=kc_mult, use_tr=use_tr, histogram_len=histogram_len, source=source, last=last)

    def __idc_sqzmom(self, data:pd.Series = None, bb_len:int = 20, bb_mult:float = 1.5, kc_len:int = 20, kc_mult:float = 1.5, use_tr:bool = True, histogram_len:int = 50, source:str = 'Close', last:int = None):
        """
        Squeeze momentum.
        ----
        This function calculates the Squeeze Momentum, which is inspired by the Squeeze Momentum Indicator available on TradingView. While the concept is based on the original indicator,\n
        please note that this implementation may not fully replicate the exact functionality of the original.\n
        Credit for the concept of the Squeeze Momentum goes to its original developer.\n
        This function is an adaptation and may differ from the original version.\n
        Please note that this function is intended for use in backtesting scenarios, where it may utilize real market data or simulated random data.\n
        It should be used for research and educational purposes only, and should not be considered as financial advice.\n
        Return:
        ----
        Return an pd.Dataframe with the value of sqzmom and histogram for each step.\n
        Columns: 'sqzmom','histogram'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        basis = np.flip(self.__idc_sma(length=bb_len))
        dev = bb_mult * data[source].rolling(window=bb_len).std(ddof=0)

        upper_bb = basis + dev
        lower_bb = basis - dev

        ma = np.flip(self.__idc_sma(length=kc_len))
        range_ = np.flip(self.__idc_sma(data=np.flip(self.__idc_trange()) if use_tr else data['High']-data['Low'], length=kc_len))
        
        upper_kc = ma + range_ * kc_mult
        lower_kc = ma - range_ * kc_mult

        sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

        if histogram_len < 1: 
            result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index)})
            return result.apply(lambda col: col.iloc[len(result.index)-last if last != None and last < len(result.index) else 0:], axis=0)
        elif histogram_len > last: histogram = last
        
        histogram_len += kc_len
        d = data[source] - ((data['Low'].rolling(window=kc_len).min() + data['High'].rolling(window=kc_len).max()) / 2 + np.flip(self.__idc_sma(length=kc_len))) / 2
        histogram = self.__idc_rlinreg(data=d[len(d.index)-histogram_len if len(d.index) > histogram_len else 0:], length=kc_len, offset=0)

        result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index), 'histogram':pd.Series(histogram)}, index=data.index)
        return result.apply(lambda col: col.iloc[len(result.index)-last if last != None and last < len(result.index) else 0:], axis=0) 

    def __idc_rlinreg(self, data:pd.Series = None, length:int = 5, offset:int = 1):
        """
        Rolling linear regression.
        ----
        This function is not very efficient. I recommend that the data does not exceed 50 in length.\n
        Return an pd.Series with the value of each linear regression.\n
        Calculated linear regression: m * (length - 1 - offset) + b\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Parameters:
        --
        >>> data:pd.Series = None
        >>> length:int = 5
        >>> offset:int = 1
        \n
        data: \n
        \tYou can do the calculation of stochastic with your own data.\n
        length: \n
        \tLength of each window.\n
        """
        data = self.__data if data is None else data

        x = np.arange(length)
        y = data.rolling(window=length)

        m = y.apply(lambda y: np.polyfit(x, y, 1)[0])
        b = y.mean() - (m * np.mean(x)) 

        return m * (length - 1 - offset) + b

    def idc_mom(self, length:int = 10, source:str = 'Close', last:int = None):
        """
        Momentum.
        ----
        Return an pd.Series with all the steps of momentum with the length you indicate.\n
        Parameters:
        --
        >>> length:int = 10
        >>> source:str = 'Close'
        >>> last:int = None
        \n
        length: \n
        \tLength to calculate momentum.\n
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

        # Calc momentum.
        return self.__idc_mom(length=length, source=source, last=last)

    def __idc_mom(self, data:pd.Series = None, length:int = 10, source:str = 'Close', last:int = None):
        """
        Momentum.
        ----
        Return an pd.Series with all the steps of momentum with the length you indicate.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data
        mom = data[source] - data[source].shift(length)

        return np.flip(mom[len(mom)-last if last != None and last < len(mom) else 0:])

    def idc_ichimoku(self, tenkan_period:int = 9, kijun_period=26, senkou_span_b_period=52, ichimoku_lines:bool = True, last:int = None):
        """
        Ichimoku cloud.
        ----
        Return an pd.Dataframe with the value of ichimoku cloud, tenkan_sen and kijun_sen for each step.\n
        Columns: 'senkou_a','senkou_b'.\n
        If ichimoku lines is true these columns are added to the dataframe.\n
        Added: 'tenkan_sen','kijun_sen'.\n
        Parameters:
        --
        >>> tenkan_period:int = 12
        >>> kijun_period:int = 26
        >>> senkou_span_b_period:int = 9
        >>> ichimoku_lines:str = 'ema'
        >>> last:int = None
        \n
        tenkan_period: \n
        \tWindow length to calculate tenkan.\n
        kijun_period: \n
        \tWindow length to calculate kijun.\n
        senkou_span_b_period: \n
        \tWindow length to calculate senkou span.\n
        ichimoku_lines: \n
        \tIf ichimoku lines is true these columns are added to the dataframe: 'tenkan_sen','kijun_sen'.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if tenkan_period > 5000 or tenkan_period <= 0: raise ValueError("'tenkan_period' it has to be greater than 0 and less than 5000.")
        elif kijun_period > 5000 or kijun_period <= 0: raise ValueError("'kijun_period' it has to be greater than 0 and less than 5000.")
        elif senkou_span_b_period > 5000 or senkou_span_b_period <= 0: raise ValueError("'senkou_span_b_period' it has to be greater than 0 and less than 5000.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc ichimoku.
        return self.__idc_ichimoku(tenkan_period=tenkan_period, kijun_period=kijun_period, senkou_span_b_period=senkou_span_b_period, ichimoku_lines=ichimoku_lines, last=last)

    def __idc_ichimoku(self, data:pd.Series = None, tenkan_period:int = 9, kijun_period=26, senkou_span_b_period=52, ichimoku_lines:bool = True, last:int = None):
        """
        Ichimoku cloud.
        ----
        Return an pd.Dataframe with the value of ichimoku cloud, tenkan_sen and kijun_sen for each step.\n
        Columns: 'senkou_a','senkou_b'.\n
        If ichimoku lines is true these columns are added to the dataframe.
        Added: 'tenkan_sen','kijun_sen'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        tenkan_sen_val = (data['High'].rolling(window=tenkan_period).max() + data['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen_val = (data['High'].rolling(window=kijun_period).max() + data['Low'].rolling(window=kijun_period).min()) / 2

        senkou_span_a_val = ((tenkan_sen_val + kijun_sen_val) / 2)
        senkou_span_b_val = ((data['High'].rolling(window=senkou_span_b_period).max() + data['Low'].rolling(window=senkou_span_b_period).min()) / 2)
        senkou_span = pd.DataFrame({'senkou_a':senkou_span_a_val,'senkou_b':senkou_span_b_val, 'tenkan_sen':tenkan_sen_val,'kijun_sen':kijun_sen_val}) if ichimoku_lines else pd.DataFrame({'senkou_a':senkou_span_a_val,'senkou_b':senkou_span_b_val})
        
        return senkou_span.apply(lambda col: col.iloc[len(senkou_span.index)-last if last != None and last < len(senkou_span.index) else 0:], axis=0)

    def idc_fibonacci(self, start:int = None, end:int = 30, met:bool = False, source:str = 'Low/High'):
        """
        Fibonacci retracement.
        ----
        Return an pd.DataFrame with the fibonacci levels and values.\n
        Columns: 'Level','Value'.\n
        Parameters:
        --
        >>> start:int = None
        >>> end:int = 30
        >>> met:bool = False
        >>> source:str = 'Low/High'
        \n
        start: \n
        \tWhere do you want level 0 to be.\n
        \tIf 'met' is false 'start' is the number of candles back to open the fibonacci.\n
        \tNone == 0.\n
        end: \n
        \tWhere do you want level 1 to be.\n
        \tIf 'met' is false 'end' is the number of candles back to close the fibonacci.\n
        \tNone == 0.\n
        met: \n
        \tIf left false, 'start' and 'end' are the number of candles backwards,\n
        \totherwise 'start' and 'end' are the value from which Fibonacci opens.\n
        source: \n
        \tData that is extracted.\n
        \tStart and end data format: 's/s where 's' is each source.'\n
        \tValues supported in source: ('Close', 'Open', 'High', 'Low')\n
        \tIf left 'met' in true 'source' does not work.\n
        """
        if met: return self.__idc_fibonacci(start=start, end=end)

        source = source.split('/')
        if len(source) != 2 or source[0] == '' or source[1] == '' or not source[0] in ('Close', 'Open', 'High', 'Low') or not source[1] in ('Close', 'Open', 'High', 'Low'): 
            raise ValueError("'source' it has to be in this format: 's/s' where 's' is each source.\nValues supported in source: ('Close', 'Open', 'High', 'Low')")
        data_start = self.__data[source[1]]; data_end = self.__data[source[0]]

        # Fibonacci calc.
        return self.__idc_fibonacci(start=data_start.iloc[len(data_start)-start-1 if start != None and start < len(data_start) else 0], end=data_end.iloc[len(data_end)-end-1 if end != None and end < len(data_end) else 0])

    def __idc_fibonacci(self, start:int = 10, end:int = 1):
        """
        Fibonacci retracement.
        ----
        Return an pd.DataFrame with the fibonacci levels and values.\n
        Columns: 'Level','Value'.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        """
        fibo_levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 3.618, 4.236])

        return pd.DataFrame({'Level':fibo_levels,'Value':(start + (end - start) * fibo_levels)})

    def idc_atr(self, length:int = 14, smooth:str = 'wma', last:int = None):
        """
        Average true range.
        ----
        Return an pd.Series with the value of average true range for each step.\n
        Parameters:
        --
        >>> length:int = 14
        >>> smooth:str = 'wma'
        >>> last:int = None
        \n
        length: \n
        \tWindow length to smooth 'atr'.\n
        smooth: \n
        \tMa used to smooth 'atr'.\n
        last: \n
        \tHow much data starting from the present backwards do you want to be returned.\n
        \tIf you leave it at None, the data for all times is returned.\n
        """
        if length > 5000 or length <= 0: raise ValueError("'length' it has to be greater than 0 and less than 5000.")
        elif not smooth in ('smma', 'sma','ema','wma'): raise ValueError("'smooth' only these values: 'smma', 'sma', 'ema', 'wma'.")
        elif last != None:
            if last <= 0 or last > self.__data["Close"].shape[0]: raise ValueError("Last has to be less than the length of 'data' and greater than 0.")

        # Calc atr.
        return self.__idc_atr(length=length, smooth=smooth, last=last)
    
    def __idc_atr(self, length:int = 14, smooth:str = 'wma', last:int = None):
        """
        Average true range.
        ----
        Return an pd.Series with the value of average true range for each step.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        tr = np.flip(self.__idc_trange())

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

    def __idc_trange(self, data:pd.Series = None, last:int = None):
        """
        True range.
        ----
        Return an pd.Series with the value of true range for each step.\n
        Hidden function to prevent user modification.\n
        Function without exception handling.\n
        Data parameter:
        --
        You can do the calculation of stochastic with your own data.\n
        """
        data = self.__data if data is None else data

        close = data['Close'].shift(1)
        close.fillna(data['Low'], inplace=True)

        hl = data['High'] - data['Low']
        hyc = abs(data['High'] - close)
        lyc = abs(data['Low'] - close)
        tr = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)
        return np.flip(tr[len(tr)-last if last != None and last < len(tr) else 0:])

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