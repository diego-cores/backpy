"""
Strategy.
----
Here is the main class that has to be inherited in order to 
create your own strategy.

Class:
---
>>> StrategyClass
"""

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from . import exception
from . import utils

class StrategyClass(ABC):
    """
    StrategyClass.
    ----
    This is the class you have to inherit to create your strategy.

    Example:
    --
    >>> class Strategy(backpy.StrategyClass)

    To use the functions use the self instance.
    Create your strategy within the StrategyClass.next() structure.

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
    def __init__(self, data:pd.DataFrame = pd.DataFrame(), 
                 trades_cl:pd.DataFrame = pd.DataFrame(), 
                 trades_ac:pd.DataFrame = pd.DataFrame(),
                 commission:float = 0, init_funds:int = 0) -> None: 
        """
        __init__
        ----
        Builder.
        
        Parameters:
        --
        >>> data:pd.DataFrame = pd.DataFrame()
        >>> trades_cl:pd.DataFrame = pd.DataFrame()
        >>> trades_ac:pd.DataFrame = pd.DataFrame()
        
        data:
          All data from the step and previous ones.
        trades_cl:
          Closed trades.
        trades_ac:
          Open trades.
        commission:
          Commission per trade.
        
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
        if not type(data) is pd.DataFrame: 
            raise exception.StyClassError('Data is empty.')

        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.date = None

        self.__data = pd.DataFrame()

        if not data.empty:
            self.__data_updater(data=data)

        self.__commission = commission
        self.__init_funds = init_funds

        self.__trade = pd.DataFrame()
        self.__trades_ac = trades_ac
        self.__trades_cl = trades_cl

       

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
    
    def __data_updater(self, data:pd.DataFrame) -> None:
        """
        Data updater.
        ----
        All data updater.

        Parameters:
        --
        >>> data:pd.DataFrame

        data:
          - All data from the step and previous ones.
        """
        if data.empty:
            raise exception.StyClassError('Data is empty.')

        self.open = data["Open"].iloc[-1]
        self.high = data["High"].iloc[-1]
        self.low = data["Low"].iloc[-1]
        self.close = data["Close"].iloc[-1]
        self.volume = data["Volume"].iloc[-1]
        self.date = data.index[-1]

        self.__data = data
        self.__trade = pd.DataFrame()

    def __before(self, data=pd.DataFrame()):
        """
        Before.
        ----
        This function is used to run trades and other things.

        Parameters:
        --
        >>> data:pd.DataFrame = pd.DataFrame()

        data:
          - Data from the current and previous steps.
        """
        if not data.empty:
            self.__data_updater(data=data)
        
        self.next()

        # Check if a trade needs to be closed.
        self.__trades_ac.apply(lambda row: self.__act_close(index=row.name) 
                            if (not row['Type'] and 
                            (self.__data["Low"].iloc[-1] <= row['TakeProfit'] 
                            or self.__data["High"].iloc[-1] >= row['StopLoss']))
                            or (row['Type'] and 
                            (self.__data["High"].iloc[-1] >= row['TakeProfit'] 
                            or self.__data["Low"].iloc[-1] <= row['StopLoss'])) 
                            else None, axis=1) 

        # Concat new trade.
        if (not self.__trade.empty and 
            np.isnan(self.__trade['StopLoss'].iloc[0]) and 
            np.isnan(self.__trade['TakeProfit'].iloc[0])):

            self.__trades_cl = pd.concat([self.__trades_cl, self.__trade], 
                                         ignore_index=True)
            self.__trades_cl.reset_index(drop=True, inplace=True)
        elif not self.__trade.empty: 
            self.__trades_ac = pd.concat([self.__trades_ac, self.__trade], 
                                         ignore_index=True)

        self.__trades_ac.reset_index(drop=True, inplace=True)

        return self.__trades_ac, self.__trades_cl

    def prev(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev.
        ----
        Returns the data from the previous steps.
        Columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'index'.

        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        
        label:
          - Data column, if you leave it at None all columns will be returned.
          - If you leave 'index', all indexes will be returned, 
           ignoring the last parameter.

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
        """
        __data = self.__data
        if label == 'index': 
            return __data.index
        elif label != None:
            __data = __data[label]

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))
        
        return __data.iloc[len(__data)-last 
                           if last != None and last < len(__data) else 0:]
    
    def prev_trades_cl(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades closed.
        ----
        Returns the data from the closed trades.
        Columns: 'Date', 'Close', 'Low', 'High', 
        'StopLoss', 'TakeProfit', 'PositionClose', 
        'PositionDate', 'Amount', 'ProfitPer', 'Profit', 'Type', 'index'.

        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        
        label:
          - Data column, if you leave it at None all columns will be returned.
          - If you leave 'index', all indexes will be returned, 
           ignoring the last parameter.
          
        last:
          - How much data starting from the present backwards
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.

        Info:
        --
        '__trades_cl' columns, the same columns you can 
        access with prev_trades_ac.

        Date:
          The step date where trade began.
        Close:
          The 'Close' of the step when the trade began.
        Low:
          The 'Low' of the step when the trade began.
        High:
          The 'High' of the step when the trade began.
        StopLoss:
          The stoploss position.
        TakeProfit:
          The takeprofit position.
        PositionClose:
          The 'Close' of the step in which the trade ends.
        PositionDate:
          The step date where trade ends.
        Amount:
          Chosen amount.
        ProfitPer:
          Trade profit in percentage.
        Profit:
          Trade profit based on amount.
        Type:
          Type of trade.
        """
        __trades_cl = self.__trades_cl

        if label == 'index': return __trades_cl.index
        elif __trades_cl.empty: return pd.DataFrame()
        elif label != None: __trades_cl = __trades_cl[label]

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        return __trades_cl.iloc[len(__trades_cl)-last 
                                if last != None and 
                                last < len(__trades_cl) else 0:] 
    
    def prev_trades_ac(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades active.
        ----
        Returns the data from the active trades.
        Columns: 'Date', 'Close', 'Low', 'High', 
        'StopLoss', 'TakeProfit', 'Amount', 'Type', 'index'.

        Parameters:
        --
        >>> label:str = None
        >>> last:int = None
        
        label:
          - Data column, if you leave it at None all columns will be returned.
          - If you leave 'index', all indexes will be returned, 
           ignoring the 'last' parameter.
          
        last:
          - How much data starting from the present backwards
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.

        Info:
        --
        '__trades_ac' columns, the same columns you can 
        access with prev_trades_cl.

        Date:
          The step date where trade began.
        Close:
          The 'Close' of the step when the trade began.
        Low:
          The 'Low' of the step when the trade began.
        High:
          The 'High' of the step when the trade began.
        StopLoss:
          The stoploss position.
        TakeProfit:
          The takeprofit position.
        PositionClose:
          The 'Close' of the step in which the trade ends.
        PositionDate:
          The step date where trade ends.
        Amount:
          Chosen amount.
        ProfitPer:
          Trade profit in percentage.
        Profit:
          Trade profit based on amount.
        Type:
          Type of trade.
        """
        __trades_ac = self.__trades_ac
        if label == 'index': return __trades_ac.index
        elif __trades_ac.empty: return pd.DataFrame()
        elif label != None: __trades_ac = __trades_ac[label]

        if (last != None and 
              (last <= 0 or last > self.__data["Close"].shape[0])): 
                raise ValueError(utils.text_fix("""
                                Last has to be less than the length of 
                                'data' and greater than 0.
                                """, newline_exclude=True))

        return __trades_ac.iloc[len(__trades_ac)-last 
                                if last != None and 
                                last < len(__trades_ac) else 0:]
    
    def idc_hvolume(self, start:int = 0, end:int = None, 
                    bar:int = 10) -> pd.DataFrame:
        """
        Indicator horizontal volume.
        ----
        Return a pd.DataFrame with the position of each bar and the volume.
        Columns: 'Pos','H_Volume'.

        Alert:
        --
        Using this function with the 'end' parameter set to None
        is not recommended and may cause slowness in the program.

        Parameters:
        --
        >>> start:int = 0
        >>> end:int = None
        >>> bar:int = 10
        
        start:
          - Counting from now onwards, when you want the data capture to start 
           to return the horizontal volume.
          
        end:
          - Counting from now onwards, when you want the data capture to end to 
           return the horizontal volume.
          - If left at None the data will be captured from the beginning.
        
        bar:
          - The number of horizontal volume bars 
          (the more bars, the more precise).
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
        data_range.apply(vol_calc, axis=1); result["H_Volume"] += bar_list

        return result
        
    def idc_ema(self, length:int = any, 
                source:str = 'Close', last:int = None) -> np.array:
        """
        Exponential moving average.
        ----
        Returns an pd.Series with all the steps of an ema 
        with the length you indicate.

        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Ema length.
        
        source:
          - Allowed parameters: ('Close','Open','High','Low','Volume').
        
        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_ema(length=length, source=source, last=last)

    def __idc_ema(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None) -> np.array:
        """
        Exponential moving average.
        ----
        Returns an pd.Series with all the steps of an ema 
        with the length you indicate.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of ema with your own data.
        """
        data = self.__data[source] if data is None else data
        ema = data.ewm(span=length, adjust=False).mean()
    
        return np.flip(ema[len(ema)-last 
                           if last != None and last < len(ema) else 0:])
    
    def idc_sma(self, length:int = any, source:str = 'Close', last:int = None):
        """
        Simple moving average.
        ----
        Return an pd.Series with all the steps of an sma 
        with the length you indicate.

        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Sma length.

        source:
          - Allowed parameters: ('Close','Open','High','Low','Volume').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_sma(length=length, source=source, last=last)
    
    def __idc_sma(self, data:pd.Series = None, length:int = any, 
                  source:str = 'Close', last:int = None):
        """
        Simple moving average.
        ----
        Return an pd.Series with all the steps of an sma 
        with the length you indicate.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of sma with your own data.
        """
        data = self.__data[source] if data is None else data
        sma = data.rolling(window=length).mean()

        return np.flip(sma[len(sma)-last 
                           if last != None and last < len(sma) else 0:])
    
    def idc_wma(self, length:int = any, source:str = 'Close', 
                invt_weight:bool = False, last:int = None):
        """
        Linear weighted moving average.
        ----
        Return an pd.Series with all the steps of an wma 
        with the length you indicate.

        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> invt_weight:bool = False
        >>> last:int = None
        
        length:
          - Wma length.

        source:
          - Allowed parameters: ('Close','Open','High','Low','Volume').

        invt_weight:
          - The distribution of weights is done the other way around.

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_wma(length=length, source=source, 
                              invt_weight=invt_weight, last=last)
    
    def __idc_wma(self, data:pd.Series = None, 
                  length:int = any, source:str = 'Close', 
                  invt_weight:bool = False, last:int = None):
        """
        Linear weighted moving average.
        ----
        Return an pd.Series with all the steps of an wma 
        with the length you indicate.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of wma with your own data.
        """
        data = self.__data[source] if data is None else data

        weight = (np.arange(1, length+1)[::-1] 
                  if invt_weight else np.arange(1, length+1))
        wma = data.rolling(window=length).apply(
            lambda x: (x*weight).sum() / weight.sum(), raw=True)

        return np.flip(wma[len(wma)-last 
                           if last != None and last < len(wma) else 0:])
    
    def idc_smma(self, length:int = any, 
                 source:str = 'Close', last:int = None):
        """
        Smoothed moving average.
        ---- 
        Returns an pd.Series with all the steps of an smma 
        with the length you indicate.

        Parameters:
        --
        >>> length:int = any
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Smma length.

        source:
          - Allowed parameters: ('Close','Open','High','Low','Volume').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_smma(length=length, source=source, last=last)
    
    def __idc_smma(self, data:pd.Series = None, length:int = any, 
                   source:str = 'Close', last:int = None):
        """
        Smoothed moving average.
        ---- 
        Returns an pd.Series with all the steps of an smma 
        with the length you indicate.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of ema with your own data.
        """
        data = self.__data[source] if data is None else data
        smma = data.ewm(alpha=1/length, adjust=False).mean()

        return np.flip(smma[len(smma)-last 
                            if last != None and last < len(smma) else 0:])
    
    def idc_smema(self, length:int = 9, method:str = 'sma', 
                  smooth:int = 5, only:bool = False, 
                  source:str = 'Close', last:int = None):
        """
        Smoothed exponential moving average.
        ---- 
        Return an pd.DataFrame with the value of ema and 
        the smoothed ema for each step.
        Columns: 'ema','smoothed'.

        Parameters:
        --
        >>> length:int = any
        >>> method:str = 'sma'
        >>> smooth:int = 5
        >>> only:bool = False
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Ema length.

        method:
          - Smooth method.

        smooth:
          - 'method' length.

        only:
          - If left true, only one pd.Series will be returned with 
           the values ​​of 'method'.

        ma_type:
          - Ma type.

        source:
          - Allowed parameters: ('Close','Open','High','Low','Volume').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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

        # Smema calc.
        return self.__idc_smema(length=length, method=method, smooth=smooth, 
                                only=only, source=source, last=last)
    
    def __idc_smema(self, data:pd.Series = None, length:int = 9, 
                    method:str = 'sma', smooth:int = 5, only:bool = False, 
                    source:str = 'Close', last:int = None):
        """
        Smoothed exponential moving average.
        ---- 
        Return an pd.DataFrame with the value of ema and 
        the smoothed ema for each step.
        Columns: 'ema','smoothed'.

        Hidden function to prevent user modification.
        Function without exception handling.

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

        if only: 
            smema = np.flip(smema)
            return np.flip(smema[len(smema)-last 
                                 if last != None and last < len(smema) else 0:])
        
        smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)

        return smema.apply(
            lambda col: col.iloc[len(smema.index)-last 
                                 if last != None and 
                                 last < len(smema.index) else 0:], axis=0)

    def idc_bb(self, length:int = 20, std_dev:float = 2, 
               ma_type:str = 'sma', source:str = 'Close', last:int = None):
        """
        Bollinger bands.
        ----
        Return an pd.DataFrame with the value of the upper band, 
        the base ma and the position of the lower band for each step.
        Columns: 'Upper','{ma_type}','Lower'.

        Parameters:
        --
        >>> length:int = any
        >>> std_dev:float = 2
        >>> ma_type:str = 'sma'
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Window length.

        std_dev:
          - Standard deviation.

        ma_type:
          - Ma type.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_bb(length=length, std_dev=std_dev, 
                             ma_type=ma_type, source=source, last=last)
    
    def __idc_bb(self, data:pd.Series = None, length:int = 20, 
                 std_dev:float = 2, ma_type:str = 'sma', 
                 source:str = 'Close', last:int = None):
        """
        Bollinger bands.
        ----
        Return an pd.DataFrame with the value of the upper band, 
        the base ma and the position of the lower band for each step.
        Columns: 'Upper','Base','Lower'.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of bb with your own data.
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

        return bb.apply(lambda col: col.iloc[len(bb.index)-last 
                                             if last != None and 
                                             last < len(bb.index) else 0:], 
                                             axis=0)

    def idc_rsi(self, length_rsi:int = 14, length:int = 14, 
                rsi_ma_type:str = 'smma', base_type:str = 'sma', 
                bb_std_dev:float = 2, source:str = 'Close', last:int = None):
        """
        Relative strength index.
        ----
        Return an pd.DataFrame with the value of rsi and 
        'base_type' for each step.
        Columns: 'rsi',('base_type').

        Parameters:
        --
        >>> length_rsi:int 14
        >>> length:int = 14
        >>> rsi_ma_type:str = 'wma'
        >>> base_type:str = 'sma'
        >>> bb_std_dev:float = 2
        >>> source:str = 'Close'
        >>> last:int = None
        
        length_rsi:
          - Window length of 'rsi_ma_type'.

        length:
          - Window length of 'base_type'.

        rsi_ma_type:
          - Type of ma used for calculating rsi.

        base_type:
          - Type of ma base used applied to rsi.

        bb_std_dev:
          - Standard deviation for bb.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_rsi(length_rsi=length_rsi, length=length, 
                              rsi_ma_type=rsi_ma_type, base_type=base_type, 
                              bb_std_dev=bb_std_dev, source=source, last=last)

    def __idc_rsi(self, data:pd.Series = None, length_rsi:int = 14, 
                  length:int = 14, rsi_ma_type:str = 'wma', 
                  base_type:str = 'sma', bb_std_dev:float = 2, 
                  source:str = 'Close', last:int = None):
        """
        Relative strength index.
        ----
        Return an pd.DataFrame with the value of rsi and 
        'base_type' for each step.
        Columns: 'rsi',('base_type').

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of rsi with your own data.
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
        rsi = np.flip(100 - (100 / (1+ma_gain/ma_loss)))

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
                       length_d:int = 3, d_type:int = 'sma', 
                       source:str = 'Close', last:int = None):
        """
        Stochastic.
        ----
        Return an pd.DataFrame with the value of stochastic and 
        'd_type' for each step.
        Columns: 'stoch',('d_type').

        Parameters:
        --
        >>> length_k:int 14
        >>> smooth_k:int = 1
        >>> length_d:int = 3
        >>> d_type:int = 'sma'
        >>> source:str = 'Close'
        >>> last:int = None
        
        length_k:
          - Window length to calculate 'stoch'.

        smooth_k:
          - Window length of 'stoch'.

        length_d:
          - Window length of 'd_type'.

        d_type:
          - Type of ma base used applied to stochastic.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_stochastic(length_k=length_k, smooth_k=smooth_k, 
                                     length_d=length_d, d_type=d_type, 
                                     source=source, last=last)

    def __idc_stochastic(self, data:pd.Series = None, length_k:int = 14, 
                         smooth_k:int = 1, length_d:int = 3, d_type:int = 'sma', 
                         source:str = 'Close', last:int = None):
        """
        Stochastic.
        ----
        Return an pd.DataFrame with the value of stochastic and 
        'd_type' for each step.
        Columns: 'stoch',('d_type').

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
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
                only:bool = False, last:int = None):
        """
        Average directional index.
        ----
        Return an pd.Dataframe with the value of adx, +di and
        -di for each step.
        Columns: 'adx','+di','-di'.

        Parameters:
        --
        >>> smooth:int = 14
        >>> length_di:int = 14
        >>> only:bool = False
        >>> last:int = None
        
        smooth:
          - Smooth length.

        length_di:
          - Window length for calculate 'di'.

        only:
          - If left true, only one pd.Series will be returned with 
           the values ​​of adx.

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_adx(smooth=smooth, length_di=length_di, 
                              only=only, last=last)

    def __idc_adx(self, data:pd.Series = None, smooth:int = 14, 
                  length_di:int = 14, only:bool = False, last:int = None):
        """
        Average directional index.
        ----
        Return an pd.Dataframe with the value of adx, +di and 
        -di for each step.
        Columns: 'adx','+di','-di'.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
        """
        data = self.__data if data is None else data

        atr = self.__idc_atr(length=length_di, smooth='smma')

        dm_p = data['High'].diff()
        dm_n = -data['Low'].diff()

        di_p = 100 * np.flip(self.__idc_smma(data=dm_p.where(dm_p >= 0, 0), 
                                             length=length_di)/atr)
        di_n = 100 * np.flip(self.__idc_smma(data=dm_n.where(dm_n >= 0, 0), 
                                             length=length_di)/atr)
        adx = self.__idc_smma(data=100 * np.abs((di_p - di_n) / (di_p + di_n)), 
                              length=smooth)

        if only: 
            adx = np.flip(adx) 
            return np.flip(adx[len(adx)-last 
                               if last != None and last < len(adx) else 0:])
        adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

        return adx.apply(
            lambda col: col.iloc[len(adx.index)-last 
                                 if last != None and 
                                 last < len(adx.index) else 0:], axis=0) 

    def idc_macd(self, short_len:int = 12, long_len:int = 26, 
                 signal_len:int = 9, macd_ma_type:str = 'ema', 
                 signal_ma_typ:str = 'ema', histogram:bool = True, 
                 source:str = 'Close', last:int = None):
        """
        Convergence/divergence of the moving average.
        ----
        Return an pd.Dataframe with the value of macd and 
        signal for each step.
        Columns: 'macd','signal',('histogram' if histogram is True).

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
        
        short_len:
          - Short ma length.
          - The short ma is used to calculate macd.

        long_len:
          - Long ma length.
          - The long ma is used to calculate macd.

        signal_len:
          - Signal ma length.
          - The signal ma is the smoothed macd.

        macd_ma_type:
          - Type of ma to calculate macd.

        signal_ma_typ:
          - Type of ma to smooth macd.

        histogram:
          - An extra column will be returned with the histogram.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        elif not signal_ma_typ in ('ema','sma'): 
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
        return self.__idc_macd(short_len=short_len, long_len=long_len, 
                               signal_len=signal_len, macd_ma_type=macd_ma_type, 
                               signal_ma_typ=signal_ma_typ, histogram=histogram, 
                               source=source, last=last)

    def __idc_macd(self, data:pd.Series = None, short_len:int = 12, 
                   long_len:int = 26, signal_len:int = 9, 
                   macd_ma_type:str = 'ema', signal_ma_typ:str = 'ema', 
                   histogram:bool = True, source:str = 'Close', 
                   last:int = None):
        """
        Convergence/divergence of the moving average.
        ----
        Return an pd.Dataframe with the value of macd and 
        signal for each step.
        Columns: 'macd','signal',('histogram' if histogram is True).

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
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
                   source:str = 'Close', last:int = None):
        """
        Squeeze momentum.
        ----
        This function calculates the Squeeze Momentum, which is inspired by the 
        Squeeze Momentum Indicator available on TradingView. 
        While the concept is based on the original indicator,
        please note that this implementation may not fully replicate the exact 
        functionality of the original.
        Credit for the concept of the Squeeze Momentum goes to 
        its original developer.
        This function is an adaptation and 
        may differ from the original version.
        Please note that this function is intended for use in backtesting 
        scenarios, where it may utilize real market data or 
        simulated random data.
        It should be used for research and educational purposes only, 
        and should not be considered as financial advice.

        Return:
        ----
        Return an pd.Dataframe with the value of sqz and 
        histogram for each step.
        Columns: 'sqzmom','histogram'.

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
        
        bb_len:
          - Bollinger band length.

        bb_mult:
          - Bollinger band standard deviation.

        kc_len:
          - Kc length.

        kc_mult:
          - Kc standard deviation.

        use_tr:
          - If left false, ('High'-'Low') will be used 
           instead of the true range.

        histogram_len:
          - How many steps from the present backward 
           do you want the histogram to be calculated.
          - The higher the number, the less efficient.
          - If you leave it at 0, the 'historiogram' column 
           will not be returned.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_sqzmom(bb_len=bb_len, bb_mult=bb_mult, 
                                 kc_len=kc_len, kc_mult=kc_mult, 
                                 use_tr=use_tr, histogram_len=histogram_len, 
                                 source=source, last=last)

    def __idc_sqzmom(self, data:pd.Series = None, 
                     bb_len:int = 20, bb_mult:float = 1.5, 
                     kc_len:int = 20, kc_mult:float = 1.5, 
                     use_tr:bool = True, histogram_len:int = 50, 
                     source:str = 'Close', last:int = None):
        """
        Squeeze momentum.
        ----
        This function calculates the Squeeze Momentum, which is inspired by the 
        Squeeze Momentum Indicator available on TradingView. 
        While the concept is based on the original indicator,
        please note that this implementation may not fully replicate the exact 
        functionality of the original.
        Credit for the concept of the Squeeze Momentum goes to 
        its original developer.
        This function is an adaptation and 
        may differ from the original version.
        Please note that this function is intended for use in backtesting 
        scenarios, where it may utilize real market data or 
        simulated random data.
        It should be used for research and educational purposes only, 
        and should not be considered as financial advice.

        Return:
        ----
        Return an pd.Dataframe with the value of sqzmom and 
        histogram for each step.
        Columns: 'sqzmom','histogram'.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
        """
        data = self.__data if data is None else data

        basis = np.flip(self.__idc_sma(length=bb_len))
        dev = bb_mult * data[source].rolling(window=bb_len).std(ddof=0)

        upper_bb = basis + dev
        lower_bb = basis - dev

        ma = np.flip(self.__idc_sma(length=kc_len))
        range_ = np.flip(self.__idc_sma(data=np.flip(self.__idc_trange()) 
                                        if use_tr else data['High']-data['Low'], 
                                        length=kc_len))
        
        upper_kc = ma + range_ * kc_mult
        lower_kc = ma - range_ * kc_mult

        sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

        if histogram_len < 1: 
            result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index)})
            return result.apply(
                lambda col: col.iloc[len(result.index)-last 
                                     if last != None and 
                                     last < len(result.index) else 0:], axis=0)
        elif histogram_len > last: histogram = last
        
        histogram_len += kc_len
        d = data[source] - ((data['Low'].rolling(window=kc_len).min() + 
                             data['High'].rolling(window=kc_len).max()) / 2 + 
                             np.flip(self.__idc_sma(length=kc_len))) / 2
        histogram = self.__idc_rlinreg(data=d[len(d.index)-histogram_len 
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
                      length:int = 5, offset:int = 1):
        """
        Rolling linear regression.
        ----
        This function is not very efficient. 
        I recommend that the data does not exceed 50 in length.
        Return an pd.Series with the value of each linear regression.
        Calculated linear regression: m * (length - 1 - offset) + b
        Hidden function to prevent user modification.
        Function without exception handling.

        Parameters:
        --
        >>> data:pd.Series = None
        >>> length:int = 5
        >>> offset:int = 1
        
        data:
          - You can do the calculation of stochastic with your own data.

        length:
          - Length of each window.
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
        Return an pd.Series with all the steps of momentum 
        with the length you indicate.

        Parameters:
        --
        >>> length:int = 10
        >>> source:str = 'Close'
        >>> last:int = None
        
        length:
          - Length to calculate momentum.

        source:
          - Allowed parameters: ('Close','Open','High','Low').

        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_mom(length=length, source=source, last=last)

    def __idc_mom(self, data:pd.Series = None, length:int = 10, 
                  source:str = 'Close', last:int = None):
        """
        Momentum.
        ----
        Return an pd.Series with all the steps of momentum 
        with the length you indicate.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
        """
        data = self.__data if data is None else data
        mom = data[source] - data[source].shift(length)

        return np.flip(mom[len(mom)-last 
                           if last != None and last < len(mom) else 0:])

    def idc_ichimoku(self, tenkan_period:int = 9, kijun_period=26, 
                     senkou_span_b_period=52, ichimoku_lines:bool = True, 
                     last:int = None):
        """
        Ichimoku cloud.
        ----
        Return an pd.Dataframe with the value of ichimoku cloud, 
        tenkan_sen and kijun_sen for each step.
        Columns: 'senkou_a','senkou_b'.
        If ichimoku lines is true these columns are added to the dataframe.
        Added: 'tenkan_sen','kijun_sen'.

        Parameters:
        --
        >>> tenkan_period:int = 12
        >>> kijun_period:int = 26
        >>> senkou_span_b_period:int = 9
        >>> ichimoku_lines:str = 'ema'
        >>> last:int = None
        
        tenkan_period:
          - Window length to calculate tenkan.

        kijun_period:
          - Window length to calculate kijun.

        senkou_span_b_period:
          - Window length to calculate senkou span.

        ichimoku_lines:
          - If ichimoku lines is true these columns are added to the dataframe: 
           'tenkan_sen','kijun_sen'.
        last:
          - How much data starting from the present backwards 
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_ichimoku(tenkan_period=tenkan_period, 
                                   kijun_period=kijun_period, 
                                   senkou_span_b_period=senkou_span_b_period, 
                                   ichimoku_lines=ichimoku_lines, 
                                   last=last)

    def __idc_ichimoku(self, data:pd.Series = None, tenkan_period:int = 9, 
                       kijun_period=26, senkou_span_b_period=52, 
                       ichimoku_lines:bool = True, last:int = None):
        """
        Ichimoku cloud.
        ----
        Return an pd.Dataframe with the value of ichimoku cloud, 
        tenkan_sen and kijun_sen for each step.
        Columns: 'senkou_a','senkou_b'.

        If ichimoku lines is true these columns are added to the dataframe.
        Added: 'tenkan_sen','kijun_sen'.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
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
                      met:bool = False, source:str = 'Low/High'):
        """
        Fibonacci retracement.
        ----
        Return an pd.DataFrame with the fibonacci levels and values.
        Columns: 'Level','Value'.

        Parameters:
        --
        >>> start:int = None
        >>> end:int = 30
        >>> met:bool = False
        >>> source:str = 'Low/High'
        
        start:
          - Where do you want level 0 to be.
          - If 'met' is false 'start' is the number of candles back to 
           open the fibonacci.
          - None == 0.

        end:
          - Where do you want level 1 to be.
          - If 'met' is false 'end' is the number of candles back to 
           close the fibonacci.
          - None == 0.

        met:
          - If left false, 'start' and 'end' are 
           the number of candles backwards,
           otherwise 'start' and 'end' are 
           the value from which Fibonacci opens.

        source:
          - Data that is extracted.
          - Start and end data format: 's/s where 's' is each source.'
          - Values supported in source: ('Close', 'Open', 'High', 'Low')
          - If left 'met' in true 'source' does not work.
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
        return self.__idc_fibonacci(
            start=data_start.iloc[len(data_start)-start-1 
                                  if start != None and 
                                  start < len(data_start) else 0], 
            end=data_end.iloc[len(data_end)-end-1 
                              if end != None and end < len(data_end) else 0])

    def __idc_fibonacci(self, start:int = 10, end:int = 1):
        """
        Fibonacci retracement.
        ----
        Return an pd.DataFrame with the fibonacci levels and values.
        Columns: 'Level','Value'.

        Hidden function to prevent user modification.
        Function without exception handling.
        """
        fibo_levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 
                                0.786, 1, 1.618, 2.618, 3.618, 4.236])

        return pd.DataFrame({'Level':fibo_levels,
                             'Value':(start + (end - start) * fibo_levels)})

    def idc_atr(self, length:int = 14, smooth:str = 'wma', last:int = None):
        """
        Average true range.
        ----
        Return an pd.Series with the value of 
        average true range for each step.

        Parameters:
        --
        >>> length:int = 14
        >>> smooth:str = 'wma'
        >>> last:int = None
        
        length:
          - Window length to smooth 'atr'.

        smooth:
          - Ma used to smooth 'atr'.

        last:
          - How much data starting from the present backwards
           do you want to be returned.
          - If you leave it at None, the data for all times is returned.
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
        return self.__idc_atr(length=length, smooth=smooth, last=last)
    
    def __idc_atr(self, length:int = 14, smooth:str = 'wma', last:int = None):
        """
        Average true range.
        ----
        Return an pd.Series with the value of 
        average true range for each step.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
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
        Return an pd.Series with the value of true range for each step.

        Hidden function to prevent user modification.
        Function without exception handling.

        Data parameter:
        --
        You can do the calculation of stochastic with your own data.
        """
        data = self.__data if data is None else data

        close = data['Close'].shift(1)
        close.fillna(data['Low'], inplace=True)

        hl = data['High'] - data['Low']
        hyc = abs(data['High'] - close)
        lyc = abs(data['Low'] - close)
        tr = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)
        return np.flip(tr[len(tr)-last 
                          if last != None and last < len(tr) else 0:])

    def act_open(self, type:int = 1, stop_loss:int = np.nan, 
                 take_profit:int = np.nan, amount:int = np.nan) -> None:
        """
        Open action.
        ----
        Open an action.

        Warning:
        --
        If you leave the stop loss and take profit in np.nan your trade 
        will be counted as closed and you can't modify it or close it.

        Parameters:
        --
        >>> type:int = 1
        >>> stop_loss:int = np.nan
        >>> take_profit:int = np.nan
        >>> amount:int = np.nan
        
        type:
          - 0 is a sell position, 1 is a buy position.

        stop_loss:
          - The price where the stop loss will go, 
           if it is left at np.nan the position is opened without stop loss.

        take_profit:
          - The price where the take profit will go, 
           if it is left at np.nan the position is opened without take profit.

        amount:
          - The amount of imaginary points with which 
           you would enter the position.

        Info:
        --
        '__trade' columns, the same columns you can access with prev_trades.

        Date:
          The step date where trade began.
        Close:
          The 'Close' of the step when the trade began.
        Low:
          The 'Low' of the step when the trade began.
        High:
          The 'High' of the step when the trade began.
        StopLoss:
          The stoploss position.
        TakeProfit:
          The takeprofit position.
        PositionClose:
          The 'Close' of the step in which the trade ends.
        PositionDate:
          The step date where trade ends.
        Amount:
          Chosen amount.
        ProfitPer:
          Trade profit in percentage.
        Profit:
          Trade profit based on amount.
        Type:
          Type of trade.
        """
        # Check if type is 1 or 0.
        if not type in {1,0}: 
            raise exception.ActionError("'type' only 1 or 0.")
        # Check exceptions.
        if amount <= 0: 
            raise exception.ActionError("'amount' can only be greater than 0.")
        if ((type and (self.__data["Close"].iloc[-1] <= stop_loss or 
                       self.__data["Close"].iloc[-1] >= take_profit)) or 
            (not type and (self.__data["Close"].iloc[-1] >= stop_loss or 
                           self.__data["Close"].iloc[-1] <= take_profit))): 

            raise exception.ActionError(
                utils.text_fix("""
                               'stop_loss' or 'take_profit' 
                               incorrectly configured for the position type.
                               """, newline_exclude=True))
        # Create new trade.
        self.__trade = pd.DataFrame({'Date':self.__data.index[-1],
                                     'Close':self.__data["Close"].iloc[-1],
                                     'Low':self.__data["Low"].iloc[-1],
                                     'High':self.__data["High"].iloc[-1],
                                     'StopLoss':stop_loss,
                                     'TakeProfit':take_profit,
                                     'PositionClose':np.nan,
                                     'PositionDate':np.nan,
                                     'Amount':amount,
                                     'ProfitPer':np.nan,
                                     'Profit':np.nan,
                                     'Type':type},index=[1])

    def act_close(self, index:int = 0) -> None:
        """
        Close action.
        ----
        Close an action.

        Parameters:
        --
        >>> index:int = 0
        
        index:
          - The index of the active trade you want to close.
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
        Close action.
        ----
        Close an action.

        Hidden function to prevent user modification.
        Function without exception handling.
        """
        # Get trade to close.
        trade = self.__trades_ac.iloc[lambda x: x.index==index].copy()
        self.__trades_ac = self.__trades_ac.drop(trade.index)
        # Get PositionClose
        take = trade['TakeProfit'].iloc[0];stop = trade['StopLoss'].iloc[0]
        position_close = ((stop if self.__data["Low"].iloc[-1] <= stop else take
                           if self.__data["High"].iloc[-1] >= take 
                           else self.__data["Close"].iloc[-1]) 
                           if trade['Type'].iloc[0] 
                           else (stop if self.__data["High"].iloc[-1] >= stop 
                                 else take 
                                 if self.__data["Low"].iloc[-1] <= take 
                                 else self.__data["Close"].iloc[-1]))
        # Fill data.
        trade['PositionClose'] = position_close
        trade['PositionDate'] = self.__data.index[-1]
        open = trade['Close'].iloc[0]
        trade['ProfitPer'] = ((position_close-open)/open*100 
                              if trade['Type'].iloc[0] 
                              else (open-position_close)/open*100)
        trade['Profit'] = (trade['Amount'].iloc[0]*trade['ProfitPer'].iloc[0]
                           /100-trade['Amount']*(self.__commission/100) 
                           if not np.isnan(trade['Amount'].iloc[0]) else np.nan)

        self.__trades_cl = pd.concat([self.__trades_cl,trade], 
                                     ignore_index=True) 
        self.__trades_cl.reset_index(drop=True, inplace=True)

    def act_mod(self, index:int = 0, new_stop:int = None, 
                new_take:int = None) -> None:
        """
        Modify action.
        ----
        Modify an action.

        Alert:
        --
        If an invalid stop loss or invalid takeprofit is submitted,
        the program will return None and will not execute any changes.

        Parameters:
        --
        >>> index:int = 0
        >>> new_stop:int = None
        >>> new_take:int = None
        
        index:
          - The index of the active trade you want to modify.

        new_stop:
          - The price where the new stop loss will be,
           if it is left at None the stop loss will not be modified and 
           if it is left at np.nan the stop loss will be removed.

        new_take:
          - The price where the new take profit will be,
           if it is left at None the take profit will not be modified and 
           if it is left at np.nan the take profit will be removed.
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
