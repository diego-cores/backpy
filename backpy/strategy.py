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
    """ 
    def __init__(self, data:pd.DataFrame = any, trades_cl:pd.DataFrame = pd.DataFrame(), trades_ac:pd.DataFrame = pd.DataFrame()) -> None: 
        """
        __init__
        ----
        """
        if data.empty: raise exception.StyClassError('Data is empty.')

        self.open = data["Open"].iloc[-1]
        self.high = data["High"].iloc[-1]
        self.low = data["Low"].iloc[-1]
        self.close = data["Close"].iloc[-1]
        self.volume = data["Volume"].iloc[-1]
        self.date = data.index[-1]

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
        This function is used to run trades.
        """

        self.next()

        # Check if a trade needs to be closed.
        self.__trades_ac.apply(lambda row: self.act_close(row.name) if self.__data["High"].iloc[-1] >= max(row['TakeProfit'],row['StopLoss']) or self.__data["Low"].iloc[-1] <= min(row['TakeProfit'],row['StopLoss']) else None, axis=1) 

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
        """
        __data = self.__data
        if label == 'index': return __data.index
        elif label != None: __data = __data[label]
        
        return __data.iloc[len(self.__data[label])-last if last != None and last < len(self.__data[label]) else 0:]
    
    def prev_trades_cl(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades closed.
        ----
        """
        __trades_cl = self.__trades_cl
        if label == 'index': return __trades_cl.index
        elif label != None: __trades_cl = __trades_cl[label]

        return self.__trades_cl.iloc[len(self.__trades_cl[label])-last if last != None and last < len(self.__trades_cl[label]) else 0:] if not self.__trades_cl.empty else None
    
    def prev_trades_ac(self, label:str = None, last:int = None) -> pd.DataFrame:
        """
        Prev of trades active.
        ----
        """
        __trades_ac = self.__trades_ac
        if label == 'index': return __trades_ac.index
        elif label != None: __trades_ac = __trades_ac[label]
        
        return self.__trades_ac.iloc[len(self.__trades_ac[label])-last if last != None and last < len(self.__trades_ac[label]) else 0:] if not self.__trades_ac.empty else None
        
    def idc_ema(self, period, last:int = None) -> np.array:
        """
        Indicator ema.
        ----
        """
        ema = self.__data["Close"].ewm(span=period, adjust=False).mean(); return ema[len(ema)-last if last != None and last < len(ema) else 0:]
        
    def idc_hvolume(self, start:int = 0, end:int = None, bar:int = 10) -> pd.DataFrame:
        """
        Indicator horizontal volume.
        ----
        """
        data_len = self.__data.shape[0]; data_range = self.__data.iloc[abs(data_len-end) if end != None and end < data_len else 0:abs(data_len-start) if start < data_len else data_len]

        if bar == 1: return pd.DataFrame({"H_Volume":data_range["Volume"].sum()}, index=[1])

        bar_list = np.array([0]*bar, dtype=np.int64)
        result = pd.DataFrame({"Pos":(data_range["High"].max()-data_range["Low"].min())/(bar-1) * range(bar) + data_range["Low"].min(),"H_Volume":bar_list})

        def vol_calc(row) -> None: bar_list[np.argmin(np.abs(np.subtract(result["Pos"].values, row["Low"]))):np.argmin(np.abs(np.subtract(result["Pos"].values, row["High"])))+1] += int(row["Volume"])
        data_range.apply(vol_calc, axis=1); result["H_Volume"] += bar_list

        return result
    
    def act_open(self, type:int = 1, stop_loss:int = np.nan, take_profit:int = np.nan, amount:int = np.nan) -> None:
        """
        Open action.
        ----
        """
        # Check if type is 1 or 0.
        if not type in {1,0}: raise ValueError("Only 1 or 0.")
        # Check exceptions.
        if (type and self.__data["Close"].iloc[-1] <= stop_loss and self.__data["Close"].iloc[-1] >= take_profit) or (not type and self.__data["Close"].iloc[-1] >= stop_loss and self.__data["Close"].iloc[-1] <= take_profit): raise ValueError
        # Create new trade.
        self.__trade = pd.DataFrame({'Date':self.__data.index[-1],'Close':self.__data["Close"].iloc[-1],'Low':self.__data["Low"].iloc[-1],'High':self.__data["High"].iloc[-1],'StopLoss':stop_loss,'TakeProfit':take_profit,'PositionClose':np.nan,'PositionDate':np.nan,'Amount':amount,'ProfitPer':np.nan,'Profit':np.nan,'Type':type},index=[1])
        #raise KeyError()

    def act_close(self, index:int = 0) -> None:
        """
        Close action.
        ----
        """
        # Check exceptions.
        if self.__trades_ac.empty: raise exception.ActionError('There are no active trades.')
        if not index in self.__trades_ac.index.to_list(): raise ValueError("Index does not exist.")
        # Get trade to close.
        trade = self.__trades_ac.iloc[lambda x: x.index==index]
        self.__trades_ac = self.__trades_ac.drop(trade.index)
        
        take = trade['TakeProfit'].iloc[0];stop = trade['StopLoss'].iloc[0]
        position_close = (take if self.__data["High"].iloc[-1] >= take else stop if self.__data["Low"].iloc[-1] <= stop else self.__data["Close"].iloc[-1]) if trade['Type'].iloc[0] else (take if self.__data["Low"].iloc[-1] <= take else stop if self.__data["High"].iloc[-1] >= stop else self.__data["Close"].iloc[-1])
        # Fill data.
        trade['PositionClose'] = position_close; trade['PositionDate'] = self.__data.index[-1]
        open = trade['Close'].iloc[0]

        trade['ProfitPer'] = (position_close-open)/open*100 if trade['Type'].iloc[0] else (open-position_close)/open*100
        trade['Profit'] = trade['Amount'].iloc[0]*trade['ProfitPer'].iloc[0]/100 if trade['Amount'].iloc[0] else np.nan

        self.__trades_cl = pd.concat([self.__trades_cl,trade], ignore_index=True) ; self.__trades_cl.reset_index(drop=True, inplace=True)

    def act_mod(self, index:int = 0, new_stop:int = None, new_take:int = None) -> None:
        """
        Modify action.
        ----
        """
        # Check exceptions.
        if self.__trades_ac.empty: raise exception.ActionError('There are no active trades.')
        if not (new_stop or new_take): raise ValueError
        # Get trade to modify.
        trade = self.__trades_ac.loc[lambda x: x.index==index]
        # Set new stop.
        if (new_stop < self.__data["Close"].iloc[-1] and trade['Type']) or (not trade['Type'] and new_stop > self.close): trade['StopLoss'] = new_stop 
        else: raise ValueError
        # Set new take.
        if (new_take > self.__data["Close"].iloc[-1] and trade['Type']) or (not trade['Type'] and new_take < self.close): trade['TakeProfit'] = new_take
        else: raise ValueError
