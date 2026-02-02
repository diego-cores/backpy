"""
Indicators module

This module contains the main logic of the indicators.

Note:
    These functions do not have exception handling.

Variables:
    logger (Logger): Logger variable.

Functions:
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
    idc_rlinreg: This function calculates the rolling linear regression.
    idc_mom: Calculates the Momentum indicator (MOM).
    idc_ichimoku: Calculates the Ichimoku indicator.
    idc_atr: Calculates the Average True Range (ATR).
    idc_trange: This function calculates the true range.
"""

import pandas as pd
import numpy as np
import logging

from . import _commons as _cm

logger:logging.Logger = logging.getLogger(__name__)

@_cm._store_decorator
def idc_fibonacci(lv0:int = 10, lv1:int = 1) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels.

    This function calculates the Fibonacci retracement levels.

    Args:
        lv0 (float, optional): Level 0 position.
        lv1 (float, optional): Level 1 position.

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

@_cm._store_decorator
def idc_ema(self, data:pd.Series | None = None, length:int = 10, 
                source:str = 'close', last:int | None = None, 
                cut:bool = False) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    This function calculates the EMA.

    Args:
        data (Series | None, optional): Series of data to perform the EMA calculation.
        length (int): The length of the EMA.
        source (str, optional): The data source for the EMA calculation. Allowed 
            parameters are 'close', 'open', 'high', 'low', and 'volume'.
        last (int | None, optional): Number of data points to return from the 
            present backwards. If None, returns data for all time.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper containing the EMA values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data
    ema = v_data.ewm(span=length, adjust=False).mean()

    return ema


@_cm._store_decorator
def idc_sma(self, data:pd.Series | None = None, length:int = 10, 
                source:str = 'close', last:int | None = None, 
                cut:bool = False) -> pd.Series:
    """
    Simple Moving Average (SMA).

    This function calculates the SMA.

    Args:
        data (Series | None, optional): Series of data to perform the SMA calculation.
        length (int): Length of the SMA.
        source (str, optional): Data source for SMA calculation. Allowed values are 
                        ('close', 'open', 'high', 'low', 'volume').
        last (int | None, optional): Number of data points to return from the present 
                                backwards. If None, returns data for all times.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper containing the SMA values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data
    sma = v_data.rolling(window=length).mean()

    return sma

@_cm._store_decorator
def idc_wma(self, data:pd.Series | None = None, 
                length:int = 10, source:str = 'close', 
                invt_weight:bool = False, last:int | None = None, 
                cut:bool = False) -> pd.Series:
    """
    Weighted Moving Average (WMA).

    This function calculates the WMA.

    Args:
        data (Series | None, optional): Series of data to perform the WMA calculation.
        length (int): Length of the WMA.
        source (str, optional): Data source for WMA calculation. Allowed values are 
                        ('close', 'open', 'high', 'low', 'volume').
        invt_weight (bool, optional): If True, the distribution of weights is reversed.
        last (int | None, optional): Number of data points to return from the present 
                                backwards. If None, returns data for all times.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper containing the WMA values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data

    weight = (np.arange(1, length+1)[::-1] 
                if invt_weight else np.arange(1, length+1))
    wma = v_data.rolling(window=length).apply(
        lambda x: (x*weight).sum() / weight.sum(), raw=True)

    return wma

@_cm._store_decorator
def idc_smma(self, data:pd.Series|None = None, length:int = 10, 
                source:str = 'close', last:int|None = None, 
                cut:bool = False) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).

    This function calculates the SMMA.

    Args:
        data (Series | None, optional): Series of data to perform the SMMA calculation.
        length (int): Length of the SMMA.
        source (str, optional): Data source for SMMA calculation. Allowed values are 
                        ('close', 'open', 'high', 'low', 'volume').
        last (int | None, optional): Number of data points to return from the present 
                                backwards. If None, returns data for all times.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper containing the SMMA values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data

    smma = v_data.ewm(alpha=1/length, adjust=False).mean()
    smma.shift(1)

    return smma

@_cm._store_decorator
def idc_sema(self, data:pd.Series | None = None, length:int = 9, 
                method:str = 'sma', smooth:int = 5, only:bool = False, 
                source:str = 'close', last:int | None = None, 
                cut:bool = False) -> pd.DataFrame|np.ndarray:
    """
    Smoothed Exponential Moving Average (SEMA).

    This function calculates the SEMA.

    Args:
        data (Series | None, optional): Series of data to perform the SEMA calculation.
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
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'ema'
        - 'smoothed'

    Returns:
        DataWrapper: DataWrapper containing 'ema' and 'smoothed' values for 
                        each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data
    ema = v_data.ewm(span=length, adjust=False).mean()

    match method:
        case 'sma': 
            smema = idc_sma(self, data=ema, length=smooth).unwrap()
        case 'ema': smema = idc_ema(self, data=ema, length=smooth).unwrap()
        case 'smma': smema = idc_smma(self, data=ema, length=smooth).unwrap()
        case 'wma': smema = idc_wma(self, data=ema, length=smooth).unwrap()
        case _: smema = idc_sma(self, data=ema, length=smooth).unwrap()

    if only: 
        smema = np.flip(smema)
        return np.flip(smema[len(smema)-last 
                                if last != None and last < len(smema) else 0:])
    
    smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)
    return smema

@_cm._store_decorator
def idc_bb(self, data:pd.Series | None = None, length:int = 20, 
                std_dev:float = 2, ma_type:str = 'sma', source:str = 'close', 
                last:int | None = None, cut:bool = False) -> pd.DataFrame:
    """
    Bollinger Bands (BB).

    This function calculates the BB.

    Args:
        data (Series | None, optional): Series of data to perform the Bollinger Bands 
            calculation.
        length (int, optional): Window length for calculating Bollinger Bands.
        std_dev (float, optional): Number of standard deviations for the bands.
        ma_type (str, optional): Type of moving average. For example, 'sma' for simple 
                        moving average.
        source (str, optional): Data source for calculation. Allowed values are 
                        ('close', 'open', 'high', 'low').
        last (int | None, optional): Number of data points to return from the present 
                                backwards. If None, returns data for all times.
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'upper'
        - '{ma_type}'
        - 'lower'

    Returns:
        DataWrapper: DataWrapper containing 'upper', '{ma_type}', and 'lower' 
                        values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data

    match ma_type:
        case 'sma': ma = idc_sma(self, data=v_data, length=length).to_series()
        case 'ema': ma = idc_ema(self, data=v_data, length=length).to_series()
        case 'wma': ma = idc_wma(self, data=v_data, length=length).to_series()
        case 'smma': ma = idc_smma(self, data=v_data, length=length).to_series()
        case _: ma = idc_sma(self, data=v_data, length=length).to_series()

    std_ = (std_dev * v_data.rolling(window=length).std())
    bb = pd.DataFrame({'upper':ma + std_,
                        ma_type:ma,
                        'lower':ma - std_}, index=ma.index)

    return bb

@_cm._store_decorator
def idc_rsi(self, data:pd.Series | None = None, length_rsi:int = 14, 
                length:int = 14, rsi_ma_type:str = 'smma', 
                base_type:str = 'sma', bb_std_dev:float = 2, 
                source:str = 'close', last:int | None = None, 
                cut:bool = False)  -> pd.DataFrame:
    """
    Relative Strength Index (RSI).

    This function calculates the RSI.

    Args:
        data (Series | None, optional): Series of data to perform the RSI calculation.
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
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'rsi'
        - '{base_type}'

    Returns:
        DataWrapper: DataWrapper containing 'rsi' and '{base_type}' values for 
                        each step.
    """

    delta = self._StrategyClass__data_adf[source].diff() if data is None else data.diff()

    ma = idc_sma
    match rsi_ma_type:
        case 'sma': ma = idc_sma
        case 'ema': ma = idc_ema
        case 'wma': ma = idc_wma
        case 'smma': ma = idc_smma

    ma_gain = ma(self, data = delta.where(delta > 0, 0), 
                    length=length_rsi, source=source).to_series()
    ma_loss = ma(self, data = -delta.where(delta < 0, 0), 
                    length=length_rsi, source=source).to_series()
    rsi = 100 - (100 / (1+ma_gain/ma_loss))

    match base_type:
        case 'sma': mv = idc_sma(self, data=rsi, length=length).to_series()
        case 'ema': mv = idc_ema(self, data=rsi, length=length).to_series()
        case 'wma': mv = idc_wma(self, data=rsi, length=length).to_series()
        case 'smma': mv = idc_smma(self, data=rsi, length=length).to_series()
        case 'bb': mv = idc_bb(self, data=rsi, length=length,
                                        std_dev=bb_std_dev).to_dataframe()
        case _: mv = idc_sma(self, data=rsi, length=length).to_series()

    if type(mv) == pd.Series: mv.name = base_type

    rsi:pd.DataFrame = pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

    return rsi

@_cm._store_decorator
def idc_stochastic(self, data:pd.DataFrame | None = None, length_k:int = 14, 
                        smooth_k:int = 1, length_d:int = 3, d_type:str = 'sma', 
                        source:str = 'close', last:int | None = None, 
                        cut:bool = False) -> pd.DataFrame:
    """
    Stochastic Oscillator.

    This function calculates the stochastic oscillator.

    Args:
        data (DataFrame | None, optional): Series of data to perform the stochastic calculation.
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
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'stoch'
        - '{d_type}'

    Returns:
        DataWrapper: DataWrapper containing 'stoch' and '{d_type}' values for each 
                        step.
    """

    v_data = self._StrategyClass__data_adf if data is None else data

    low_data = v_data.loc[:, 'low'].rolling(window=length_k).min()
    high_data = v_data.loc[:, 'high'].rolling(window=length_k).max()

    ma = idc_sma
    match d_type:
        case 'sma': ma = idc_sma
        case 'ema': ma = idc_ema
        case 'wma': ma = idc_wma
        case 'smma': ma = idc_smma

    stoch = (((v_data[source] - low_data) / 
                (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
    result = pd.DataFrame({'stoch':stoch, 
                            d_type:ma(self, data=stoch, length=length_d).to_series()})

    return result

@_cm._store_decorator
def idc_adx(self, data:pd.DataFrame | None = None, smooth:int = 14, 
                length_di:int = 14, only:bool = False, 
                last:int | None = None, cut:bool = False) -> pd.DataFrame:
    """
    Average Directional Index (ADX).

    This function calculates the ADX.

    Args:
        data (DataFrame | None, optional): Series of data to perform the ADX calculation.
        smooth (int, optional): Smoothing length. Default is 14.
        length_di (int, optional): Window length for calculating +DI and -DI. Default is 14.
        only (bool, optional): If True, returns only a Series with the ADX values.
        last (int | None, optional): Number of data points to return from the present 
                                backwards. If None, returns data for all times.
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'adx'
        - '+di'
        - '-di'

    Returns:
        DataWrapper: DataWrapper containing 'adx', '+di', and '-di' values for 
                        each step.
    """

    v_data = self._StrategyClass__data_adf if data is None else data

    atr = idc_atr(self, length=length_di, smooth='smma').unwrap()

    dm_p_raw = v_data.loc[:, 'high'].diff()
    dm_n_raw = -v_data.loc[:, 'low'].diff()
    
    dm_p = pd.Series(
        np.where((dm_p_raw > dm_n_raw) & (dm_p_raw > 0), dm_p_raw, 0), 
        index=v_data.index)
    dm_n = pd.Series(
        np.where((dm_n_raw > dm_p_raw) & (dm_n_raw > 0), dm_n_raw, 0), 
        index=v_data.index)

    di_p = 100 * idc_smma(self, dm_p, length=length_di).to_series() / atr
    di_n = 100 * idc_smma(self, dm_n, length=length_di).to_series() / atr

    adx = idc_smma(self,
        data=100 * np.abs((di_p - di_n) / (di_p + di_n).replace(0, 1)), 
        length=smooth).to_series()

    if only: 
        return adx
    adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

    return adx

@_cm._store_decorator
def idc_macd(self, data:pd.Series | None = None, short_len:int = 12, 
                long_len:int = 26, signal_len:int = 9, 
                macd_ma_type:str = 'ema', signal_ma_type:str = 'ema', 
                histogram:bool = True, source:str = 'close', 
                last:int | None = None, cut:bool = False) -> pd.DataFrame:
    """
    Calculate the convergence/divergence of the moving average (MACD).

    This function calculates the MACD.

    Args:
        data (Series | None, optional): The data used for calculation of MACD.
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
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'macd'
        - 'signal'
        - 'histogram'  

    Returns:
        DataWrapper: A DataWrapper with MACD values and signal line for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data

    macd_ma = idc_ema
    match macd_ma_type:
        case 'ema':
            macd_ma = idc_ema
        case 'sma':
            macd_ma = idc_sma

    signal_ma = idc_ema
    match signal_ma_type:
        case 'ema':
            signal_ma = idc_ema
        case 'sma':
            signal_ma = idc_sma
    
    short_ema = macd_ma(self, data=v_data, length=short_len).to_series()
    long_ema = macd_ma(self, data=v_data, length=long_len).to_series()
    macd = short_ema - long_ema

    signal_line = signal_ma(self, data=macd, length=signal_len).to_series()

    result = pd.DataFrame({'macd':macd, 'signal':signal_line, 
                            'histogram':macd-signal_line} 
                            if histogram else 
                            {'macd':macd, 'signal':signal_line})

    return result

@_cm._store_decorator
def idc_sqzmom(self, data:pd.DataFrame | None = None, 
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

    Args:
        data (DataFrame | None, optional): The data used for calculating the Squeeze Momentum.
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
        cut (bool, optional): True to return the trimmed data with current index.

    Columns:
        - 'sqzmom'
        - 'histogram'

    Returns:
        DataWrapper: A DataWrapper with Squeeze Momentum values and histogram for 
            each step.
    """

    v_data = self._StrategyClass__data_adf if data is None else data

    basis = idc_sma(self, length=bb_len).unwrap()
    dev = bb_mult * v_data.loc[:, source].rolling(window=bb_len).std(ddof=0)

    upper_bb = basis + dev
    lower_bb = basis - dev

    ma = idc_sma(self, length=kc_len).unwrap()
    range_ = idc_sma(self, data=idc_trange(self).to_series()
                            if use_tr else v_data['high']-v_data['low'], 
                            length=kc_len).unwrap()

    upper_kc = ma + range_ * kc_mult
    lower_kc = ma - range_ * kc_mult

    sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

    d = v_data[source] - ((v_data.loc[:, 'low'].rolling(window=kc_len).min() + 
                            v_data.loc[:, 'high'].rolling(window=kc_len).max()) / 2 + 
                            idc_sma(self, length=kc_len).unwrap()) / 2

    histogram = idc_rlinreg(self, data=d, length=kc_len, offset=0).unwrap()

    result = pd.DataFrame({'sqzmom':pd.Series(sqz, index=v_data.index), 
                            'histogram':histogram}, 
                            index=v_data.index)
    return result

@_cm._store_decorator
def idc_rlinreg(self, data:pd.Series | None = None, 
                source:str = 'close',
                length:int = 5, offset:int = 1,
                cut:bool = False) -> pd.Series:
    """
    Calculate rolling linear regression values.

    This function calculates the rolling linear regression.

    Args:
        data (Series | None, optional): The data used for linear regression calculations.
        source (str, optional): Data source for momentum calculation. Allowed values:
            'close', 'open', 'high', 'low'.
        length (int, optional): Length of each window for the rolling regression.
        offset (int, optional): Offset used in the regression calculation.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: Array with the linear regression values for each window.
    """

    v_data = self._StrategyClass__data_adf[source]  if data is None else data

    x = np.arange(length)
    y = v_data.rolling(window=length)

    m = y.apply(lambda y: np.polyfit(x, y.values, 1)[0])
    b = y.mean() - (m * float(np.mean(x))) 

    return m * (length - 1 - offset) + b

@_cm._store_decorator
def idc_mom(self, data:pd.Series | None = None, length:int = 10, 
                source:str = 'close', last:int | None = None,
                cut:bool = False) -> pd.Series:
    """
    Calculate momentum values (MOM).

    This function calculates the MOM.

    Args:
        data (Series | None, optional): The data used to calculate momentum.
        length (int, optional): Length for calculating momentum.
        source (str, optional): Data source for momentum calculation. Allowed values:
            'close', 'open', 'high', 'low'.
        last (int | None, optional): Number of data points to return starting from the
            present backward. If None, returns data for all available periods.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper with the momentum values for each step.
    """

    v_data = self._StrategyClass__data_adf[source] if data is None else data
    mom = v_data - v_data.shift(length)

    return mom

@_cm._store_decorator
def idc_ichimoku(self, data:pd.DataFrame | None = None, tenkan_period:int = 9, 
                    kijun_period:int = 26, senkou_span_b_period:int = 52, 
                    ichimoku_lines:bool = True, 
                    last:int | None = None, cut:bool = False) -> pd.DataFrame:
    """
    Calculate Ichimoku cloud values.

    This function calculates the Ichimoku cloud.

    Args:
        data (DataFrame | None, optional): The data used to calculate the Ichimoku cloud values.
        tenkan_period (int, optional): Window length to calculate the Tenkan-sen line.
        kijun_period (int, optional): Window length to calculate the Kijun-sen line.
        senkou_span_b_period (int, optional): Window length to calculate the Senkou Span B.
        ichimoku_lines (bool, optional): If True, adds the columns 'tenkan_sen' and
            'kijun_sen' to the returned DataFrame.
        last (int | None, optional): Number of data points to return starting from the
            present backwards. If None, returns data for all available periods.
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

    v_data = self._StrategyClass__data_adf if data is None else data

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

@_cm._store_decorator
def idc_atr(self, length:int = 14, smooth:str = 'smma', 
                last:int | None = None, cut:bool = False) -> np.ndarray:
    """
    Calculate the average true range (ATR).

    This function calculates the ATR.

    Args:
        length (int, optional): Window length used to smooth the average true range (ATR).
        smooth (str, optional): Type of moving average used to smooth the ATR. 
        last (int | None, optional): Number of data points to return starting from the 
            present backward. If None, returns data for all available periods.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: Series with the average true range values for each step.
    """

    tr = idc_trange(self).to_series()

    match smooth:
        case 'wma':
            atr:np.ndarray = idc_wma(self, data=tr, length=length, 
                                    last=last).unwrap()
        case 'sma':
            atr:np.ndarray = idc_sma(self, data=tr, length=length, 
                                    last=last).unwrap()
        case 'ema':
            atr:np.ndarray = idc_ema(self, data=tr, length=length, 
                                    last=last).unwrap()
        case 'smma':
            atr:np.ndarray = idc_smma(self, data=tr, length=length, 
                                    last=last).unwrap()
        case _:
            atr:np.ndarray = idc_wma(self, data=tr, length=length, 
                                    last=last).unwrap()

    return atr

@_cm._store_decorator
def idc_trange(self, data:pd.DataFrame | None = None, 
                    handle_na: bool = True, last:int | None = None,
                    cut:bool = False) -> pd.Series:
    """
    Calculate the true range.

    This function calculates the true range.

    Args:
        data (DataFrame | None, optional): The data used to perform the calculation.
        handle_na (bool, optional): Whether to handle NaN values in 'close'.
        last (int | None, optional): Number of data points to return starting from the 
            present backward. If None, returns data for all available periods.
        cut (bool, optional): True to return the trimmed data with current index.

    Returns:
        DataWrapper: DataWrapper with the true range values for each step.
    """

    v_data = self._StrategyClass__data_adf if data is None else data

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
