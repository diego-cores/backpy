"""
Indicators module

This module contains the main logic of the indicators.

Variables:
    logger (Logger): Logger variable.

Functions:
    idct_fibonacci: Calculates Fibonacci retracement levels.
    idct_ema: Calculates the Exponential Moving Average (EMA) indicator.
    idct_sma: Calculates the Simple Moving Average (SMA) indicator.
    idct_wma: Calculates the Weighted Moving Average (WMA) indicator.
    idct_smma: Calculates the Smoothed Moving Average (SMMA) indicator.
    idct_sema: Calculates the Smoothed Exponential Moving Average (SEMA) indicator.
    idct_bb: Calculates the Bollinger Bands indicator (BB).
    idct_rsi: Calculates the Relative Strength Index (RSI).
    idct_stochastic: Calculates the Stochastic Oscillator indicator.
    idct_adx: Calculates the Average Directional Index (ADX).
    idct_macd: Calculates the Moving Average Convergence Divergence (MACD).
    idct_sqzmom: Calculates the Squeeze Momentum indicator (SQZMOM).
    idct_rlinreg: This function calculates the rolling linear regression.
    idct_mom: Calculates the Momentum indicator (MOM).
    idct_ichimoku: Calculates the Ichimoku indicator.
    idct_atr: Calculates the Average True Range (ATR).
    idct_trange: This function calculates the true range.
"""

import pandas as pd
import numpy as np
import logging

from . import _commons as _cm

logger:logging.Logger = logging.getLogger(__name__)

@_cm._store_decorator
def idct_fibonacci(lv0:float = 10, lv1:float = 1) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels.

    This function calculates the Fibonacci retracement levels.

    Args:
        lv0 (float, optional): Level 0 position.
        lv1 (float, optional): Level 1 position.

    Returns:
        DataFrame: A DataFrame with Fibonacci levels and their corresponding
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
def idct_ema(data:pd.Series, length:int = 10) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    This function calculates the EMA.

    Args:
        data (Series): Series of data to perform the EMA calculation.
        length (int): The length of the EMA.

    Returns:
        Series: Series containing the EMA values for each step.
    """

    return data.ewm(span=length, adjust=False).mean()

@_cm._store_decorator
def idct_sma(data:pd.Series, length:int = 10) -> pd.Series:
    """
    Simple Moving Average (SMA).

    This function calculates the SMA.

    Args:
        data (Series): Series of data to perform the SMA calculation.
        length (int): Length of the SMA.

    Returns:
        Series: Series containing the SMA values for each step.
    """
    if length <= 0:
        raise ValueError("'length' cannot be less than or equal to 0.")

    return data.rolling(window=length).mean()

@_cm._store_decorator
def idct_wma(data:pd.Series, length:int = 10, 
            invt_weight:bool = False) -> pd.Series:
    """
    Weighted Moving Average (WMA).

    This function calculates the WMA.

    Args:
        data (Series): Series of data to perform the WMA calculation.
        length (int): Length of the WMA.
        invt_weight (bool, optional): If True, the distribution of weights is reversed.

    Returns:
        Series: Series containing the WMA values for each step.
    """
    if length <= 0:
        raise ValueError("'length' cannot be less than or equal to 0.")

    weight = (np.arange(1, length+1)[::-1] 
                if invt_weight else np.arange(1, length+1))
    wma = data.rolling(window=length).apply(
        lambda x: (x*weight).sum() / weight.sum(), raw=True)

    return wma

@_cm._store_decorator
def idct_smma(data:pd.Series, length:int = 10) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).

    This function calculates the SMMA.

    Args:
        data (Series): Series of data to perform the SMMA calculation.
        length (int): Length of the SMMA.

    Returns:
        Series: Series containing the SMMA values for each step.
    """ 
    if length > len(data):
        return pd.Series(np.nan, index=data.index)

    tail_vals = data.iloc[length:].to_numpy()
    tail = np.empty(len(tail_vals) + 1)

    tail[0] = data.iloc[:length].mean()
    tail[1:] = tail_vals

    smma = pd.Series(np.nan, index=data.index)
    smma.iloc[length-1:] = pd.Series(tail).ewm(alpha=1/length, adjust=False).mean()

    return smma

@_cm._store_decorator
def idct_sema(data:pd.Series, length:int = 9, method:str = 'sma', 
            smooth:int = 5, only:bool = False) -> pd.DataFrame:
    """
    Smoothed Exponential Moving Average (SEMA).

    This function calculates the SEMA.

    Args:
        data (Series): Series of data to perform the SEMA calculation.
        length (int, optional): Length of the EMA.
        method (str, optional): Smoothing method. Choices include various smoothing 
            methods: 'sma', 'ema', 'smma', 'wma'.
        smooth (int, optional): Length of the smoothing method.
        only (bool, optional): If True, returns only a Series with the values of the 
                    'method'.

    Columns:
        - 'ema'
        - 'smoothed'

    Returns:
        DataFrame: DataFrame containing 'ema' and 'smoothed' values for 
                        each step.
    """

    ema = data.ewm(span=length, adjust=False).mean()

    match method:
        case 'sma': 
            smema = idct_sma(data=ema, length=smooth)
        case 'ema': smema = idct_ema(data=ema, length=smooth)
        case 'smma': smema = idct_smma(data=ema, length=smooth)
        case 'wma': smema = idct_wma(data=ema, length=smooth)
        case _: raise ValueError(f"'{method}' not is a valid smothing method.")

    if only: return smema
    
    smema = pd.DataFrame({'ema':ema, 'smoothed':smema}, index=ema.index)
    return smema

@_cm._store_decorator
def idct_bb(data:pd.Series, length:int = 20, std_dev:float = 2, 
            ma_type:str = 'sma') -> pd.DataFrame:
    """
    Bollinger Bands (BB).

    This function calculates the BB.

    Args:
        data (Series): Series of data to perform the Bollinger Bands 
            calculation.
        length (int, optional): Window length for calculating Bollinger Bands.
        std_dev (float, optional): Number of standard deviations for the bands.
        ma_type (str, optional): Type of moving average. For example, 'sma' for simple 
                        moving average.

    Columns:
        - 'upper'
        - '{ma_type}'
        - 'lower'

    Returns:
        DataFrame: DataFrame containing 'upper', '{ma_type}', and 'lower' 
                        values for each step.
    """
    if std_dev <= 0:
        raise ValueError("'std_dev' cannot be less than or equal to 0.")

    match ma_type:
        case 'sma': ma = idct_sma(data=data, length=length)
        case 'ema': ma = idct_ema(data=data, length=length)
        case 'wma': ma = idct_wma(data=data, length=length)
        case 'smma': ma = idct_smma(data=data, length=length)
        case _: raise ValueError(f"'{ma_type}' not is a valid type of moving average.") 
    std_ = (std_dev * data.rolling(window=length).std(ddof=0))

    return pd.DataFrame({
        'upper':ma + std_, 
        ma_type:ma,
        'lower':ma - std_
    }, index=ma.index)

@_cm._store_decorator
def idct_rsi(data:pd.Series, length_rsi:int = 14, 
            length:int = 14, rsi_ma_type:str = 'smma', 
            base_type:str = 'sma', bb_std_dev:float = 2)  -> pd.DataFrame:
    """
    Relative Strength Index (RSI).

    This function calculates the RSI.

    Args:
        data (Series): Series of data to perform the RSI calculation.
        length_rsi (int, optional): Window length for the RSI calculation using 
                            `rsi_ma_type`. Default is 14.
        length (int, optional): Window length for the moving average applied to RSI. 
                        Default is 14.
        rsi_ma_type (str, optional): Type of moving average used for calculating RSI. 
                            For example, 'wma' for weighted moving average.
        base_type (str, optional): Type of moving average applied to RSI. For example, 
                        'sma' for simple moving average.
        bb_std_dev (float, optional): Standard deviation for Bollinger Bands. Default is 2.

    Columns:
        - 'rsi'
        - '{base_type}'

    Returns:
        DataFrame: DataFrame containing 'rsi' and '{base_type}' values for 
                        each step.
    """
    if bb_std_dev <= 0:
        raise ValueError("'bb_std_dev' cannot be less than or equal to 0.")

    delta = data.diff()

    match rsi_ma_type:
        case 'sma': ma = idct_sma
        case 'ema': ma = idct_ema
        case 'wma': ma = idct_wma
        case 'smma': ma = idct_smma
        case _: raise ValueError(f"'{rsi_ma_type}' not is a valid type of moving average.") 

    ma_gain = ma(data = delta.where(delta > 0, 0), 
                    length=length_rsi)
    ma_loss = ma(data = -delta.where(delta < 0, 0), 
                    length=length_rsi)
    rsi = 100 - (100 / (1+ma_gain/ma_loss))

    match base_type:
        case 'sma': mv = idct_sma(data=rsi, length=length)
        case 'ema': mv = idct_ema(data=rsi, length=length)
        case 'wma': mv = idct_wma(data=rsi, length=length)
        case 'smma': mv = idct_smma(data=rsi, length=length)
        case 'bb': mv = idct_bb(data=rsi, length=length,
                                        std_dev=bb_std_dev)
        case _: raise ValueError(f"'{base_type}' not is a valid type of moving average.")

    if type(mv) == pd.Series: mv.name = base_type

    return pd.concat([pd.DataFrame({'rsi':rsi}), mv], axis=1)

@_cm._store_decorator
def idct_stochastic(data:pd.DataFrame, length_k:int = 14, smooth_k:int = 1, 
                    length_d:int = 3, d_type:str = 'sma', 
                    source:str = 'close') -> pd.DataFrame:
    """
    Stochastic Oscillator.

    This function calculates the stochastic oscillator.

    Args:
        data (DataFrame): The data used to perform the calculation. 
            Columns used: source, 'low', 'high'.
        length_k (int, optional): Window length for calculating the stochastic values.
        smooth_k (int, optional): Smoothing window length for the stochastic values.
        length_d (int, optional): Window length for the moving average applied to 
                        the stochastic values.
        d_type (str, optional): Type of moving average used for the stochastic oscillator. 
                        For example, 'sma' for simple moving average.
        source (str, optional): Data source for calculation.

    Columns:
        - 'stoch'
        - '{d_type}'

    Returns:
        DataFrame: DataFrame containing 'stoch' and '{d_type}' values for each step.
    """
    if length_k <= 0:
        raise ValueError("'length_k' cannot be less than or equal to 0.")
    elif smooth_k <= 0:
        raise ValueError("'smooth_k' cannot be less than or equal to 0.")
    elif length_d <= 0:
        raise ValueError("'length_d' cannot be less than or equal to 0.")

    low_data = data.loc[:, 'low'].rolling(window=length_k).min()
    high_data = data.loc[:, 'high'].rolling(window=length_k).max()

    match d_type:
        case 'sma': ma = idct_sma
        case 'ema': ma = idct_ema
        case 'wma': ma = idct_wma
        case 'smma': ma = idct_smma
        case _: raise ValueError(f"'{d_type}' not is a valid type of moving average.") 

    stoch = (((data[source] - low_data) / 
                (high_data - low_data)) * 100).rolling(window=smooth_k).mean()
    result = pd.DataFrame({'stoch':stoch, 
                            d_type:ma(data=stoch, length=length_d)})

    return result

@_cm._store_decorator
def idct_adx(data:pd.DataFrame, smooth:int = 14, 
            length_di:int = 14, only:bool = False) -> pd.DataFrame:
    """
    Average Directional Index (ADX).

    This function calculates the ADX.

    Args:
        data (DataFrame): The data used to perform the calculation. 
            Columns used: 'close', 'low', 'high'.
        smooth (int, optional): Smoothing length. Default is 14.
        length_di (int, optional): Window length for calculating +DI and -DI. Default is 14.
        only (bool, optional): If True, returns only a Series with the ADX values.

    Columns:
        - 'adx'
        - '+di'
        - '-di'

    Returns:
        DataFrame: DataFrame containing 'adx', '+di', and '-di' values for 
                        each step.
    """

    atr = idct_atr(data=data, length=length_di, smooth='smma')

    dm_p_raw = data.loc[:, 'high'].diff()
    dm_n_raw = -data.loc[:, 'low'].diff()
    
    dm_p = pd.Series(
        np.where((dm_p_raw > dm_n_raw) & (dm_p_raw > 0), dm_p_raw, 0), 
        index=data.index)
    dm_n = pd.Series(
        np.where((dm_n_raw > dm_p_raw) & (dm_n_raw > 0), dm_n_raw, 0), 
        index=data.index)

    di_p = 100 * idct_smma(dm_p, length=length_di) / atr
    di_n = 100 * idct_smma(dm_n, length=length_di) / atr

    adx = idct_smma(
        data=100 * np.abs((di_p - di_n) / (di_p + di_n).replace(0, 1)), 
        length=smooth)

    if only: 
        return adx
    adx = pd.DataFrame({'adx':adx, '+di':di_p, '-di':di_n})

    return adx

@_cm._store_decorator
def idct_macd(data:pd.Series, short_len:int = 12, 
            long_len:int = 26, signal_len:int = 9, 
            macd_ma_type:str = 'ema', signal_ma_type:str = 'ema', 
            histogram:bool = True) -> pd.DataFrame:
    """
    Calculate the convergence/divergence of the moving average (MACD).

    This function calculates the MACD.

    Args:
        data (Series): The data used for calculation of MACD.
        short_len (int, optional): Length of the short moving average used to calculate MACD.
        long_len (int, optional): Length of the long moving average used to calculate MACD.
        signal_len (int, optional): Length of the moving average for the MACD signal line.
        macd_ma_type (str, optional): Type of moving average used to calculate MACD.
        signal_ma_type (str, optional): Type of moving average used to smooth the MACD.
        histogram (bool, optional): If True, includes an additional 'histogram' column.

    Columns:
        - 'macd'
        - 'signal'
        - 'histogram'  

    Returns:
        DataFrame: A DataFrame with MACD values and signal line for each step.
    """

    match macd_ma_type:
        case 'ema':
            macd_ma = idct_ema
        case 'sma':
            macd_ma = idct_sma
        case 'wma': 
            macd_ma = idct_wma
        case 'smma': 
            macd_ma = idct_smma
        case _:
            raise ValueError(f"'{macd_ma_type}' not is a valid type of moving average.") 

    match signal_ma_type:
        case 'ema':
            signal_ma = idct_ema
        case 'sma':
            signal_ma = idct_sma
        case 'wma': 
            signal_ma = idct_wma
        case 'smma': 
            signal_ma = idct_smma
        case _:
            raise ValueError(f"'{signal_ma_type}' not is a valid type of moving average.") 

    short_ema = macd_ma(data=data, length=short_len)
    long_ema = macd_ma(data=data, length=long_len)
    macd = short_ema - long_ema

    signal_line = signal_ma(data=macd, length=signal_len)

    return pd.DataFrame({'macd':macd, 'signal':signal_line, 
                        'histogram':macd-signal_line} 
                        if histogram else 
                        {'macd':macd, 'signal':signal_line})

@_cm._store_decorator
def idct_sqzmom(data:pd.DataFrame, bb_len:int = 20, 
                bb_mult:float = 1.5, kc_len:int = 20, 
                kc_mult:float = 1.5, use_tr:bool = True,
                source:str = 'close') -> pd.DataFrame:
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
        data (DataFrame): The data used for calculating the Squeeze Momentum.
            Columns used: source, 'close', 'low', 'high'.
        bb_len (int, optional): Bollinger band length.
        bb_mult (float, optional): Bollinger band standard deviation.
        kc_len (int, optional): Keltner channel length.
        kc_mult (float, optional): Keltner channel standard deviation.
        use_tr (bool, optional): If False, ('high' - 'low') is used instead of the true 
            range.
        source (str, optional): Data source for calculations.

    Columns:
        - 'sqzmom'
        - 'histogram'

    Returns:
        DataFrame: A DataFrame with Squeeze Momentum values and histogram for 
            each step.
    """

    basis = idct_sma(data=data[source], length=bb_len)
    dev = bb_mult * data[source].rolling(window=bb_len).std(ddof=0)

    upper_bb = basis + dev
    lower_bb = basis - dev

    ma = idct_sma(data=data[source], length=kc_len)
    range_ = idct_sma(data=idct_trange(data=data) if use_tr else 
                    data['high']-data['low'], length=kc_len)

    upper_kc = ma + range_ * kc_mult
    lower_kc = ma - range_ * kc_mult

    sqz = np.where((lower_bb > lower_kc) & (upper_bb < upper_kc), 1, 0)

    d = data[source] - ((data.loc[:, 'low'].rolling(window=kc_len).min() + 
                            data.loc[:, 'high'].rolling(window=kc_len).max()) / 2 + 
                            idct_sma(data=data[source], length=kc_len)) / 2

    histogram = idct_rlinreg(data=d, length=kc_len, offset=0)

    return pd.DataFrame({'sqzmom':pd.Series(sqz, index=data.index), 
                        'histogram':histogram}, 
                        index=data.index)

@_cm._store_decorator
def idct_rlinreg(data:pd.Series, length:int = 5, offset:int = 1) -> pd.Series:
    """
    Calculate rolling linear regression values.

    This function calculates the rolling linear regression.

    Args:
        data (Series): The data used for linear regression calculations.
        length (int, optional): Length of each window for the rolling regression.
        offset (int, optional): Offset used in the regression calculation.

    Returns:
        Series: Series with the linear regression values for each window.
    """
    if offset < 0 or offset >= length:
        raise ValueError("'offset' cannot be less than 0 and cannot be more or equal than 'length'")

    x = np.arange(length)
    y = data.rolling(window=length)

    m = y.apply(lambda y: np.polyfit(x, y.values, 1)[0])
    b = y.mean() - (m * float(np.mean(x))) 

    return m * (length - 1 - offset) + b

@_cm._store_decorator
def idct_mom(data:pd.Series, length:int = 10) -> pd.Series:
    """
    Calculate momentum values (MOM).

    This function calculates the MOM.

    Args:
        data (Series): The data used to calculate momentum.
        length (int, optional): Length for calculating momentum.

    Returns:
        Series: Series with the momentum values for each step.
    """
    if length <= 0:
        raise ValueError("'length' cannot be less than or equal to 0.")

    return data - data.shift(length)

@_cm._store_decorator
def idct_ichimoku(data:pd.DataFrame, tenkan_period:int = 9, 
                kijun_period:int = 26, senkou_span_b_period:int = 52, 
                ichimoku_lines:bool = True,) -> pd.DataFrame:
    """
    Calculate Ichimoku cloud values.

    This function calculates the Ichimoku cloud.
    The displacement is not calculated, the results are the same, but the data is not displaced.

    Args:
        data (DataFrame): The data used to calculate the Ichimoku cloud values.
            Columns used: 'low', 'high'.
        tenkan_period (int, optional): Window length to calculate the Tenkan-sen line.
        kijun_period (int, optional): Window length to calculate the Kijun-sen line.
        senkou_span_b_period (int, optional): Window length to calculate the Senkou Span B.
        ichimoku_lines (bool, optional): If True, adds the columns 'tenkan_sen' and
            'kijun_sen' to the returned DataFrame.

    Columns:
        - 'senkou_a'
        - 'senkou_b'
        - 'tenkan_sen'
        - 'kijun_sen'

    Returns:
        DataFrame: A DataFrame with Ichimoku cloud values and optionally
            'tenkan_sen' and 'kijun_sen' columns if `ichimoku_lines` is True.
    """
    if tenkan_period <= 0:
        raise ValueError("'tenkan_period' cannot be less than or equal to 0.")
    elif kijun_period <= 0:
        raise ValueError("'kijun_period' cannot be less than or equal to 0.")
    elif senkou_span_b_period <= 0:
        raise ValueError("'senkou_span_b_period' cannot be less than or equal to 0.")

    tenkan_sen_val = (data.loc[:, 'high'].rolling(window=tenkan_period).max() + 
                        data.loc[:, 'low'].rolling(window=tenkan_period).min()) / 2
    kijun_sen_val = (data.loc[:, 'high'].rolling(window=kijun_period).max() + 
                        data.loc[:, 'low'].rolling(window=kijun_period).min()) / 2

    senkou_span_a_val = ((tenkan_sen_val + kijun_sen_val) / 2)
    senkou_span_b_val = ((data.loc[:, 'high'].rolling(
        window=senkou_span_b_period).max() + 
        data.loc[:, 'low'].rolling(window=senkou_span_b_period).min()) / 2)
    senkou_span = (pd.DataFrame({'senkou_a':senkou_span_a_val,
                                'senkou_b':senkou_span_b_val, 
                                'tenkan_sen':tenkan_sen_val,
                                'kijun_sen':kijun_sen_val}) 
                    if ichimoku_lines else 
                    pd.DataFrame({'senkou_a':senkou_span_a_val,
                                'senkou_b':senkou_span_b_val}))

    return senkou_span

@_cm._store_decorator
def idct_atr(data:pd.DataFrame, length:int = 14, smooth:str = 'smma', 
            handle_na:bool = True) -> np.ndarray:
    """
    Calculate the average true range (ATR).

    This function calculates the ATR.

    Args:
        data (DataFrame): The data used to perform the calculation. 
            Columns used: 'close', 'low', 'high'.
        length (int, optional): Window length used to smooth the average true range (ATR).
        smooth (str, optional): Type of moving average used to smooth the ATR. 
        handle_na (bool, optional): Whether to handle NaN values in 'close' (TR).

    Returns:
        ndarray: ndarray with the average true range values for each step.
    """

    tr = idct_trange(data=data, handle_na=handle_na)

    match smooth:
        case 'wma':
            atr:np.ndarray = idct_wma(data=tr, length=length)
        case 'sma':
            atr:np.ndarray = idct_sma(data=tr, length=length)
        case 'ema':
            atr:np.ndarray = idct_ema(data=tr, length=length)
        case 'smma':
            atr:np.ndarray = idct_smma(data=tr, length=length)
        case _:
            raise ValueError(f"'{smooth}' not is a valid type of moving average.") 

    return atr

@_cm._store_decorator
def idct_trange(data:pd.DataFrame, handle_na:bool = True) -> pd.Series:
    """
    Calculate the true range.

    This function calculates the true range.

    Args:
        data (DataFrame): The data used to perform the calculation. 
            Columns used: 'close', 'low', 'high'.
        handle_na (bool, optional): Whether to handle NaN values in 'close'.

    Returns:
        Series: Series with the true range values for each step.
    """

    close = data.loc[:, 'close'].shift(1)

    if handle_na:
            close.fillna(data['low'], inplace=True)
                    
    hl = data.loc[:, 'high'] - data.loc[:, 'low']
    hyc = abs(data['high'] - close)
    lyc = abs(data['low'] - close)
    tr:pd.Series[float] = pd.concat([hl, hyc, lyc], axis=1).max(axis=1)

    if not handle_na:
        tr[close.isna()] = np.nan

    return tr
