"""
Indicators Test
"""

from backpy import _commons as _cm
from backpy import indicators
import backpy

from pandas.testing import assert_frame_equal, assert_series_equal
from pathlib import Path
import pandas as pd
import numpy as np
import unittest

from typing import Sequence, Callable

TEST_DIR = Path(__file__).resolve().parent
DATA_PATH = TEST_DIR / "data4k.bpd" # Binance spot data # interval: 1d

class TestIndicators(unittest.TestCase):
    def setUp(self) -> None:
        backpy.load_data_bpd(DATA_PATH, statistics=False, progress=False)

        self.data = getattr(_cm, '__data')
        self.assertIsNotNone(self.data, 'data not loaded')

    def _assert_lookahead(self, func:Callable, params:Sequence[dict], 
                        points:Sequence[int] = (100, 1000, 1500, 2000), pass_all:bool = True):
        """
        Assert lookahead

        Verify that the indicator does not have lookahead bias.

        Args:
            func (Callable): Indicator.
            params (Sequence[dict]): Sequence of parameters to test.
            points (Sequence[int], optional): Sequence of cut points.
            pass_all (bool, optional): Pass the entire data DataFrame or just the 'close' column.
        """

        for p in params:
            for point in points:
                data = self.data if pass_all else self.data['close']
                full = func(data, **p)
                cuted = full.iloc[:point]
                recalc = func(data[:point], **p)

                self.assertTrue(isinstance(full, (pd.Series, pd.DataFrame)), msg='Unsupported type')
                self.assertEqual(len(cuted), len(recalc), msg='Different length')

                (assert_series_equal if isinstance(full, pd.Series) else assert_frame_equal)(
                    cuted, recalc, obj=f'params={p}, point={point}')

    def _assert_equal(self, func:Callable, precalc:dict, params:Sequence[dict], 
                    kwargs:dict = {}, pass_all:bool = True) -> None:
        """
        Assert equal

        Compare the result of 'func' with the 'precalc' results.

        Note:
            If 'kwargs' does not include 'places' or 'delta' then 'places:2' is added.

        Args:
            func (Callable): Indicator.
            precalc (dict): Dictionary with precalculations in this form: 
                {index in the data:sequence of precalculations}, 
                where if the result is a 'DataFrame' the precalculations must be in dict form.
            params (Sequence[dict]): Sequence of parameters to test.
            kwargs (dict, optional): Arguments that will be passed to 'assertAlmostEqual'.
            pass_all (bool, optional): Pass the entire data DataFrame or just the 'close' column.
        """

        for key, values in precalc.items():
            for i, value in enumerate(values):
                idc = func(self.data if pass_all else self.data['close'], **params[i])

                if 'places' not in kwargs and 'delta' not in kwargs:
                    kwargs.update({'places':2})
                kwargs.update({'msg':f'params={params[i]}, key={key}'})

                self.assertTrue(isinstance(idc, (pd.Series, pd.DataFrame)), msg='Unsupported type')
                if isinstance(idc, pd.DataFrame):
                    for col in idc.columns:
                        self.assertAlmostEqual(idc.iloc[key][col], value[col], **kwargs)
                else: self.assertAlmostEqual(idc.iloc[key], value, **kwargs)

    def _assert_raises(self, func:Callable, params:Sequence[dict], 
                        exception:type = Exception, pass_all:bool = True) -> None:
        """
        Assert raises

        Verify that the parameters throwing an error.

        Args:
            func (Callable): Indicator.
            params (Sequence[dict]): Sequence of parameters to test.
            exception (type, optional): Expected exception.
            pass_all (bool, optional): Pass the entire data DataFrame or just the 'close' column.
        """

        data = self.data if pass_all else self.data['close']
        msg_f = lambda p, re=None: f'params={p}, result={re}'

        class UpdateMsg:
            def __init__(self, msg:Callable):
                self.msg = msg
                self.msg_kwargs = {}

            def update(self, kwargs):
                self.msg_kwargs.update(kwargs)
                return self

            def __str__(self):
                return self.msg(**self.msg_kwargs)

        for p in params:
            msg = UpdateMsg(msg_f)
            with self.assertRaises(exception, msg=msg.update({'p':p})):
                re = func(data, **p)
                msg.update({'re':re})

    def test_hasattr(self) -> None:
        """
        Test attribute

        Verify that all indicators have the '_store' attribute.
        """

        for name in dir(indicators):
            obj = getattr(indicators, name)
            if not callable(obj):
                continue

            self.assertTrue(hasattr(obj, '_store'), f"'{obj}' does not have the '_store' attribute")

    def test_idct_fibonacci(self) -> None:
        """
        Test 'idct_fibonacci'

        Verify that it works correctly.
        """

        fibonacci_precalc = pd.DataFrame({
            'Level':[0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 3.618, 4.236], 
            'Value':[64323.3, 64473.8, 64566.8, 64642.1, 64717.3, 64824.4, 
                64960.8, 65354.8, 65992.4, 66629.9, 67023.9]
        })
        fibonacci = indicators.idct_fibonacci(lv0=64323.3, lv1=64960.8)

        assert_frame_equal(fibonacci, fibonacci_precalc, 
            check_exact=False, atol=0.05, check_dtype=False)

        fibonacci_precalc = pd.DataFrame({
            'Level':[0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 3.618, 4.236], 
            'Value':[0.00000478, 0.00000482, 0.00000484, 0.00000485, 0.00000487, 0.00000489, 
                0.00000492, 0.00000501, 0.00000515, 0.00000530, 0.00000538]
        })
        fibonacci = indicators.idct_fibonacci(lv0=0.00000478, lv1=0.00000492)

        assert_frame_equal(fibonacci, fibonacci_precalc, 
                    check_exact=False, atol=0.00000005, check_dtype=False)

    def test_idct_fibonacci_invalid(self) -> None:
        """
        Test 'idct_fibonacci'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'lv0':1, 'lv1':3}
        overrides = (
            {'lv0':np.ndarray([44,23])}, {'lv0':np.ndarray([0,3,2,5,6,3,3,6,8,1,5])}, {'lv0':None},
            {'lv1':np.ndarray([0,3,2,5,6,3,3,6,8,1,5])}, {'lv1':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(
            lambda data, *args, **kwargs: indicators.idct_fibonacci(*args, **kwargs), params=params, pass_all=True)

    def test_idct_ema(self) -> None:
        """
        Test 'idct_ema'

        Verify that it works correctly.
        """

        params = [{'length':10}, {'length':100}]
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (9437.28, 8762.98),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (17485.84, 18051.91),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (115649.39, 110551.21),
        }

        self._assert_equal(indicators.idct_ema, precalc=precalc, params=params, pass_all=False)

    def test_idct_ema_lookahead(self) -> None:
        """
        Test 'idct_ema'

        Verify that the indicator does not have look-ahead bias
        """

        params = [{'length':10}, {'length':5}, {'length':1}, {'length':100}, {'length':1000}]
        self._assert_lookahead(indicators.idct_ema, params=params, pass_all=False)

    def test_idct_ema_invalid(self) -> None:
        """
        Test 'idct_ema'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10}
        overrides = (
            {'length':0}, {'length':-450}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_ema, params=params, pass_all=False)

    def test_idct_sma(self) -> None:
        """
        Test 'idct_sma'

        Verify that it works correctly.
        """

        params = [{'length':9}, {'length':100}]
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (9416.98, 8090.58),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (17345.9, 17900.54),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (116622.27, 111098.55),
        }

        self._assert_equal(indicators.idct_sma, precalc=precalc, params=params, pass_all=False)

    def test_idct_sma_lookahead(self) -> None:
        """
        Test 'idct_sma'

        Verify that the indicator does not have look-ahead bias
        """

        params = [{'length':9}, {'length':5}, {'length':1}, {'length':100}, {'length':1000}]
        self._assert_lookahead(indicators.idct_sma, params=params, pass_all=False)

    def test_idct_sma_invalid(self) -> None:
        """
        Test 'idct_sma'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10}
        overrides = (
            {'length':0}, {'length':-450}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_sma, params=params, pass_all=False)

    def test_idct_wma(self) -> None:
        """
        Test 'idct_wma'

        Verify that it works correctly.
        """

        params = [{'length':9}, {'length':100}]
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (9399.5, 8859.56),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (17624.43, 17302.34),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (115169.18, 113727.67),
        }

        self._assert_equal(indicators.idct_wma, precalc=precalc, params=params, pass_all=False)

    def test_idct_wma_lookahead(self) -> None:
        """
        Test 'idct_wma'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':9,'invt_weight':True}, 
            {'length':5,'invt_weight':False}, 
            {'length':1,'invt_weight':True}, 
            {'length':100,'invt_weight':True}, 
            {'length':1000,'invt_weight':True}
        )
        self._assert_lookahead(indicators.idct_wma, params=params, pass_all=False)

    def test_idct_wma_invalid(self) -> None:
        """
        Test 'idct_wma'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10, 'invt_weight':True}
        overrides = (
            {'length':0}, {'length':-450}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_wma, params=params, pass_all=False)

    def test_idct_smma(self) -> None:
        """
        Test 'idct_smma'

        Verify that it works correctly.
        """

        params = [{'length':7}, {'length':100}]
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (9455.91, 8497.91),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (17356.85, 20967.87),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (115988.62, 103320.75),
        }

        self._assert_equal(indicators.idct_smma, precalc=precalc, params=params, pass_all=False)

    def test_idct_smma_lookahead(self) -> None:
        """
        Test 'idct_smma'

        Verify that the indicator does not have look-ahead bias
        """

        params = [{'length':9}, {'length':5}, {'length':1}, {'length':100}, {'length':1000}]
        self._assert_lookahead(indicators.idct_smma, params=params, pass_all=False)

    def test_idct_smma_invalid(self) -> None:
        """
        Test 'idct_smma'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10}
        overrides = (
            {'length':0}, {'length':-450}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_smma, params=params, pass_all=False)

    def test_idct_sema(self) -> None:
        """
        Test 'idct_sema'

        Verify that it works correctly.
        """

        params = ({'length':10, 'method':'sma', 'smooth':5},{'length':100, 'method':'smma', 'smooth':25})
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'ema':9437.28,'smoothed':9481.34}, {'ema':8762.98,'smoothed':8363.64}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'ema':17485.84,'smoothed':17093.26}, {'ema':18051.91,'smoothed':18839.8}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'ema':115649.39,'smoothed':116901.25}, {'ema':110551.21,'smoothed':106438.96}),
        }

        self._assert_equal(indicators.idct_sema, precalc=precalc, params=params, pass_all=False)

    def test_idct_sema_lookahead(self) -> None:
        """
        Test 'idct_sema'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':10, 'method':'sma', 'smooth':5, 'only':False},
            {'length':5, 'method':'ema', 'smooth':2, 'only':True},
            {'length':1, 'method':'wma', 'smooth':1, 'only':False},
            {'length':100, 'method':'smma', 'smooth':25, 'only':False},
            {'length':1000, 'method':'smma', 'smooth':100, 'only':False},
        )
        self._assert_lookahead(indicators.idct_sema, params=params, pass_all=False)

    def test_idct_sema_invalid(self) -> None:
        """
        Test 'idct_sema'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10, 'method':'sma', 'smooth':5, 'only':False}
        overrides = (
            {'length':0}, {'length':-5}, {'length':None},
            {'smooth':0}, {'smooth':-5}, {'smooth':None},
            {'method':'NONE'},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_sema, params=params, pass_all=False)

    def test_idct_bb(self) -> None:
        """
        Test 'idct_bb'

        Verify that it works correctly.
        """

        params = ({'length':20, 'std_dev':2, 'ma_type':'sma'},{'length':40, 'std_dev':0.5, 'ma_type':'smma'})
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'upper':10035.25,'sma':9584.06,'lower':9132.88}, {'upper':9050.71,'smma':8896.03,'lower':8741.36}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'upper':18062.51,'sma':16985.12,'lower':15907.74}, {'upper':17856.8,'smma':17632.49,'lower':17408.18}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'upper':121822.86,'sma':116462.79,'lower':111102.72}, {'upper':113486.76,'smma':112346.09,'lower':111205.41}),
        }

        self._assert_equal(indicators.idct_bb, precalc=precalc, params=params, kwargs={'delta':0.01}, pass_all=False)

    def test_idct_bb_lookahead(self) -> None:
        """
        Test 'idct_bb'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':20, 'std_dev':2, 'ma_type':'sma'},
            {'length':1, 'std_dev':0.1, 'ma_type':'ema'},
            {'length':40, 'std_dev':0.5, 'ma_type':'smma'},
            {'length':100, 'std_dev':2, 'ma_type':'wma'},
            {'length':1000, 'std_dev':10, 'ma_type':'sma'},
        )
        self._assert_lookahead(indicators.idct_bb, params=params, pass_all=False)

    def test_idct_bb_invalid(self) -> None:
        """
        Test 'idct_bb'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':20, 'std_dev':2, 'ma_type':'sma'}
        overrides = (
            {'std_dev':0}, {'std_dev':-5}, {'std_dev':None},
            {'length':0}, {'length':-5}, {'length':None},
            {'ma_type':'NONE'},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_bb, params=params, pass_all=False)

    def test_idct_rsi(self) -> None:
        """
        Test 'idct_rsi'

        Verify that it works correctly.
        """

        params = (
            {'length_rsi':14, 'rsi_ma_type':'smma'},
            {'length_rsi':4, 'rsi_ma_type':'smma'},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'rsi':48.45,'sma':51.41}, {'rsi':40.24,'sma':45.16}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'rsi':81.01,'sma':54.60}, {'rsi':99.22,'sma':71.70}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'rsi':40.44,'sma':52.56}, {'rsi':24.9,'sma':50.2}),
        }

        self._assert_equal(indicators.idct_rsi, precalc=precalc, params=params, pass_all=False)

    def test_idct_rsi_lookahead(self) -> None:
        """
        Test 'idct_rsi'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length_rsi':14, 'rsi_ma_type':'smma'},
            {'length_rsi':4, 'rsi_ma_type':'sma'},
            {'length_rsi':1, 'rsi_ma_type':'ema'},
            {'length_rsi':100, 'rsi_ma_type':'wma'},
            {'length_rsi':1000, 'rsi_ma_type':'smma'},
        )
        self._assert_lookahead(indicators.idct_rsi, params=params, pass_all=False)

    def test_idct_rsi_invalid(self) -> None:
        """
        Test 'idct_rsi'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length_rsi':14, 'length':14, 'rsi_ma_type':'smma', 'base_type':'sma', 'bb_std_dev':2}
        overrides = (
            {'length_rsi':0}, {'length_rsi':-5}, {'length_rsi':None},
            {'length':0}, {'length':-5}, {'length':None},
            {'rsi_ma_type':'NONE'},
            {'base_type':'NONE'},
            {'bb_std_dev':0}, {'bb_std_dev':-2}, {'bb_std_dev':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_rsi, params=params, pass_all=False)

    def test_idct_stochastic(self) -> None:
        """
        Test 'idct_stochastic'

        Verify that it works correctly.
        """

        params = (
            {'length_k':14, 'smooth_k':1, 'length_d':3, 'd_type':'sma'},
            {'length_k':5, 'smooth_k':3, 'length_d':6, 'd_type':'sma'},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'stoch':41.44,'sma':40.78}, {'stoch':58.01,'sma':62.79}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'stoch':90.29,'sma':93.96}, {'stoch':91.81,'sma':83.32}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'stoch':3.89,'sma':6.91}, {'stoch':13.36,'sma':11.46}),
        }

        self._assert_equal(indicators.idct_stochastic, precalc=precalc, params=params, pass_all=True)

    def test_idct_stochastic_lookahead(self) -> None:
        """
        Test 'idct_stochastic'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length_k':14, 'smooth_k':1, 'length_d':3, 'd_type':'sma', 'source':'close'},
        )
        self._assert_lookahead(indicators.idct_stochastic, params=params, pass_all=True)

    def test_idct_stochastic_invalid(self) -> None:
        """
        Test 'idct_stochastic'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length_k':14, 'smooth_k':1, 'length_d':3, 'd_type':'sma', 'source':'close'}
        overrides = (
            {'length_k':0}, {'length_k':-5}, {'length_k':None},
            {'smooth_k':0}, {'smooth_k':-5}, {'smooth_k':None},
            {'length_d':0}, {'length_d':-5}, {'length_d':None},
            {'d_type':'NONE'},
            {'source':'NONE'},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_stochastic, params=params, pass_all=True)

    def test_idct_adx(self) -> None:
        """
        Test 'idct_adx'

        Verify that it works correctly.
        """

        params = (
            {'smooth':14, 'length_di':14, 'only':True},
            {'smooth':14, 'length_di':14, 'only':False},
            {'smooth':5, 'length_di':3, 'only':False},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (19.31, {'adx':19.31,'+di':14.17,'-di':22.64}, {'adx':68.69,'+di':2.91,'-di':25.98}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (22.23, {'adx':22.23,'+di':47.84,'-di':10.43}, {'adx':89.57,'+di':79.06,'-di':0.07}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (17.98, {'adx':17.98,'+di':18.28,'-di':20.68}, {'adx':58.40,'+di':4.12,'-di':28.00}),
        }

        self._assert_equal(indicators.idct_adx, precalc=precalc, params=params, pass_all=True)

    def test_idct_adx_lookahead(self) -> None:
        """
        Test 'idct_adx'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'smooth':14, 'length_di':14, 'only':True},
            {'smooth':4, 'length_di':3, 'only':True},
            {'smooth':1, 'length_di':1, 'only':False},
            {'smooth':100, 'length_di':20, 'only':False},
            {'smooth':3, 'length_di':7, 'only':False},
        )
        self._assert_lookahead(indicators.idct_adx, params=params, pass_all=True)

    def test_idct_adx_invalid(self) -> None:
        """
        Test 'idct_adx'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'smooth':14, 'length_di':14, 'only':True}
        overrides = (
            {'smooth':0}, {'smooth':-5}, {'smooth':None},
            {'length_di':0}, {'length_di':-5}, {'length_di':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_adx, params=params, pass_all=True)

    def test_idct_macd(self) -> None:
        """
        Test 'idct_macd'

        Verify that it works correctly.
        """

        params = (
            {'short_len':12, 'long_len':26, 'signal_len':9, 'macd_ma_type':'ema', 'signal_ma_type':'ema', 'histogram':True},
            {'short_len':12, 'long_len':26, 'signal_len':9, 'macd_ma_type':'ema', 'signal_ma_type':'ema', 'histogram':False},
            {'short_len':6, 'long_len':12, 'signal_len':4, 'macd_ma_type':'sma', 'signal_ma_type':'sma', 'histogram':True},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'macd':18.82,'signal':76.15,'histogram':-57.34},{'macd':18.82,'signal':76.15},
                {'macd':-62.18,'signal':-97.88,'histogram':35.70}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'macd':256.91,'signal':50.69,'histogram':206.21},{'macd':256.91,'signal':50.69},
                {'macd':406.88,'signal':269.16,'histogram':137.72}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'macd':-348.34,'signal':387.36,'histogram':-735.70},{'macd':-348.34,'signal':387.36},
                {'macd':-2200.06,'signal':-1268.36,'histogram':-931.71}),
        }

        self._assert_equal(indicators.idct_macd, precalc=precalc, params=params, pass_all=False)

    def test_idct_macd_lookahead(self) -> None:
        """
        Test 'idct_macd'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'short_len':12, 'long_len':26, 'signal_len':9, 'macd_ma_type':'ema', 'signal_ma_type':'ema', 'histogram':True},
            {'short_len':67, 'long_len':10, 'signal_len':67, 'macd_ma_type':'ema', 'signal_ma_type':'sma', 'histogram':False},
            {'short_len':54, 'long_len':32, 'signal_len':16, 'macd_ma_type':'wma', 'signal_ma_type':'ema', 'histogram':True},
            {'short_len':6, 'long_len':13, 'signal_len':5, 'macd_ma_type':'smma', 'signal_ma_type':'ema', 'histogram':False},
            {'short_len':120, 'long_len':20, 'signal_len':10, 'macd_ma_type':'ema', 'signal_ma_type':'sma', 'histogram':True},
            {'short_len':1, 'long_len':1, 'signal_len':1, 'macd_ma_type':'ema', 'signal_ma_type':'ema', 'histogram':False},
        )
        self._assert_lookahead(indicators.idct_macd, params=params, pass_all=False)

    def test_idct_macd_invalid(self) -> None:
        """
        Test 'idct_macd'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'short_len':12, 'long_len':26, 'signal_len':9, 'macd_ma_type':'ema', 'signal_ma_type':'ema', 'histogram':True}
        overrides = (
            {'short_len':0}, {'short_len':-5}, {'short_len':None},
            {'long_len':0}, {'long_len':-5}, {'long_len':None},
            {'signal_len':0}, {'signal_len':-5}, {'signal_len':None},
            {'macd_ma_type':'NONE'},
            {'signal_ma_type':'NONE'},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_macd, params=params, pass_all=False)

    def test_idct_sqzmom(self) -> None:
        """
        Test 'idct_sqzmom'

        Verify that it works correctly.
        """

        params = (
            {'bb_len':20, 'bb_mult':1.5, 'kc_len':20, 'kc_mult':1.5, 'use_tr':True, 'source':'close'},
            {'bb_len':25, 'bb_mult':1.5, 'kc_len':40, 'kc_mult':1.5, 'use_tr':False, 'source':'close'},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'sqzmom':1,'histogram':-308.00},{'sqzmom':1,'histogram':-18.83},),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'sqzmom':0,'histogram':746.97},{'sqzmom':0,'histogram':365.11},),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'sqzmom':1,'histogram':-694.23},{'sqzmom':0,'histogram':-1961.88},),
        }

        self._assert_equal(indicators.idct_sqzmom, precalc=precalc, params=params, pass_all=True)

    def test_idct_sqzmom_lookahead(self) -> None:
        """
        Test 'idct_sqzmom'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'bb_len':20, 'bb_mult':1.5, 'kc_len':20, 'kc_mult':1.5, 'use_tr':True, 'source':'close'},
            {'bb_len':1, 'bb_mult':0.1, 'kc_len':2, 'kc_mult':0.1, 'use_tr':False, 'source':'close'},
            {'bb_len':100, 'bb_mult':5, 'kc_len':100, 'kc_mult':5, 'use_tr':True, 'source':'close'},
        )
        self._assert_lookahead(indicators.idct_sqzmom, params=params, pass_all=True)

    def test_idct_sqzmom_invalid(self) -> None:
        """
        Test 'idct_sqzmom'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'bb_len':20, 'bb_mult':1.5, 'kc_len':20, 'kc_mult':1.5, 'use_tr':True, 'source':'close'}
        overrides = (
            {'bb_len':0}, {'bb_len':-5}, {'bb_len':None},
            {'bb_mult':0}, {'bb_mult':-5}, {'bb_mult':None},
            {'kc_len':0}, {'kc_len':-5}, {'kc_len':None},
            {'kc_mult':0}, {'kc_mult':-5}, {'kc_mult':None},
            {'source':'NONE'},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_sqzmom, params=params, pass_all=False)

    def test_idct_rlinreg_lookahead(self) -> None:
        """
        Test 'idct_rlinreg'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':5, 'offset':1},
        )
        self._assert_lookahead(indicators.idct_rlinreg, params=params, pass_all=True)

    def test_idct_rlinreg_invalid(self) -> None:
        """
        Test 'idct_rlinreg'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':5, 'offset':1}
        overrides = (
            {'offset':-5}, {'offset':None}, {'offset':15}, 
            {'length':0}, {'length':-5}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_rlinreg, params=params, pass_all=False)

    def test_idct_mom(self) -> None:
        """
        Test 'idct_mom'

        Verify that it works correctly.
        """

        params = (
            {'length':10},
            {'length':67},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (-526.05, 2490.25),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (2173.75, -2058.96),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (-6186.00, 6905.99),
        }

        self._assert_equal(indicators.idct_mom, precalc=precalc, params=params, pass_all=False)

    def test_idct_mom_lookahead(self) -> None:
        """
        Test 'idct_mom'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':10},
            {'length':67},
            {'length':1},
            {'length':100},
        )
        self._assert_lookahead(indicators.idct_mom, params=params, pass_all=False)

    def test_idct_mom_invalid(self) -> None:
        """
        Test 'idct_mom'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':10}
        overrides = (
            {'length':0}, {'length':-5}, {'length':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_mom, params=params, pass_all=False)

    def test_idct_ichimoku(self) -> None:
        """
        Test 'idct_ichimoku'

        Verify that it works correctly.
        """

        params = (
            {'tenkan_period':9, 'kijun_period':26, 'senkou_span_b_period':52, 'ichimoku_lines':True},
            {'tenkan_period':4, 'kijun_period':14, 'senkou_span_b_period':25, 'ichimoku_lines':False},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                ({'senkou_a':9394.86, 'senkou_b':9248.50, 'tenkan_sen':9249.73, 'kijun_sen':9540.00}, {'senkou_a':9409.78, 'senkou_b':9595.87}),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                ({'senkou_a':17785.76, 'senkou_b':17366.84, 'tenkan_sen':17884.85, 'kijun_sen':17686.67}, {'senkou_a':17917.93, 'senkou_b':17686.67}),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                ({'senkou_a':118220.92, 'senkou_b':114787.10, 'tenkan_sen':118244.84, 'kijun_sen':118197.00}, {'senkou_a':116512.27, 'senkou_b':118197.00}),
        }

        self._assert_equal(indicators.idct_ichimoku, precalc=precalc, params=params, kwargs={'delta':0.01}, pass_all=True)

    def test_idct_ichimoku_lookahead(self) -> None:
        """
        Test 'idct_ichimoku'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'tenkan_period':9, 'kijun_period':26, 'senkou_span_b_period':52, 'ichimoku_lines':True},
            {'tenkan_period':4, 'kijun_period':14, 'senkou_span_b_period':25, 'ichimoku_lines':False},
            {'tenkan_period':1, 'kijun_period':1, 'senkou_span_b_period':1, 'ichimoku_lines':True},
            {'tenkan_period':100, 'kijun_period':100, 'senkou_span_b_period':100, 'ichimoku_lines':True},
        )
        self._assert_lookahead(indicators.idct_ichimoku, params=params, pass_all=True)

    def test_idct_ichimoku_invalid(self) -> None:
        """
        Test 'idct_ichimoku'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'tenkan_period':9, 'kijun_period':26, 'senkou_span_b_period':52, 'ichimoku_lines':True}
        overrides = (
            {'tenkan_period':0}, {'tenkan_period':-5}, {'tenkan_period':None},
            {'kijun_period':0}, {'kijun_period':-5}, {'kijun_period':None},
            {'senkou_span_b_period':0}, {'senkou_span_b_period':-5}, {'senkou_span_b_period':None},
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_ichimoku, params=params, pass_all=True)

    def test_idct_atr(self) -> None:
        """
        Test 'idct_atr'

        Verify that it works correctly.
        """

        params = (
            {'length':14, 'smooth':'smma', 'handle_na':True},
            {'length':7, 'smooth':'ema', 'handle_na':True},
        )
        precalc = {
            abs(self.data.index-pd.Timestamp('2020-06-20').timestamp()/86400).argmin(): # 20 June 2020
                (368.01,271.18),
            abs(self.data.index-pd.Timestamp('2023-01-12').timestamp()/86400).argmin(): # 12 January 2023
                (392.45,585.55),
            abs(self.data.index-pd.Timestamp('2025-08-21').timestamp()/86400).argmin(): # 21 August 2025
                (2850.91,2814.91),
        }

        self._assert_equal(indicators.idct_atr, precalc=precalc, params=params, pass_all=True)

    def test_idct_atr_lookahead(self) -> None:
        """
        Test 'idct_atr'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'length':14, 'smooth':'smma', 'handle_na':True},
            {'length':7, 'smooth':'ema', 'handle_na':True},
            {'length':50, 'smooth':'wma', 'handle_na':False},
            {'length':1, 'smooth':'sma', 'handle_na':True},
        )
        self._assert_lookahead(indicators.idct_atr, params=params, pass_all=True)

    def test_idct_atr_invalid(self) -> None:
        """
        Test 'idct_atr'

        Verify that the function throws an error when sending incorrect parameters.
        """

        base = {'length':14, 'smooth':'smma', 'handle_na':True}
        overrides = (
            {'length':0}, {'length':-5}, {'length':None},
            {'smooth':'NONE'}
        )
        params = [{**base, **o} for o in overrides]

        self._assert_raises(indicators.idct_atr, params=params, pass_all=True)

    def test_idct_trange_lookahead(self) -> None:
        """
        Test 'idct_trange'

        Verify that the indicator does not have look-ahead bias
        """

        params = (
            {'handle_na':True},
            {'handle_na':False},
        )
        self._assert_lookahead(indicators.idct_trange, params=params, pass_all=True)

if __name__ == '__main__':
    unittest.main()
