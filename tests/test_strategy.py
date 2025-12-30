"""
Strategy Test
"""

from backpy.flex_data import DataWrapper, CostsValue
from backpy import _commons as _cm
from backpy import strategy as st

from unittest.mock import patch, MagicMock
import unittest

import pandas as pd

class TestFunctions(unittest.TestCase):
    def test_idc_decorator(self) -> None:
        """
        Test 'idc_decorator'

        Verify that it works correctly.
        """

        def empty_func() -> None: ...

        func = st.idc_decorator(empty_func)
        self.assertEqual(getattr(func, '_uidc'), True)

    def test_data_info(self) -> None:
        """
        Test '_data_info'

        Verify that it works correctly.
        """

        setattr(_cm, '__data_interval', 1)
        setattr(_cm, '__data_width', 1)
        setattr(_cm, '__data_icon', 1)

        self.assertEqual(st._data_info(), (1,1,1))

class TestStrategyClass(unittest.TestCase):
    def setUp(self) -> None:
        setattr(_cm, '__data_interval', 1)
        setattr(_cm, '__data_width', 1)
        setattr(_cm, '__data_icon', 'test')

        data = pd.DataFrame({
            'Low':[7,7,13],
            'High':[13,16,33],
            'Close':[10,15,30],
            'Open':[8,10,15],
            'Volume':[1,1,1]
        })

        class TestStrategyClass(st.StrategyClass):
            def next(self): ...

        self.instance = TestStrategyClass(data=data)
        setattr(self.instance, '_StrategyClass__idc_data', {})

    def tearDown(self) -> None:
        pass

    def test_tzs(self) -> None:
        """
        Test 'tzs'

        Verify that the time zone functions are working properly.
        """

        self.assertIsInstance(self.instance.tz_new_york(), bool)
        self.assertIsInstance(self.instance.tz_london(), bool)
        self.assertIsInstance(self.instance.tz_sydney(), bool)
        self.assertIsInstance(self.instance.tz_tokyo(), bool)

    def test_balance(self) -> None:
        """
        Test 'balance'

        Verify that the balance functions are working properly.
        """

        self.assertIsInstance(self.instance.get_balance(), (int, float))
        self.assertIsInstance(self.instance.get_balance_rec(), DataWrapper)

    def test_costs_ainit(self) -> None:
        """
        Test 'get' functions

        Verify that they return the correct type.
        """

        self.assertIsInstance(self.instance.get_spread(), CostsValue)
        self.assertIsInstance(self.instance.get_slippage(), CostsValue)
        self.assertIsInstance(self.instance.get_commission(), CostsValue)
        self.assertIsInstance(self.instance.get_init_funds(), (int, float))

    def test__store_decorator(self) -> None:
        """
        Test '__store_decorator'

        Verify that it works correctly.
        """

        def empty_func() -> None: ...

        func = getattr(st.StrategyClass, '_StrategyClass__store_decorator')(empty_func)
        self.assertEqual(getattr(func, '_store'), True)

    @patch("backpy.StrategyClass._StrategyClass__func_idg")
    @patch("backpy.StrategyClass._StrategyClass__data_cut")
    def test__data_store(self, mock_data_cut:MagicMock, mock_idg:MagicMock) -> None:
        """
        Test '__data_store'

        Verify that it works correctly.
        """

        def empty_func(x=0, cut=True) -> None: ...

        mock_idg.return_value = ('empty_func_id', {'cut': True})

        deco_func = getattr(st.StrategyClass, '_StrategyClass__data_store')(self.instance, empty_func)

        deco_func(10, cut=True)

        mock_idg.assert_called_once()
        mock_data_cut.assert_called_once()

        deco_func(10)

    @patch("backpy.StrategyClass._StrategyClass__func_idg")
    @patch("backpy.StrategyClass._StrategyClass__uidc_cut")
    def test__uidc(self, mock_uidc_cut:MagicMock, mock_idg:MagicMock) -> None:
        """
        Test '__uidc'

        Verify that it works correctly.
        """

        def empty_func(data, x=0) -> None: ...

        mock_idg.return_value = ('empty_func_id')

        func = empty_func
        deco_func = getattr(self.instance, '_StrategyClass__uidc')(func)

        deco_func(10)

        mock_idg.assert_called_once()
        mock_uidc_cut.assert_called_once()

        deco_func(10)

if __name__ == '__main__':
    unittest.main()
