"""
Commons Test
"""

from backpy import _commons as _cm
import unittest

import pandas as pd

class TestCommons(unittest.TestCase):
    def setUp(self) -> None:
        test:dict = {
            'name':'test',
            'trades':pd.DataFrame([{
                'date':1,
                'positionOpen':1,
                'commission':0.08,
                'amount':50,
                'typeSide':True,
                'unionId':1122334455667788,
                'positionClose':2,
                'positionDate':2,
                'profitPer':10,
                'profit':10,
            }, {
                'date':2,
                'positionOpen':1,
                'commission':0.08,
                'amount':50,
                'typeSide':True,
                'unionId':2233445566778899,
                'positionClose':2,
                'positionDate':3,
                'profitPer':10,
                'profit':10,
            }]),
            'balance_rec':[60,70],
            'init_funds':50,
            'd_year_days':365,
            'd_width_day':1,
            'd_width':1,
        }

        test1 = {k:v for k,v in test.items()}
        test1['name'] = 'test1'
        test2 = {k:v for k,v in test.items()}
        test2['name'] = 'test2'

        self.backtest = [test, test1, test2]
        setattr(_cm, '__backtests', self.backtest)

    def test_get_backtest_names(self) -> None:
        """
        Test 'get_backtest_names'

        Verify that when requesting backtest names, the same result as '__get_names' was returned.
        """

        self.assertEqual(_cm.get_backtest_names(), getattr(_cm, '__get_names')(getattr(_cm, '__backtests')))

    def test__get_names(self) -> None:
        """
        Test '__get_names'

        Verify that the backtest names are returned correctly.
        """

        self.assertEqual(getattr(_cm, '__get_names')(getattr(_cm, '__backtests')), 
                         [i['name'] for i in self.backtest])

    def test__get_dtrades_all(self) -> None:
        """
        Test '__get_dtrades'

        Verify that all trades are returned correctly.
        """

        backtest_names = getattr(_cm, '__get_names')(getattr(_cm, '__backtests'))
        result = getattr(_cm, '__get_dtrades')(backtest_names)

        for i,v in enumerate(backtest_names):
            pd.testing.assert_frame_equal(result[v], self.backtest[i]['trades'])

    def test__get_dtrades_unq(self) -> None:
        """
        Test '__get_dtrades'

        Verify that the trades from a single backtest are returned correctly.
        """

        result = getattr(_cm, '__get_dtrades')(None)
        for k in result.keys():
            pd.testing.assert_frame_equal(result[k], {self.backtest[-1]['name']:self.backtest[-1]['trades']}[k])

    def test__get_trades_all(self) -> None:
        """
        Test '__get_trades'

        Verify that all trades are returned correctly.
        """

        trades:list[pd.DataFrame] = [self.backtest[-1]['trades']] * 3
        all_trades = pd.concat(trades, ignore_index=True)
        all_trades = all_trades.sort_values(by='positionDate', 
                                            ascending=True).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            getattr(_cm, '__get_trades')(getattr(_cm, '__get_names')(getattr(_cm, '__backtests'))), 
            all_trades)

    def test__get_trades_unq(self) -> None:
        """
        Test '__get_trades'

        Verify that the trades from a single backtest are returned correctly.
        """

        pd.testing.assert_frame_equal(getattr(_cm, '__get_trades')(None), self.backtest[-1]['trades'])

    def test__get_strategy_n(self) -> None:
        """
        Test '__get_strategy'

        Verify that '__get_strategy' works with 'None'.
        """

        self.assertEqual(getattr(_cm, '__get_strategy')(None), self.backtest[-1])

    def test__get_strategy_i(self) -> None:
        """
        Test '__get_strategy'

        Verify that '__get_strategy' works with index.
        """

        self.assertEqual(getattr(_cm, '__get_strategy')(0), self.backtest[0])

    def test__get_strategy_s(self) -> None:
        """
        Test '__get_strategy'

        Verify that '__get_strategy' works with string.
        """

        backtest_names = getattr(_cm, '__get_names')(getattr(_cm, '__backtests'))
        self.assertEqual(getattr(_cm, '__get_strategy')(backtest_names[0]), self.backtest[0])

    def test__gen_fname(self) -> None:
        """
        Test '__gen_fname'

        Verify that the new names are generated correctly.
        """

        mod_backtest = [{'name':'test'},{'name':'test1'},{'name':'test2'}]

        self.assertEqual(getattr(_cm, '__gen_fname')('test', from_=mod_backtest), 'test3')
        self.assertEqual(getattr(_cm, '__gen_fname')('test_', from_=mod_backtest), 'test_')

if __name__ == '__main__':
    unittest.main()
