"""
Main Test
"""

from backpy import strategy
from backpy import main

from pandas import DataFrame

import contextlib
import unittest
import io

class TestSmoke(unittest.TestCase):
    def test_smoke(self) -> None:
        """
        Test smoke

        This test uses the library in a basic way, to confirm that the module can be executed.
        """

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):

            main.load_data(DataFrame({
                'open':[1,3,5],
                'high':[3,5,7],
                'low':[1,2,4],
                'close':[2,4,6],
                'volume':[1,1,1]
            }))

            class MinTest(strategy.StrategyClass):
                def next(self):
                    if len(self.prev_positions('index')) > 0:
                        self.act_close(0)

                    self.act_taker(amount=100)

            main.run(MinTest)

if __name__ == '__main__':
    unittest.main()
