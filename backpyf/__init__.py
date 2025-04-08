"""
Back Test Py.

BackPy is a library used to test strategies in the market.

Version:
    0.9.62b3

Repository:
    https://github.com/Diego-Cores/BackPy

License:
    MIT License

    Copyright (c) 2025 Diego

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from .strategy import StrategyClass
from ._commons import (
    __binance_timeout,
    extract_without,
    run_timer,
    alert,
    dots,
)
from .main import (
    plot_strategy_decorator,
    load_yfinance_data, 
    load_binance_data,
    plot_strategy_add,
    plot_strategy,
    stats_trades,
    stats_icon, 
    load_data, 
    plot, 
    run, 
    )

from .utils import utils
from .utils import (
    get_drawdowns,
    max_drawdown,
)

__doc__ = """
BackPy documentation.

BackPy is a library used to test strategies in the market. It allows you 
to provide your own data or use the Yfinance module.

Important Notice:
    Understanding the Risks of Trading and Financial Data Analysis.

    Trading financial instruments and using financial data for analysis 
    involves significant risks, including the possibility of loss of 
    capital. Markets can be volatile and data may contain errors. Before 
    engaging in trading activities or using financial data, it is important 
    to understand and carefully consider these risks and seek independent 
    financial advice if necessary.

Disclaimer Regarding Accuracy of BackPy:
    It is essential to acknowledge that the backtesting software utilized 
    for financial chart analysis may not be entirely accurate and could 
    contain errors, leading to results that may not reflect real-world 
    outcomes.

What can I do with BackPy?
    - Determine the position of different indicators for each point.
    - Create your own indicators based on price, date, and volume.
    - Consult previous data such as previous closings and active actions 
      for each point.
    - Display data with or without a logarithmic scale.
    - Print statistics of the uploaded data.
"""

__all__ = [
    'plot_strategy_decorator',
    'load_yfinance_data',
    'load_binance_data',
    '__binance_timeout',
    'plot_strategy_add',
    'extract_without',
    'StrategyClass',
    'plot_strategy',
    'get_drawdowns',
    'max_drawdown',
    'stats_trades',
    'stats_icon',
    'load_data',
    'run_timer',
    'utils',
    'alert',
    'plot',
    'dots',
    'run',
]
