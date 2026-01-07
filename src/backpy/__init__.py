"""
Back Test Py

BackPy is a library used to test strategies in the market.

Version:
    1.0.0.post1

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

from . import _commons

from .strategy import (
    StrategyClass,
    idc_decorator,
)

from .custom_plt import (
    def_style
)

from .flex_data import (
    ChunkWrapper,
    DataWrapper,
    CostsValue,
)

from .main import (
    load_binance_data_futures,
    plot_strategy_decorator,
    load_binance_data_spot,
    load_yfinance_data, 
    plot_strategy_add,
    default_logging,
    run_animation,
    plot_strategy,
    load_data_bpd,
    save_data_bpd,
    stats_trades,
    stats_icon,
    run_config, 
    load_data,
    plot, 
    run, 
)

from .stats import (
    monte_carlo_chart,
    monte_carlo_bsim,
    perf_tzone_chart,
    get_drawdowns,
    max_drawdown,
    correlation
)

from . import utils

__version__ = '1.0.0.post1'

__doc__ = """
BackPy documentation

BackPy is a module for backtesting data. 
You can create your own data or use 
    functions that extract data from other modules.

Important Notice: 
    Understanding the Risks of Trading and Financial Data Analysis.
    Trading financial instruments and using financial data for analysis 
    involves significant risks, including the possibility of loss of capital. 
    Markets can be volatile and data may contain errors. Before engaging in 
    trading activities or using financial data, it is important to understand 
    and carefully consider these risks and seek independent financial advice 
    if necessary.

Disclaimer Regarding Accuracy of BackPy:
    It is essential to acknowledge that the backtesting software 
    utilized for financial chart analysis may not be entirely 
    accurate and could contain errors, leading to results that 
    may not reflect real-world outcomes.

Disclaimer Regarding Financial Advice:
    It is crucial to emphasize that the backtesting software, including BackPy, 
    should not be construed as a substitute for professional financial advice. 
    While BackPy provides tools for financial data analysis, it does not 
    constitute financial advice or recommendations for trading decisions. 
    Users should exercise caution and seek advice from qualified financial 
    professionals before making any financial decisions based on the 
    results obtained from BackPy or any similar software.

Terms and Conditions:
    By using BackPy, you acknowledge that you have read and understood the 
    above notices and disclaimers and agree to abide by them. Your use of 
    BackPy constitutes your acceptance of these terms and conditions. If you 
    do not agree with these terms, you should not use BackPy.

"""

__all__ = [
    'load_binance_data_futures',
    'plot_strategy_decorator',
    'load_binance_data_spot',
    'load_yfinance_data',
    'monte_carlo_chart',
    'plot_strategy_add',
    'monte_carlo_bsim',
    'perf_tzone_chart',
    'default_logging',
    'StrategyClass',
    'idc_decorator',
    'plot_strategy',
    'get_drawdowns',
    'load_data_bpd',
    'save_data_bpd',
    'run_animation',
    'max_drawdown',
    'stats_trades',
    'ChunkWrapper',
    'DataWrapper',
    'correlation',
    'stats_icon',
    'CostsValue',
    'run_config',
    'load_data',
    'def_style',
    '_commons',
    'utils',
    'plot',
    'run',
]

__author__ = 'Diego Cores'
__url__ = 'https://github.com/diego-cores'
__email__ = '89626622+diego-cores@users.noreply.github.com'

import logging

_logger = logging.getLogger(__name__)
