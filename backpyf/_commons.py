"""
Commons hidden Module.

This module contains all global variables for better manipulation.

Variables:
    alert (bool): If True, shows alerts in the console.
    extract_without (bool): If it is true, the trades that are opened without 
        takeprofit or stoploss will closed.
    dots (bool): If false, the '.' will be replaced by commas "," in prints.
    run_timer (bool): If false the execution timer will never appear in the console.
    max_bar_updates (int): Number of times the 'run' loading bar is updated, 
        a very high number will greatly increase the execution time. 

Hidden Variables:
    _init_funds: Initial capital for the backtesting (hidden variable).
    __data_year_days: Number of operable days in 1 year (hidden variable).
    __data_width_day: Width of the day (hidden variable).
    __data_interval: Interval of the loaded data (hidden variable).
    __data_width: Width of the dataset (hidden variable).
    __data_icon: Data icon (hidden variable).
    __data: Loaded dataset (hidden variable).
    __trades: List of trades executed during backtesting (hidden variable).
    __custom_plot: Dict of custom graphical statistics (hidden variable).
    __binance_timeout: Time out between each request to the binance api (hidden variable).
    __COLORS: Dictionary with printable colors (hidden variable).
"""

import pandas as pd

alert = True
dots = True
run_timer = True

max_bar_updates = 1000
extract_without = False

__data_year_days = 365
__data_width_day = None
__data_interval = None
__data_width = None
__data_icon = None
__data = None

__trades = pd.DataFrame()
_init_funds = 0

__custom_plot = {}

__binance_timeout = 0.2

__COLORS = {
    'RED': "\033[91m",
    'GREEN': "\033[92m",
    'YELLOW': "\033[93m",
    'BLUE': "\033[94m",
    'MAGENTA': "\033[95m",
    'CYAN': "\033[96m",
    'WHITE': "\033[97m",
    'ORANGE': "\033[38;5;214m", # Only on terminals with 256 colors.
    'PURPLE': "\033[38;5;129m",
    'TEAL': "\033[38;5;37m",
    'GRAY': "\033[90m",
    'LIGHT_GRAY': "\033[37m",
    'BOLD': "\033[1m",
    'UNDERLINE': "\033[4m",
    'RESET': "\033[0m",
}
