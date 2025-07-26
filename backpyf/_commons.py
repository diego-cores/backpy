"""
Commons hidden module

This module contains all global variables for better manipulation.

Variables:
    alert (bool): If True, shows alerts in the console.
    dots (bool): If false, the '.' will be replaced by commas "," in prints.
    run_timer (bool): If false the execution timer will never appear in the console.
    max_bar_updates (int): Number of times the 'run' loading bar is updated, 
        a very high number will greatly increase the execution time. 
    lift (bool): Set to False if you don't want tkinter windows 
        to jump over everything else when running.

Hidden Variables:
    _icon: Icon currently used by the application (hidden variable).
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
    __plt_styles: Styles for coloring trading charts (hidden variable).
"""

import pandas as pd

alert = True
dots = True
run_timer = True

max_bar_updates = 1000

lift = True

__data_year_days = 365
__data_width_day = None
__data_interval = None
__data_width = None
__data_icon = None
__data = None

__trades = pd.DataFrame()
_init_funds = 0

_icon = None
__custom_plot = {}

__binance_timeout = 0.08

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
__plt_styles = {
    # 'bg','fr','btn' are required for each style.
    'lightmode':{
        'bg': '#e5e5e5', 
        'fr': 'SystemButtonFace', 
        'btn': '#000000',
        'btna': '#333333'
    },
    'darkmode':{
        'bg': '#1e1e1e', 
        'fr': '#161616', 
        'btn': '#ffffff', 
        'btna': '#333333', 
        'vol': 'gray'
    },

    # All properties are: 'bg', 'gdir', 'fr', 'btn', 'btna', 'vol', 'mk'.
    # light
    'sunrise': {
        'bg': ('#FFF7E6', '#FFDAB9'), 'gdir': True,
        'fr': '#FFF1D6', 'btn': '#FF8C42', 'btna': '#CC6E34',
        'vol': '#FFDAB9', 'mk': {'u': '#FFA94D', 'd': '#CC5C2B'},
    },
    'mintfresh': {
        'bg': '#E6FFF7', 'fr': '#D6FFF1', 'btn': '#3AB795', 'btna': '#2E9C7A',
        'vol': '#A8E6CF', 'mk': {'u': '#3AB795', 'd': '#2A7766'},
    },
    'skyday': {
        'bg': ('#D6F0FF', '#AEE4FF'), 'gdir': False,
        'fr': '#BEE7FF', 'btn': '#1E90FF', 'btna': '#166ECC',
        'vol': '#87CEFA', 'mk': {'u': '#1E90FF', 'd': '#104E8B'},
    },
    'lavenderblush': {
        'bg': '#F5E6FF', 'fr': '#EAD6FF', 'btn': '#A555FF', 'btna': '#863ACC',
        'vol': '#D8BFD8', 'mk': {'u': '#A555FF', 'd': '#6B2D99'},
    },
    'peachpuff': {
        'bg': ("#FFF1E6", "#FFD3B6", "#FFB085"), 'gdir': True,
        'fr': '#FFE6D6', 'btn': '#FF7043', 'btna': '#E35B33',
        'vol': '#FFA07A', 'mk': {'u': '#FF7043', 'd': '#CC4F2D'},
    },

    # dark
    'sunrisedusk': {
        'bg': '#2B1B12', 'fr': '#3A2618', 'btn': '#FF8C42', 'btna': '#CC6E34',
        'vol': "#B65426", 'mk': {'u': '#FFA94D', 'd': '#8B3E1D'},
    },
    'embernight': {
        'bg': ('#000000', '#1A0000', '#330000'), 'gdir': False,
        'fr': '#0A0A0A', 'btn': '#E20000', 'btna': '#990000',
        'vol': '#8B0000', 'mk': {'u': '#FF6347', 'd': '#8B0000'},
    },
    'obsidian': {
        'bg': '#03000F', 'fr': '#010008', 'btn': '#b748fc', 'btna': '#9B38D6',
        'vol': '#7B68EE', 'mk': {'u': '#b748fc', 'd': '#5A1E7C'},
    },
    'neonforge': {
        'bg': ('#000912', '#001B2D', '#003347'), 'gdir': True,
        'fr': '#001B2D', 'btn': '#00FFF7', 'btna': '#00BBAF',
        'vol': '#00CED1', 'mk': {'u': '#00FFF7', 'd': '#009E9A'},
    },
    'carbonfire': {
        'bg': '#1A0000', 'fr': '#0D0000', 'btn': '#FF4500', 'btna': '#CC3700',
        'vol': '#CD5C5C', 'mk': {'u': '#FF6347', 'd': '#8B0000'},
    },
    'datamatrix': {
        'bg': ('#000A00', '#002200'), 'gdir': False,
        'fr': '#001500', 'btn': '#00FF00', 'btna': '#00CC00',
        'vol': '#32CD32', 'mk': {'u': '#00FF00', 'd': '#006400'},
    },
    'terminalblood': {
        'bg': '#0F0000', 'fr': '#080000', 'btn': '#ff3b3f', 'btna': '#CC2E32',
        'vol': '#B22222', 'mk': {'u': '#ff3b3f', 'd': '#800000'},
    },
    'plasmacore': {
        'bg': ('#170028', '#2B0040', '#3C0066'), 'gdir': True,
        'fr': '#250040', 'btn': '#E84FFF', 'btna': '#C23AD9',
        'vol': '#DA70D6', 'mk': {'u': '#E84FFF', 'd': '#9400D3'},
    }
}
