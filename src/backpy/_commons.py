"""
Commons hidden module

This module contains all global variables for better manipulation.

Variables:
    logger (Logger): Logger variable.
    dots (bool): If false, the '.' will be replaced by commas "," in prints.
    run_timer (bool): If false the execution timer will never appear in the console.
    plt_style (str | None): Last style used, if you modify this variable 
        and put one that does not exist it will give an error.
    max_bar_updates (int): Number of times the 'run' loading bar is updated, 
        a very high number will greatly increase the execution time. 
    lift (bool): Set to False if you don't want tkinter windows 
        to jump over everything else when running.

Hidden Variables:
    _random_titles: Random titles for windows (hidden variable).
    __panel_list: List of windows that will be joined into panels (hidden variable).
    __panel_wmax = Maximum number of panels; if a value greater than 4 is given, an error will occur (hidden variable).
    __linked_toolbars = List of connected toolbars (hidden variable).
    __min_gap: If left as True, gaps will not be calculated on the entry 
        of 'taker' orders (hidden variable).
    __limit_ig: If in a 'stopLimit' or 'takeLimit' the order is within the 
        same candle and this is False, it will be executed (hidden variable).
    __init_funds: Initial capital for the backtesting (hidden variable).
    __commission: Commission of each execution (hidden variable).
    __spread_pct: Market spread percentage (hidden variable).
    __slippage_pct: Slippage percentage (hidden variable).
    __orders_order: Dictionary with values to sort the order type when 
        executing (hidden variable).
    __orders_nclose: If True, orders are not ordered to be executed based 
        on the closest one (hidden variable).
    __chunk_size: Size of each chunk of the engine (hidden variable).
    __data_year_days: Number of operable days in 1 year (hidden variable).
    __data_width_day: Width of the day (hidden variable).
    __data_interval: Interval of the loaded data (hidden variable).
    __data_width: Width of the dataset (hidden variable).
    __data_icon: Data icon (hidden variable).
    __data: Loaded dataset (hidden variable).
    __backtests: List of data of each backtest, 
        containing trades and data needed for statistics (hidden variable).
    __custom_plot: Dict of custom graphical statistics (hidden variable).
    __binance_timeout: Time out between each request to the binance api 
        (hidden variable).
    __COLORS: Dictionary with printable colors (hidden variable).
    __plt_styles: Styles for coloring trading charts (hidden variable).

Functions:
    get_backtest_names: Takes the names of the saved backtests.
    del_backtest: Remove a backtest.
    c_tf: It's the same as doing: cast(float, ...).

Hidden Functions:
    __del_backtest_uniq: Remove only one backtest.
    __get_names: Takes the names of an list of dictionaries.
    __get_trades: Take trades from 1 or more saved backtests.
    __get_dtrades: Does the same thing as '__get_trades' 
        but saves each backtest in a different key in a dict.
    __get_strategy: Take data from a backtest.
    __gen_fname: Generates a name that is not duplicated in '__backtests'.
"""

from typing import Any, Callable, cast
import pandas as pd
import logging

from backpy.flex_data import CostsValue
from . import exception

logger:logging.Logger = logging.getLogger(__name__)

dots:bool = True
run_timer:bool = True
plt_style:str|None = None

max_bar_updates:int = 1_000

lift:bool = True
_random_titles:list = [
    'Python > Others',
    'Nice strategy',
    'Python window',
    'BackPy > ⚡',
    'Many trades',
    'loading...',
    'Backtest',
    'Panels!',
    'Tkinter',
    'BackPy',
    '🚀',
]

__data_year_days:int = 365
__data_width_day:None|float = None
__data_interval:None|str = None
__data_width:None|float = None
__data_icon:None|str = None
__data:None|pd.DataFrame = None
__backtests:list = []

__anim_run:bool = True
__panel_list:list = []
__panel_wmax:int = 4
__linked_toolbars:list = []

__min_gap:None|bool = None
__limit_ig:None|bool = None
__init_funds:None|int = 100
__commission:None|CostsValue = None
__spread_pct:None|CostsValue = None
__slippage_pct:None|CostsValue = None
__orders_order:None|dict = None
__orders_nclose:None|bool = None
__chunk_size:None|int = None

__custom_plot:dict = {}

__binance_timeout:float = 0.08

__COLORS:dict[str, str] = {
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
__plt_styles:dict = {
    # 'bg','fr','btn' are required for each style.
    'lightmode':{
        'bg': '#e5e5e5', 
        'fr': 'SystemButtonFace', 
        'btn': '#000000',
        'btna': "#FFFFFF"
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
        'vol': "#FFC898", 'mk': {'u': '#FFA94D', 'd': '#CC5C2B'},
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
    'emberday': {
        'bg': ("#f0f0f0", "#e5e5e5", "#dfdfdf"), 'gdir': True,
        'fr': '#0A0A0A', 'btn': "#FF6347", 'btna': "#DF2828",
        'vol': "#FF806A", 'mk': {'u': '#FF6347', 'd': "#CF0000"},
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

def del_backtest(names:list[str|int|None]|str|int|None = None) -> None:
    """
    Delete backtests

    Remove a backtest; each backtest can consume a lot of memory, 
    so if you don't need it, it's best to remove it.

    Args:
        names (list[str|int|None]|str|int|None, optional): 
            Name or index of the backtests to be deleted.
    """

    if len(__backtests) == 0:
        raise exception.DataError('There is no backtest to delete.')
    elif isinstance(names, tuple):
        raise exception.DataError("'names' cannot be a tuple.")

    if isinstance(names, list):
        nums:list[int] = sorted((x for x in names if isinstance(x, int)), reverse=True)
        strings:list[str] = [x for x in names if isinstance(x, str)]
        sorted_names = nums + strings

        for v in sorted_names: __del_backtest_uniq(name=v)
        return

    __del_backtest_uniq(name=names)

def get_backtest_names() -> list[str]:
    """
    Get names

    Takes the names of the saved backtests.

    Returns:
        list[str]: names
    """

    return __get_names(__backtests)

def __del_backtest_uniq(name:str|int|None = None) -> None:
    """
    Delete only one backtests

    Remove a backtest; each backtest can consume a lot of memory, 
    so if you don't need it, it's best to remove it.

    Args:
        name (str|int|None, optional): 
            Name or index of the backtests to be deleted.
    """

    if isinstance(name, int) or name is None:
        del __backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return __backtests[-1]

    for i, backtest in enumerate(__backtests):
        if backtest['name'] == name:
            del __backtests[i]

def __get_names(from_:list[dict]) -> list[str]:
    """
    Get names

    Takes the names of the 'from' list of dictionaries.
    'from' needs 'name' key.

    Args:
        from (list[dict], optional): List of dictionaries 
            from which the names will be obtained.

    Returns:
        list[str]: names
    """

    return [i['name'] for i in from_]

def __get_dtrades(names:list[str|int|None]|str|int|None = None) -> dict:
    """
    Get trades dict

    Take trades from 1 or more saved backtests.

    Trades will be sorted ascending based on 'positionDate'.

    One key per backtest.

    Args:
        names (list[str|int|None]|str|int|None, optional): You can pass an 
            integer index, a name, or a list of both; duplicates 
            are not allowed, None = -1.

    Returns:
        dict: trades.
    """

    trades = {
        (g_st:=__get_strategy(i))['name']: g_st['trades'].sort_values(
            by="positionDate", ascending=True).reset_index(drop=True)
        for i in (set(names or {None}) if not isinstance(names, (str, int))  else [names])
    }

    return trades

def __get_trades(names:list[str|int|None]|str|int|None = None) -> pd.DataFrame:
    """
    Get trades

    Take trades from 1 or more saved backtests.

    Trades will be sorted ascending based on 'positionDate'.

    Args:
        names (list[str|int|None]|str|int|None, optional): You can pass an 
            integer index, a name, or a list of both; duplicates 
            are not allowed, None = -1.

    Returns:
        DataFrame: trades
    """

    trades = pd.DataFrame()
    for i in (set(names or {None}) if not isinstance(names, (str, int)) else [names]):
        trades = pd.concat([trades, __get_strategy(i)['trades']])

    if not trades.empty:
        col = "positionDate" if "positionDate" in trades.columns else "positionOpen"

        trades = trades.sort_values(
            by=col, ascending=True).reset_index(drop=True)
    return trades

def __get_strategy(name:str|int|Any|None = None) -> dict:
    """
    Get strategy

    Take data from a backtest.

    Args:
        name (str|int|Any|None, optional): 
            Strategy name or index, None and Any = -1.

    Returns:
        dict: Dictionary with the following keys: 'name', 'trades', 
            'balance_rec', 'init_funds', 'd_year_days', 'd_width_day', 'd_width'.
    """

    if len(__backtests) == 0:
        return {'name':None, 
                'trades':pd.DataFrame(), 
                'balance_rec':pd.Series(),
                'init_funds':0, 
                'd_year_days':0, 
                'd_width_day':0, 
                'd_width':0}
    elif isinstance(name, int) or name is None:
        return __backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return __backtests[-1]

    for i,v in enumerate(__backtests):
        if v['name'] == name:
            return __backtests[i]

    raise exception.DataError('Name not found.')

def __gen_fname(name:str, from_:list[dict]) -> str:
    """
    Generate frame name

    Generates a name based on 'name' that is not duplicated in 'from'.

    Args:
        names (str, optional): Strategy name.
        from (list[dict], optional): List of dictionaries 
            from which the names will be obtained.

    Returns:
        str: Name not duplicated.
    """

    if len(__backtests) == 0:
        return name

    names = __get_names(from_=from_)
    mname = name
    nm = 1

    while mname in names:
        mname = f"{name}{nm}"
        nm += 1

    return mname

c_tf:Callable = lambda x: cast(float, x)
