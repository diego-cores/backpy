"""
Commons hidden module

This module contains all global variables for better manipulation.

Variables:
    logger (Logger): Logger variable.
    dots (bool): If false, the '.' will be replaced by commas "," in prints.
    run_timer (bool): If false the execution timer will never appear in the console.
    plt_style (str | None): Last style used, if you modify this variable 
        and put one that does not exist it will give an error.
    mpl_warning_supp (bool): Suppresses ignorable warnings from matplotlib.
    max_bar_updates (int): Number of times the 'run' loading bar is updated, 
        a very high number will greatly increase the execution time.
    graph_legend_fontsize (str|int): Font size in price graph legends.
        int or 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    graph_legend_pname (bool): Displays a legend for 'price' and 'volume' in price graph.
    graph_legend_rname (bool): If True, the registered name will be used; otherwise, 
        the function name will be used.
    graph_legend_args (bool): Show the indicator arguments in the legend.
    graph_panel_order (dict[str, float]): Graph order of the panels by name, 
        default: 99. Lower value renders higher.
    graph_first_size (int): Size of the first panel relative to the others 
        when graph the price. It cannot be less than 1.
    lift (bool): Set to False if you don't want tkinter windows 
        to jump over everything else when running.

Hidden Variables:
    _random_titles: Random titles for windows (hidden variable).
    __panel_list: List of windows that will be joined into panels (hidden variable).
    __panel_wmax: Maximum number of panels; if a value greater than 4 is given, 
        an error will occur (hidden variable).
    __linked_toolbars: List of connected toolbars (hidden variable).
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
    __nper_commission: Non-percentage commission cost (hidden variable).
    __data_backtests: List of data of each backtest, 
        containing trades and data needed for statistics (hidden variable).
    __data_year_days: Number of operable days in 1 year (hidden variable).
    __data_width_day: Width of the day (hidden variable).
    __data_interval: Interval of the loaded data (hidden variable).
    __data_width: Width of the dataset (hidden variable).
    __data_icon: Data icon (hidden variable).
    __data: Loaded dataset (hidden variable).
    __custom_plot: Dict of custom graphical statistics (hidden variable).
    __plot_indicators: Indicators saved for plotting (hidden variable).
    __binance_timeout: Time out between each request to the binance api 
        (hidden variable).
    __COLORS: Dictionary with printable colors (hidden variable).
    __plt_styles: Styles for coloring trading charts (hidden variable).
    __plot_indicators_def: Indicators default config for plotting 
        More info in the definition. func.__name__:dict (hidden variable).

Functions:
    c_tf: It's the same as doing: cast(float, ...).
    get_backtest_names: Takes the names of the saved backtests.
    del_backtest: Remove a backtest.
    draw_plot: Draw a line.
    drawax_hline: Draw a axis line.
    drawax_btw: Draw a axis area.
    draw_btw: Draw a area.
    draw_btw_pathcll: Draw an area using a path.
    draw_hist: Draw histogram.

Hidden Functions:
    _store_decorator: Give '_store' attribute to a function.
    __del_backtest_uniq: Remove only one backtest.
    __get_names: Takes the names of an list of dictionaries.
    __get_trades: Take trades from 1 or more saved backtests.
    __get_dtrades: Does the same thing as '__get_trades' 
        but saves each backtest in a different key in a dict.
    __get_strategy: Take data from a backtest.
    __gen_fname: Generates a name that is not duplicated in '__data_backtests'.
"""

from matplotlib.collections import PathCollection, PatchCollection, PolyCollection
from typing import Any, Callable, Sequence, Collection, cast
from matplotlib.patches import Rectangle, Polygon, Patch
from importlib.metadata import version
from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes
from matplotlib.path import Path
import pandas as pd
import numpy as np
import logging

from backpy.flex_data import CostsValue
from . import exception

logger:logging.Logger = logging.getLogger(__name__)
c_tf:Callable = lambda x: cast(float, x)

dots:bool = True
run_timer:bool = True
plt_style:str|None = None
max_bar_updates:int = 1_000
mpl_warning_supp:bool = True

graph_legend_fontsize:str|int = 'xx-small'
graph_legend_pname:bool = True
graph_legend_rname:bool = True
graph_legend_args:bool = True
graph_panel_order:dict[str, float] = {'price':0, 'volume':1}
graph_first_size:int = 2

lift:bool = True
_random_titles:list = [
    f'BackPy v{version("backpyf")}',
    'Python > Others',
    'Nice strategy',
    'Python window',
    'BackPy > ⚡',
    'Indicators!',
    'Many trades',
    'loading...',
    'Backtest',
    'Panels!',
    'Tkinter',
    'BackPy',
    '🚀',
]

__data_backtests:list = []
__data_year_days:int = 365
__data_width_day:None|float = None
__data_interval:None|str = None
__data_width:None|float = None
__data_icon:None|str = None
__data:None|pd.DataFrame = None

__anim_run:bool = True
__panel_list:list = []
__panel_wmax:int = 4
__linked_toolbars:list = []

__min_gap:None|bool = None
__limit_ig:None|bool = None
__chunk_size:None|int = None
__init_funds:None|float = 100
__commission:None|CostsValue = None
__spread_pct:None|CostsValue = None
__slippage_pct:None|CostsValue = None
__orders_order:None|dict = None
__orders_nclose:None|bool = None
__nper_commission:None|bool = None

__custom_plot:dict = {}
__plot_indicators:dict[str, dict] = {}

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

def draw_plot(value_index:int, **kwargs) -> Callable:
    """
    Draw plot

    Draw a line as decoration.

    Info:
        Draws the decoration of an indicator; it is passed the wrapper arguments. 
        It must return None, a list of collections, or a matplotlib collection to draw the legend.

    Wrapper args:
        ax (Axes): Axis where draw.
        index (Sequence): x index.
        values (Sequence): All indicator values.
        width (float): Width of each point.
        zorder (float): Position z of the element.

    Args:
        value_index (int): Index of the value to draw.
        **kwargs: Extra arguments passed to the 'plot' function.

    Return:
        Callable: Wrapper.
    """
    
    def wrapper(ax:Axes, index:Sequence, values:Sequence, zorder:float=0.4, **kn_) -> list:

        return ax.plot(index, values[value_index], **kwargs, zorder=zorder)
    return wrapper

def drawax_hline(cord:float, color:str, **kwargs) -> Callable:
    """
    Draw horizontal axis line

    Draw a line as decoration.
    For more info read 'draw_plot' docstring.

    Args:
        cord (float): Line coordinate.
        color (str): Color of the line.
        **kwargs: Extra arguments passed to the 'axhline' function.

    Return:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, *a_, zorder:float=0.4, **kn_) -> Line2D:
        return ax.axhline(cord, color=color, **kwargs, zorder=zorder)
    return wrapper

def drawax_btw(btw:tuple[float, float], color:str, **kwargs) -> Callable:
    """
    Draw axis area

    Draw a area as decoration.
    For more info read 'draw_plot' docstring.

    Args:
        btw (tuple[float, float]): Tuple with the coordinates between which it will be drawn.
        color (str): Color of the area.
        **kwargs: Extra arguments passed to the 'axhspan' function.

    Return:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, *a_, zorder:float=0.4, **kn_) -> Polygon:

        return ax.axhspan(*btw, color=color, **kwargs, zorder=zorder)
    return wrapper

def draw_btw(btw_index:tuple[int, int], color:str|Sequence, **kwargs) -> Callable:
    """
    Draw area

    Draw a area as decoration.
    For more info read 'draw_plot' docstring.

    Args:
        btw (tuple[int, int]): Tuple with the index of value between which it will be drawn.
        color (str|Squence): Color of the area. If a list is passed, the first one will be 
            used when they move away and the second one when they approach.
        **kwargs: Extra arguments passed to the 'fill_between' functions.

    Return:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence, values:Sequence, 
        zorder:float=0.4, **kn_) -> list[PolyCollection]:

        collections = []

        get_values = [np.array(values[i]) for i in btw_index][:2]
        hist = get_values[0]-get_values[1]

        cpos, cneg = (
            (color, color) if isinstance(color, str) or len(color) < 2 
            else (color[0], color[1]))

        collections.insert(0, ax.fill_between(
            index,
            *get_values, # type: ignore
            where=(hist>=0),
            interpolate=True,
            color=cpos,
            zorder=zorder,
            **kwargs
            ))

        collections.insert(1, ax.fill_between(
            index,
            *get_values, # type: ignore
            where=(hist<0),
            interpolate=True,
            color=cneg,
            zorder=zorder,
            **kwargs
            ))

        return collections
    return wrapper

def draw_btw_pathcll(btw_index:tuple[int, int], color:str|Sequence, color_d:str|Sequence) -> Callable:
    """
    Draw area 

    Draw an area using a path.
    For more info read 'draw_plot' docstring.

    Args:
        btw (tuple[int, int]): Tuple with the index of value between which it will be drawn.
        color (str|Squence): Color of the area. If a list is passed, the first one will be 
            used when they move away and the second one when they approach.
        color_d (str|Sequence): It works the same as the 'color' argument but when the difference is negative.

    Return:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence, values:Sequence, 
        zorder:float=0.4, **kn_) -> Collection:

        get_values =[np.array(values[i]) for i in btw_index][:2]
        cpos, cneg = (
            (color, color) if isinstance(color, str)
            else (color[0], color[1 if len(color) > 1 else 0]))
        cposd, cnegd = (
            (color_d, color_d) if isinstance(color_d, str)
            else (color_d[0], color_d[1 if len(color_d) > 1 else 0]))

        paths, face_colors = [], []

        for i in range(len(get_values[0])-1):
            prev = get_values[0][i]-get_values[1][i]
            hist = get_values[0][i+1]-get_values[1][i+1]

            if np.isnan(hist):
                continue

            fc = ((cpos if hist > prev else cposd) if hist >= 0 
                else (cneg if hist < prev else cnegd))

            x0, x1 = index[i], index[i+1]
            verts = [
                (x0, get_values[1][i]), (x0, get_values[0][i]),
                (x1, get_values[0][i+1]), (x1, get_values[1][i+1]),
            ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

            paths.append(Path(verts, codes))
            face_colors.append(fc)

        coll = ax.add_collection(PathCollection( # type: ignore
            paths, facecolors=face_colors, linewidths=1.2, zorder=zorder))
        ax.autoscale_view()

        return coll
    return wrapper

def draw_hist(value_index:int, color:str|Sequence, color_d:str|Sequence, **kwargs) -> Callable:
    """
    Draw histogram 

    For more info read 'draw_plot' docstring.

    Args:
        value_index (int): Index of the value to draw the hist.
        color (str|Squence): Color of the hist. If a list is passed, the first one will be 
            used when the histogram grows and the second one when it decreases.
        color_d (str|Sequence): It works the same as the 'color' argument but when the hist is negative.
        **kwargs: Extra arguments passed to the 'fill_between' functions.
    
    Return:
        Callable: Wrapper.
    """
    
    def wrapper(ax:Axes, index:Sequence, values:Sequence, 
        width:float, zorder:float=0.4, **kn_) -> list[Patch]:

        y = np.array(values[value_index])
        x = np.array(index) - width / 2
        diff = np.insert(y, 0, 0)

        color_up = color 
        color_up, color_down = (
            (color, color) if isinstance(color, str) 
            else (color[0], color[1 if len(color) > 1 else 0]))
        color_upd, color_downd = (
            (color_d, color_d) if isinstance(color_d, str)
            else (color_d[0], color_d[1 if len(color_d) > 1 else 0]))

        color_list = [(color_up if y[i] >= diff[i] else color_upd) if y[i] > 0 
            else (color_downd if y[i] > diff[i] else color_down) for i in range(len(y))]

        patches = [Rectangle((xi, 0), width, yi) for xi, yi in zip(x, y)]
        ax.add_collection(PatchCollection(patches, color=color_list, # type: ignore
            linewidth=0, zorder=zorder))
        ax.autoscale_view()

        # PatchCollection not have a handler
        return [
            Patch(facecolor=color_up, linewidth=0),
            Patch(facecolor=color_down, linewidth=0)
        ]
    return wrapper

__plot_indicators_def:dict = {
    'def':{'panel':None, 'color':[None], 'dtSource':'close'},
    # All properties are: 'panel', 'color', 'dtSource', 'styleOnDraw', 'style'.

    #func.__name:{
    # 'panel': Panel where the indicator goes: 'price', 'volume' or other indicator.
    # 'color': List of colors for each column returned by the indicator.
    # 'dtSouce': If the indicator only uses one column of data, 
    #    this can be used to specify which column to use.
    # 'styleOnDraw': List of functions to draw decoration apart from drawing the data.
    # 'style': List of functions to indicate which function is used to draw each column. 
    #    Leave as None to use 'draw_plot' or 'lambda *args, **kwargs: None' to not draw.
    #}

    # For more info read the 'draw_plot' docstring.
    'idct_ema':{'panel':'price', 'color':['blue'],'dtSource':'close',},
    'idct_bb':{'panel':'price', 'color':['#f23645', '#2D2DFF', '#089981'], 
        'dtSource':'close',
        'style':[
            draw_btw((0,2), '#6565FF12')
        ]},
    'idct_rsi':{'panel':None, 'color':['#9B40C8', '#FFF700'], 
        'dtSource':'close',
        'style':[
            drawax_hline(70, '#FFFFFF36', linestyle=(0, (1.5,1))), 
            drawax_hline(30, '#FFFFFF36', linestyle=(0, (1.5,1))), 
            drawax_hline(50, '#FFFFFF36', linestyle=(0, (1.5,1))), 
            drawax_btw((70, 30), '#BB00D824',),
        ]},
    'idct_macd':{'panel':None, 'color':['blue', 'orange'], 
        'dtSource':'close',
        'styleOnDraw':[
            None, None, 
            draw_hist(2, color=('#089981', '#f23645'), color_d=('#B0D8D5', '#ffcdd2'))
        ], 
    },  
    'idct_sqzmom':{'panel':None, 'color':[None],
        'styleOnDraw':[
            None, 
            draw_hist(1, color=('#0AAF0A', '#AF0909'), color_d=('#0A5D0A', '#5C0909')),
        ], 
    },
    'idct_ichimoku':{'panel':'price', 'color':['#C4E293', '#E29393', '#2450C0', '#C0110B'], 
        'style':[
            draw_btw((0,1), ('#81B13324', '#AA2F2F24'))
        ]},
}

def _store_decorator(func:Callable) -> Callable:
    """
    Store decorator

    Decorate a function with this to give it 
        the attribute: '_store' and have it decorated with '__data_store'.

    Note:
        Add the 'cut' and 'last' arguments; they are used for '__data_store'.
        cut (bool): Cut off the data, yes or no, depending on whether the 
            user requests it or another indicator. Default False.
        last (int): Cut the data as the user wishes. Default = None.
        The implementation is made in the '__data_store' decorator.

    Args:
        func (Callable): Function.

    Returns:
        Callable: Function.
    """

    setattr(func, '_store', True)
    return func

def del_backtest(names:Sequence[str|int|None]|str|int|None = None) -> None:
    """
    Delete backtests

    Remove a backtest; each backtest can consume a lot of memory, 
    so if you don't need it, it's best to remove it.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): 
            Name or index of the backtests to be deleted.
    """

    if len(__data_backtests) == 0:
        raise exception.DataError('There is no backtest to delete.')
    elif isinstance(names, tuple):
        raise exception.DataError("'names' cannot be a tuple.")

    if not isinstance(names, str) and isinstance(names, Sequence):
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

    return __get_names(__data_backtests)

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
        del __data_backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return __data_backtests[-1]

    for i, backtest in enumerate(__data_backtests):
        if backtest['name'] == name:
            del __data_backtests[i]

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

def __get_dtrades(names:Sequence[str|int|None]|str|int|None = None) -> dict:
    """
    Get trades dict

    Take trades from 1 or more saved backtests.

    Trades will be sorted ascending based on 'positionDate'.

    One key per backtest.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): You can pass an 
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

def __get_trades(names:Sequence[str|int|None]|str|int|None = None) -> pd.DataFrame:
    """
    Get trades

    Take trades from 1 or more saved backtests.

    Trades will be sorted ascending based on 'positionDate'.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): You can pass an 
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

    if len(__data_backtests) == 0:
        return {'name':None, 
                'trades':pd.DataFrame(), 
                'balance_rec':pd.Series(),
                'init_funds':0, 
                'd_year_days':0, 
                'd_width_day':0, 
                'd_width':0}
    elif isinstance(name, int) or name is None:
        return __data_backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return __data_backtests[-1]

    for i,v in enumerate(__data_backtests):
        if v['name'] == name:
            return __data_backtests[i]

    raise exception.DataError('Name not found.')

def __gen_fname(name:str, from_:list[str]) -> str:
    """
    Generate frame name

    Generates a name based on 'name' that is not duplicated in 'from'.

    Args:
        names (str): Name.
        from_ (list[str]): List of names to not repeat

    Returns:
        str: Name not duplicated.
    """

    if len(from_) == 0:
        return name

    mname = name
    nm = 1

    while mname in from_:
        mname = f"{name}{nm}"
        nm += 1

    return mname
