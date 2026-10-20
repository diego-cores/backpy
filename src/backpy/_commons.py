"""
Commons hidden module

This module contains all global attributes for better manipulation.

Variables:
    logger (Logger): Logger variable.
    TKINTER (bool): Constant to check if tkinter package is installed.
    COLORS (dict): Dictionary with printable colors.
    user_config (UserConfig): BackPy configuration attributes.
    data_state (DataState): Engine state attributes.
    plot_state (PlotState): Plot state attributes.

Classes:
    UserConfig: BackPy configuration attributes.
    DataState: Engine state attributes store.
    PlotState: Plot state attributes store.

Functions:
    get_commission_names: Get temporary commissions names.
    get_backtest_names: Takes the names of the saved backtests.
    del_backtest: Remove a backtest.
    draw_plot: Draw a line.
    draw_pointsln: Draw a points line as decoration.
    drawax_hline: Draw a axis line.
    drawax_btw: Draw a axis area.
    draw_btw: Draw a area.
    draw_btw_pathcll: Draw an area using a path.
    draw_hist: Draw histogram.

Hidden Functions:
    _get_version: Get backpyf version.
    _store_decorator: Give '_store' attribute to a function.
    _get_temp_commission: Get temporary commission.
    __add_temp_commission: Save a DataFrame with commission markers.
    __del_backtest_uniq: Remove only one backtest.
    __get_names: Takes the names of an list of dictionaries.
    __get_trades: Take trades from 1 or more saved backtests.
    __get_dtrades: Does the same thing as '__get_trades' 
        but saves each backtest in a different key in a dict.
    __get_strategy: Take data from a backtest.
    __gen_fname: Generates a name that is not duplicated in '__data_backtests'.
"""

from matplotlib.collections import PathCollection, PatchCollection, PolyCollection
from typing import Any, Callable, Sequence, Collection, TYPE_CHECKING, Generator
from importlib.metadata import version, PackageNotFoundError
from matplotlib.patches import Rectangle, Polygon, Patch
from dataclasses import dataclass, fields, field
from matplotlib.markers import MarkerStyle
from contextlib import contextmanager
from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes
from matplotlib.path import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import logging

from backpy.flex_data import CostsValue
from . import exception

logger:logging.Logger = logging.getLogger(__name__)

TKINTER = False
try:
    import tkinter as tk
    TKINTER = True
except ImportError:
    pass

COLORS:dict = {
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

def _get_version() -> str:
    """
    Get version

    Get backpyf version.

    Return:
        str: Installed version of backpyf, otherwise 'none'.
    """
    try:
        return version('backpyf')
    except PackageNotFoundError:
        return 'none'

@dataclass(slots=True)
class UserConfig():
    """
    User config

    BackPy configuration attributes.

    Note:
        These attribute only exist if tkinter package is installed:
        'lift', 'fps_label', 'fps_only_draw'.
        The tkinter package comes by default in a normal Python installation, but in some
        installations, as well as on systems without a graphical interface, it may not.
        To verify, you can use the boolean variable 'TKINTER'.

    Attributes:
        dots (bool): If false, the '.' will be replaced by commas "," in prints.
        run_timer (bool): If false the execution timer will never appear in the console.
        plt_style (str | None): Last style used, if you modify this variable 
            and put one that does not exist it will give an error.
        max_bar_updates (int): Number of times the 'run' loading bar is updated, 
            a very high number will greatly increase the execution time.
        mpl_warning_supp (bool): Suppresses ignorable warnings from matplotlib.
        lift (bool): Set to False if you don't want tkinter windows 
            to jump over everything else when running.
        fps_label (bool): Set to True to display an FPS counter.
        fps_only_draw (bool): Set to True to only mesure draw function.
        graph_legend_fontsize (str|int): Font size in price graph legends.
            int or 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
        graph_legend_pname (bool): Displays a legend for 'price' and 'volume' in price graph.
        graph_legend_rname (bool): If True, the registered name will be used; otherwise, 
            the function name will be used.
        graph_legend_args (bool): Show the indicator arguments in the legend.
        graph_integration (bool): If true, the plot will be integrated with the internal embed; 
            otherwise, it will be drawn with matplotlib.
        graph_panel_order (dict[str, float]): Graph order of the panels by name, 
            default: 99. Lower value renders higher.
        graph_first_size (int): Size of the first panel relative to the others 
            when graph the price. It cannot be less than 1.
        graph_dpi (int): DPI of the Matplotlib graph.
        binance_request_timeout (int): Binance request timeout in ms.
        
    Hidden Attributes:
        _max_elements (int): Maximum number of elements to return. 
            Used in functions with 'full_output' argument.
        _binance_timeout (float): Time out between each request to the binance api.

    Methods:
        to_dict: Return attributes as a dict.
        copy: Return a deepcopy of self.
        update: Modify the saved instance attributes.
        context: Context manager to modify attributes.
    """

    dots:bool = True
    run_timer:bool = True
    plt_style:str|None = None
    max_bar_updates:int = 1_000
    mpl_warning_supp:bool = True

    if TYPE_CHECKING or TKINTER:
        lift:bool = True
        fps_label:bool = False
        fps_only_draw:bool = False

    graph_legend_fontsize:str|int = 'x-small' 
    # 'xx-small' > 'x-small' > 'small' > 'medium' > 'large' > 'x-large' > 'xx-large'
    graph_legend_pname:bool = True
    graph_legend_rname:bool = True
    graph_legend_args:bool = True
    graph_integration:bool = True
    graph_panel_order:dict[str, float] = field(default_factory=lambda: {'price':0., 'volume':1.})
    graph_first_size:int = 2
    graph_dpi:int = 100

    binance_request_timeout:int = 10000

    _max_elements:int = 10000
    _binance_timeout:float = 0.08

    def to_dict(self) -> dict:
        """
        To dict
    
        Return:
            dict: All attributes such as: attr.name:value
        """

        return {f.name: getattr(self, f.name) for f in fields(self)}

    def copy(self) -> 'UserConfig':
        """
        Copy 

        Return:
            UserConfig: A deepcopy of self.
        """

        return deepcopy(self)

    def update(self, **kwargs) -> None:
        """
        Update

        Modify the saved instance attributes.

        Args:
            **kwargs: Attributes to modify.
        """

        valid_names = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if k in valid_names:
                setattr(self, k, v)
            else: raise AttributeError(f"Config has no attribute '{k}'.")

    @contextmanager
    def context(self, **kwargs) -> Generator:
        """
        Context

        Context manager to modify attributes.

        Args:
            **kwargs: Attributes to modify.
        """

        old = {k: getattr(self, k) for k in kwargs}
        self.update(**kwargs)
        try:
            yield self
        finally:
            self.update(**old)

@dataclass(slots=True)
class DataState():
    """
    Data state

    Engine state attributes store.

    Hidden Attributes:
        _data_tcommission: Dict with DataFrame containing fee, timestamp and price.
        _data_backtests: List of data of each backtest, 
            containing trades and data needed for statistics.
        _data_year_days: Number of operable days in 1 year.
        _data_width_day: Width of the day.
        _data_interval: Interval of the loaded data.
        _data_width: Width of the dataset.
        _data_icon: Data icon.
        _data: Loaded dataset.
        _min_gap: If left as True, gaps will not be calculated on 
            the entry of 'taker' orders.
        _limit_ig: If in a 'stopLimit' or 'takeLimit' the order is within the 
            same candle and this is False, it will be executed.
        _temp_comm: Names of temporary commissions to be used in the backtest.
        _chunk_size: Size of each chunk of the engine.
        _init_funds: Initial capital for the backtesting.
        _commission: Commission of each execution.
        _spread_pct: Market spread percentage.
        _slippage_pct: Slippage percentage.
        _orders_order: Dictionary with values to sort the order type when executing.
        _orders_nclose: If True, orders are not ordered to be executed based 
            on the closest one.
        _nper_commission: Non-percentage commission cost.
    """

    _data_tcommission:dict[str,pd.DataFrame] = field(default_factory=lambda: {})
    _data_backtests:list = field(default_factory=lambda: [])
    _data_year_days:int = 365
    _data_width_day:None|float = None
    _data_interval:None|str = None
    _data_width:None|float = None
    _data_icon:None|str = None
    _data:None|pd.DataFrame = None

    _min_gap:None|bool = None
    _limit_ig:None|bool = None
    _temp_comm:None|Sequence[str]|str = None
    _chunk_size:None|int = None
    _init_funds:None|float = 100
    _commission:None|CostsValue = None
    _spread_pct:None|CostsValue = None
    _slippage_pct:None|CostsValue = None
    _orders_order:None|dict = None
    _orders_nclose:None|bool = None
    _nper_commission:None|bool = None

def draw_plot(value_index:int, **kwargs) -> Callable:
    """
    Draw plot

    Draw a line as decoration.

    Info:
        Draws the decoration of an indicator; it is passed the wrapper arguments. 
        It must return None, a list of collections, or a matplotlib collection to draw the legend.

    Wrapper args:
        ax (Axes): Axis where draw.
        index (Sequence|ndarray): x index.
        values (Sequence): All indicator values.
        width (float): Width of each point.
        zorder (float): Position z of the element.

    Args:
        value_index (int): Index of the value to draw.
        **kwargs: Extra arguments passed to the 'plot' function.

    Returns:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence|np.ndarray, values:Sequence, zorder:float=0.4, **kn_) -> list:

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

    Returns:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, *a_, zorder:float=0.4, **kn_) -> Line2D:
        return ax.axhline(cord, color=color, **kwargs, zorder=zorder)
    return wrapper

def draw_pointsln(value_index:int, cord:float, marker:MarkerStyle|str, 
                marker_colors:dict, **kwargs) -> Callable:
    """
    Draw points line

    Draw a points line as decoration.
    For more info read 'draw_plot' docstring.

    Args:
        value_index (int): Index of the value to draw.
        cord (float): Line coordinate.
        marker (MarkerStyle): The marker style
        marker_colors (dict): Dictionary with line value and corresponding marker color.
            Example: `{1:'black', 0:'gray'}`.
        **kwargs: Extra arguments passed to the 'scatter' function.

    Returns:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence|np.ndarray, values:Sequence, 
            zorder:float=0.4, **kn_) -> PathCollection:
        nonlocal marker

        y = np.full(len(values[value_index]), cord)
        marker_arr_c = np.vectorize(lambda v: marker_colors.get(v, 'red'))(values[value_index])
        if not isinstance(marker, MarkerStyle):
            marker = MarkerStyle(marker)

        return ax.scatter(index, y, marker=marker, color=marker_arr_c, **kwargs, zorder=zorder)
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

    Returns:
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

    Returns:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence|np.ndarray, values:Sequence, 
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

    Returns:
        Callable: Wrapper.
    """

    def wrapper(ax:Axes, index:Sequence|np.ndarray, values:Sequence, 
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
    
    Returns:
        Callable: Wrapper.
    """
    
    def wrapper(ax:Axes, index:Sequence|np.ndarray, values:Sequence, 
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

@dataclass(slots=True)
class PlotState():
    """
    Plot state

    Plot state attributes store.

    Note:
        These attribute only exist if tkinter package is installed:
        _tkinter_root', '_panel_list', '_panel_wmax', '_anim_puntil', '_linked_toolbars'.
        The tkinter package comes by default in a normal Python installation, but in some
        installations, as well as on systems without a graphical interface, it may not.
        To verify, you can use the boolean variable 'TKINTER'.
    
    Hidden Attributes:
        _tkinter_root: Root Tk instance.
        _panel_list: List of windows that will be joined into panels.
        _panel_wmax: Maximum number of panels; if a value greater than 4 is given, 
            an error will occur.
        _anim_puntil: Timestamp indicating when the animation drawing should pause. 
            If None, the pause is indefinite.
        _linked_toolbars: Dict of connected toolbars.
        _random_titles: Random titles for windows.
        _custom_plot: Dict of custom graphical statistics.
        _plot_indicators: Indicators saved for plotting.
        _plt_styles: Styles for coloring trading charts.
        _plot_indicators_def: Indicators default config for plotting 
            More info in the definition. func.__name__:dict.
    """

    if TYPE_CHECKING or TKINTER:
        _tkinter_root:tk.Tk|None = None
        _panel_list:list = field(default_factory=lambda: [])
        _panel_wmax:int = 4
        _anim_puntil:float|None = None
        _linked_toolbars:dict = field(default_factory=lambda: {})

    _random_titles:list = field(default_factory=lambda: [
        f'BackPy v{_get_version()}',
        'Window from BackPy',
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
    ])

    _custom_plot:dict = field(default_factory=lambda: {})
    _plot_indicators:dict[str, dict] = field(default_factory=lambda: {})
    _plt_styles:dict = field(default_factory=lambda: {
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

        # All properties are: 'bg', 'gdir', 'fr', 'btn', 'btna', 'vol', 'mk', 'psmk', 'crss'.
        # light
        'emberday': {
            'bg': ("#f0f0f0", "#e5e5e5", "#dfdfdf"), 'gdir': True,
            'fr': '#0A0A0A', 'btn': "#FF6347", 'btna': "#DF2828",
            'vol': "#FF806A", 'mk': {'u': '#FF6347', 'd': "#CF0000"},
            'psmk':{'u':"#089991", 'd':"#f23651"},
        },
        'ivory': {
            'bg': ('#FBFAF2', '#F2EFE0', '#E7E2CB'), 'gdir': True,
            'fr': '#EDE8D2', 'btn': '#4A3A0E', 'btna': '#2E2400',
            'vol': '#BFA254', 'mk': {'u': '#2F6B3C', 'd': '#8F2020'},
        },
        'arctic': {
            'bg': ('#F4F8FC', '#E7F0F8', '#D6E5F2'), 'gdir': False,
            'fr': '#0E2A47', 'btn': '#BFDCF5', 'btna': '#2F7FD0',
            'vol': '#93B4CE', 'crss': '#0E2A47',
            'mk': {'u': '#0F7A5F', 'd': '#C03434'},
            'psmk': {'u': '#0F7A5F', 'd': '#D03A3A'},
        },
        'rosegold': {
                'bg': ('#FFF2F6', '#FFE2EE', '#FFCFE2', '#FFBCD6'), 'gdir': True,
                'fr': '#FFBAD4', 'btn': '#B52050', 'btna': '#8C1438',
                'vol': '#EE96B4', 'crss': '#8C1438', 
                'mk': {'u': '#1F8A6B', 'd': '#D6215C'},
        },
        'lilac': {
            'bg': ('#F6F0FC', '#EBDDF8', '#DCC9F0'), 'gdir': True,
            'fr': '#E4D3F3', 'btn': '#4A1E70', 'btna': '#2E0F4A',
            'vol': '#BE9FDD', 'mk': {'u': '#2A8F70', 'd': '#B02A6B'},
            'psmk': {'u': '#1E9E76', 'd': '#C81E74'},
        },
        'coral': {
            'bg': ('#FFF7F1', '#FFE8D9', '#FFD6BC'), 'gdir': True,
            'fr': '#0A0A0A', 'btn': '#FFD9C4', 'btna': '#F58B57',
            'vol': '#F0A377', 'crss': '#8C3312',
            'mk': {'u': '#127A5C', 'd': '#D93A1E'},
            'psmk': {'u': '#0E8A66', 'd': '#E8452A'},
        },
        'mint': {
            'bg': ('#F0FAF4', '#DFF3E7', '#CBEAD8'), 'gdir': True,
            'fr': '#D6EEE0', 'btn': '#0E4630', 'btna': '#052A1C',
            'vol': '#8FC7A9', 'mk': {'u': '#127A52', 'd': '#B8342A'},
            'psmk': {'u': '#0E8A5C', 'd': '#CC3A30'},
        },
        'porcelain': {
            'bg': ('#FDFDFE', '#F4F6F9'), 'gdir': True,
            'fr': '#ECEFF3', 'btn': '#20242B', 'btna': '#4C5BD4',
            'vol': '#C3CAD4', 'crss': '#4C5BD4',
            'mk': {'u': '#1F7A4D', 'd': '#B02A37'},
            'psmk': {'u': '#1F7A4D', 'd': '#C62828'},
        },

        # dark
        'midnight': {
            'bg': ('#03060E', '#0A1730', '#12254A'), 'gdir': False,
            'fr': '#01030A', 'btn': '#5AAEFF', 'btna': '#2E7FD9',
            'vol': '#1E3E66', 'mk': {'u': '#00D9B4', 'd': '#FF5566'},
        },
        'charcoal': {
            'bg': '#1B1B1B',
            'fr': '#0B0B0B', 'btn': '#EDEDED', 'btna': '#767676',
            'vol': '#3F3F3F', 'mk': {'u': '#3DD16A', 'd': '#FF4438'},
        },
        'mocha': {
            'bg': ('#191309', '#241C10', '#33281A'), 'gdir': True,
            'fr': '#100B04', 'btn': '#D9A05C', 'btna': '#A8742E',
            'vol': '#453118', 'mk': {'u': '#8FCB84', 'd': '#D06A54'},
        },
        'eclipse': {
            'bg': ('#0D0D0F', '#141417', '#1C1C20'), 'gdir': True,
            'fr': '#E8E8EA', 'btn': '#141416', 'btna': '#A83200',
            'vol': '#2E2E34', 'crss': '#F0F0F2',
            'mk': {'u': '#F0F0F2', 'd': '#FF3B30'},
            'psmk': {'u': '#00D68F', 'd': '#FF3B30'},
        },
        'orchid': {
            'bg': ('#160B22', '#211036', '#301A4E'), 'gdir': True,
            'fr': '#08040D', 'btn': '#F5A6D2', 'btna': '#B04084',
            'vol': '#503066', 'crss': '#EBC8FF',
            'mk': {'u': '#4FD1A5', 'd': '#FF4D8D'},
            'psmk': {'u': '#3ED9B0', 'd': '#FF3B6B'},
        },
        'bordeaux': {
            'bg': ('#1E0B12', '#2B121E', '#3C1A2A'), 'gdir': False,
            'fr': '#0A0207', 'btn': '#EFC97E', 'btna': '#B88E3A',
            'vol': '#552435', 'mk': {'u': '#7CCFA8', 'd': '#FF5C6C'},
            'psmk': {'u': '#E8C57A', 'd': '#FF5C6C'},
        },
        'abyss': {
            'bg': ('#04171D', '#072A34', '#0B404E'), 'gdir': True,
            'fr': '#031117', 'btn': '#55DDE8', 'btna': '#1F98A8',
            'vol': '#105668', 'crss': '#9BEFF2',
            'mk': {'u': '#2DE0B0', 'd': '#FF5A52'},
            'psmk': {'u': '#39E6C0', 'd': '#FF4F6D'},
        },
        'pine': {
            'bg': ('#0B1510', '#122019', '#1A2E22'), 'gdir': True,
            'fr': '#070F0B', 'btn': '#B6E2C2', 'btna': '#4F7C60',
            'vol': '#26402F', 'crss': '#DCE8D8',
            'mk': {'u': '#6FE08E', 'd': '#D9603F'},
            'psmk': {'u': '#6FE08E', 'd': '#E4674A'},
        },
    })

    _plot_indicators_def:dict = field(default_factory=lambda: {
        'def':{'panel':None, 'color':[None], 'dtSource':None},
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
        'idct_sma':{'panel':'price', 'color':['red'],'dtSource':'close',},
        'idct_wma':{'panel':'price', 'color':['orange'],'dtSource':'close',},
        'idct_smma':{'panel':'price', 'color':['yellow'],'dtSource':'close',},
        'idct_sema':{'panel':'price', 'color':['green'],'dtSource':'close',},

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
        'idct_stochastic':{'panel':None, 'color':['#2660FF', '#FF6F00'], 
            'style':[
                drawax_hline(70, '#FFFFFF36', linestyle=(0, (1.5,1))), 
                drawax_hline(30, '#FFFFFF36', linestyle=(0, (1.5,1))), 
                drawax_hline(50, '#FFFFFF36', linestyle=(0, (1.5,1))), 
                drawax_btw((70, 30), '#008CFF24',),
            ]},
        'idct_adx':{'panel':None, 'color':['#FF0000A1', '#0044FFA1', '#5D5D5DA1'],
            'style':[
                drawax_hline(23, '#CFCFCFD7', linestyle=(0, (1,1))), 
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
                draw_hist(1, color=('#0AAF0A', '#AF0909'), color_d=('#0A5D0A', '#5C0909')),
                draw_pointsln(0, 0, 'x', {1:'black', 0:'gray'}, s=2),
            ], 
        },
        'idct_mom':{'panel':None, 'color':['#2D2DFF'],
            'dtSource':'close',
            },
        'idct_ichimoku':{'panel':'price', 'color':['#C4E293', '#E29393', '#2450C0', '#C0110B'], 
            'style':[
                draw_btw((0,1), ('#81B13324', '#AA2F2F24'))
            ]},
        'idct_atr':{'panel':None, 'color':['#B71C1C']},
    })

user_config:UserConfig = UserConfig()
data_state:DataState = DataState()
plot_state:PlotState = PlotState()

def _store_decorator(func:Callable) -> Callable:
    """
    Store decorator

    Decorate a function with this to give it 
        the attribute: '_store' and have it decorated with '__data_store'.

    Note:
        The decorated function must not have parameters named 'cut' or 'last'. 
        'StrategyClass.__data_store' intercepts those two names for its own 
        caching/slicing logic, so the values you pass never reach your function.

    Args:
        func (Callable): Function.

    Returns:
        Callable: Function.
    """

    setattr(func, '_store', True)
    return func

def get_commission_names() -> list[str]:
    """
    Get temporary commissions names

    Return:
        list[str]: Names list.
    """

    return list(data_state._data_tcommission.keys())

def _get_temp_commission(names:Sequence[str]|str|None = None) -> dict[str, pd.DataFrame]:
    """
    Get temporary commission

    Args:
        names (Sequence[str]|str|None, optional): Names to extract, '*' or None = all.

    Return:
        dict[str, pd.DataFrame]: Dictionary with temporary commissions.
    """

    if isinstance(names, str):
        names = [names]
    if names is None or '*' in names:
        return data_state._data_tcommission.copy()

    return {name:data_state._data_tcommission[name] for name in names}

def __add_temp_commission(name:str, df:pd.DataFrame) -> None:
    """
    Add temporary commission

    Save a DataFrame with commission markers.

    Note:
        DataFrame columns:
        - fee: Commission as a percentage, 1 is 100%.
        - price: Commission execution price.
        - mode: Commission method current: 'funding_fee'.
        If the 'name' already exists, it is overwritten and a log is sent.

    Args:
        name (str): Name to save with.
            It becomes lowercase, and spaces before or after are removed.
        df (pd.DataFrame): Dataframe with index for date 
            and columns 'price' and 'fee'.
    """
    name = name.strip().lower()

    if not all([col in df.columns for col in ('price', 'fee')]):
        raise exception.DataError(
            "Missing column. Required columns: 'fee', 'price'.")
    elif df.empty:
        raise exception.DataError('Empty dataframe.')

    if name in data_state._data_tcommission:
        logger.warning(f"'{name}' existing temporary commission name is overwritten.")

    df = df.reindex(columns=['fee', 'price', 'mode'])
    data_state._data_tcommission.update({name:df})

def del_backtest(names:Sequence[str|int|None]|str|int|None = None) -> None:
    """
    Delete backtests

    Remove a backtest; each backtest can consume a lot of memory, 
    so if you don't need it, it's best to remove it.

    Args:
        names (Sequence[str|int|None]|str|int|None, optional): 
            Name or index of the backtests to be deleted.
    """

    if len(data_state._data_backtests) == 0:
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

    return __get_names(data_state._data_backtests)

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
        del data_state._data_backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return data_state._data_backtests[-1]

    for i, backtest in enumerate(data_state._data_backtests):
        if backtest['name'] == name:
            del data_state._data_backtests[i]

def __get_names(from_:list[dict]) -> list[str]:
    """
    Get names

    Takes the names of the 'from' list of dictionaries.
    'from' needs 'name' key.

    Args:
        from_ (list[dict]): List of dictionaries 
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
        (g_st:=__get_strategy(i))['name']: (g_st['trades'].sort_values(
            by='positionDate', ascending=True).reset_index(drop=True) 
            if 'positionDate' in g_st['trades'] else g_st['trades']) 
            if 'trades' in g_st else pd.DataFrame()
        for i in (set(names or {None}) if not isinstance(names, (str, int))  else [names])
    }

    return trades

def __get_trades(names:Sequence[str|int|None]|str|int|None = None) -> pd.DataFrame:
    """
    Get trades

    Take trades from 1 or more saved backtests.
    Trades will be sorted ascending based on 'positionDate'.

    Note:
        Dataframe columns:
        - date: Creation date.
        - positionOpen: Opening price.
        - commission: Position commissions.
        - amount: Position amount
        - typeSide: Position type.
        - unionId: Id linked to orders.
        - positionClose: Closing price.
        - positionDate: Closing date.
        - profitPer: Profit in percentage with out commissions.
        - profit: Profit on 'amount' with commissions.

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

    if len(data_state._data_backtests) == 0:
        return {'name':None, 
                'trades':pd.DataFrame(), 
                'balance_rec':pd.Series(),
                'init_funds':0, 
                'd_year_days':0, 
                'd_width_day':0, 
                'd_width':0}
    elif isinstance(name, int) or name is None:
        return data_state._data_backtests[-1 if name is None else name]
    elif not isinstance(name, str):
        return data_state._data_backtests[-1]

    for i,v in enumerate(data_state._data_backtests):
        if v['name'] == name:
            return data_state._data_backtests[i]

    raise exception.DataError('Name not found.')

def __gen_fname(name:str, from_:list[str]) -> str:
    """
    Generate frame name

    Generates a name based on 'name' that is not duplicated in 'from'.

    Args:
        name (str): Name.
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
