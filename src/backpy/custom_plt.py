"""
Custom plot module

Contains the aesthetic configuration for matplotlib.

Variables:
    logger (Logger): Logger variable.

Functions:
    def_style: Define a new style to plot your graphics.
    style_def: Takes a style from '__plt_style'.
    gradient_ax: Create a diagonal background gradient on the 'ax' with 'ax.imshow'.
    custom_ax: Aesthetically configures an axis.
    ax_view: Based on a str generates axes.
    add_window: Add a tkinter window with the integration in tkinter.
    gen_axes: Generate 'num' number of axes.
    config_ax: Backpy default axis configuration.
"""

from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
from matplotlib.axes._axes import Axes
from typing import Callable, Any, cast
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rd
import numpy as np
import logging

from . import _commons as _cm
from . import exception
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

def def_style(name:str, 
              background:str | tuple[str, ...] | list[str] = '#e5e5e5', 
              frames:str = 'SystemButtonFace', 
              buttons:str = '#000000', 
              button_act:str | None = None, 
              gardient_dir:bool = True, 
              volume:str | None = None, 
              cross:str | None = None,
              up:str | None = None, 
              down:str | None = None,
              pos_up:str | None = None,
              pos_down:str | None = None
              ) -> None:
    """
    Def style

    Define a new style to plot your graphics.
    Only valid colors for tkinter.

    Dict format:
        name:
            'bg': background, 
            'gdir': gardient_dir,
            'fr': frames, 
            'btn': buttons, 
            'btna': buttons_act,
            'vol': volume, 
            'crss': cross,
            'mk': {
            'u': up, 
            'd': down},
            'psmk': {
            'u': up,
            'd': down},

    Args:
        name (str): Name of the new style by which you will call it later.
        background (str|tuple[str,...]|list[str], optional): 
            Background color of the axes. 
            It can be a gradient of at least 2 colors using a tuple or list.
        frames (str, optional): Background color of the frames.
        buttons (str, optional): Button color.
        button_act (str|None, optional): Color of buttons when selected or sunken.
        gardient_dir (bool, optional): The gradient direction will always 
            be top to bottom and diagonal, but you can choose whether 
            it starts from the right or left, true = right.
        volume (str|None, optional): Volume color.
        cross (str|None, optional): Cross pointer color, default buttons color.
        up (str|None, optional): Color when the price rises, this influences 
            the color of the candle.
        down (str|None, optional): Color of when the price rises this influences 
            the color of the candle.
        pos_up (str|None, optional): Color when a position is positive. 
            If it is None, the color of 'mk' will be used.
        pos_down (str|None, optional): Color when a position is negative. 
            If it is None, the color of 'mk' will be used.
    """
    if name in _cm.__plt_styles.keys():
        raise exception.StyleError(f"Name already in use. '{name}'")

    up_mc = up or 'green'
    down_mc = down or 'red'
    btn_color = buttons or '#000000'

    _cm.__plt_styles.update({
        name: {
            'bg':background or '#e5e5e5', 
            'gdir':gardient_dir, 
            'fr':frames or 'SystemButtonFace',
            'btn':btn_color, 
            'btna':button_act or '#333333', 
            'vol': volume or 'tab:orange',
            'crss': cross or btn_color,
            'mk': {
                'u':up_mc,
                'd':down_mc,
            },
            'psmk': {
                'u':pos_up or up_mc,
                'd':pos_down or down_mc,
            },
    }})

def style_def(name:str|None = 'last', update:dict|None = None) -> tuple[dict, str]:
    """
    Style def

    Takes a style from '__plt_style'.

    All color styles:
        Documentation of this in the 'plot' docstring.

    Args:
        name (str|None, optional): Name of the color style. 
            If you leave it as 'last' the last one will be returned.
        update (dict|None, optional): Customize the defined style by 
            modifying the dictionary. To know what to modify, 
            read the docstring of 'def_style'.

    Returns:
        tuple[dict,str]: Style dict and style name.
    """

    if (not name is None and not (name:=name.lower()) in {'random', 'last'} | set(_cm.__plt_styles.keys())):
        raise exception.StyleError(f"Style not found. '{name}'")

    if name == 'last':
        name = _cm.plt_style

    if name is None:
        name = list(_cm.__plt_styles.keys())[0]
    elif name == 'random':
        name = rd.choice(list(_cm.__plt_styles.keys()))

    stl_colors = _cm.__plt_styles[name]
    _cm.plt_style = name

    mk = stl_colors.get('mk', {'u': 'green', 'd': 'red'})
    if 'psmk' not in stl_colors:
        stl_colors['psmk'] = mk.copy()
    if 'crss' not in stl_colors:
        stl_colors['crss'] = stl_colors['btn']

    if isinstance(update, dict):
        stl_colors.update(update)

    return stl_colors, name

def gradient_ax(ax:Axes, colors:list|tuple, right:bool=False) -> None:
    """
    Gradient axes.

    Create a diagonal background gradient on the 'ax' with 'ax.imshow'.

    Args:
        ax (Axes): Axes to draw.
        colors (list|tuple): List of the colors of the garden in order.
            Len less than 2 will default to: ['white', '#e5e5e5'].
        right (bool, optional): Corner from which the gradient 
            starts if False starts from the top left.
    """

    if len(colors) < 2:
        colors = ['white', '#e5e5e5']

    gradient = (np.linspace(0, 1, 256).reshape(-1, 1) 
                + (np.linspace(0, 1, 256) 
                   if right else -np.linspace(0, 1, 256)))

    ylim = ax.get_ylim()
    autoylim, autoxlim = ax.get_autoscaley_on(), ax.get_autoscalex_on() # pyrefly: ignore

    im = ax.imshow(gradient, aspect='auto', 
                   cmap=mpl.colors.LinearSegmentedColormap.from_list('custom_gradient', colors), 
                extent=(0., 1., 0., 1.), transform=ax.transAxes, zorder=-1)
    im.get_cursor_data = lambda event: None
    im.sticky_edges.x[:] = []; im.sticky_edges.y[:] = [] # pyrefly: ignore

    ax.set_ylim(*ylim)
    ax.set_autoscaley_on(autoylim); ax.set_autoscalex_on(autoxlim) # pyrefly: ignore

def custom_ax(ax:Axes, bg:str|tuple|list = '#e5e5e5', edge:bool = False) -> None:
    """
    Custom axes.

    Aesthetically configures an axis.

    Note:
        The gradient can change the 'ax' limits.

    Args:
        ax (Axes): Axes to config.
        bg (str|tuple|list, optional): Background color of the axis, 
            if it is a list or tuple a gradient will be created.
        edge (bool, optional): If the background is a gradient, this 
            determines which corner you launch from, false left, true right.
    """

    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5) 

    if (isinstance(bg, tuple) or isinstance(bg, list)) and len(bg) > 1:
        gradient_ax(ax, bg, right=edge)
    else:
        ax.set_facecolor(bg[0] if isinstance(bg, list) else bg)

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.title.set_color('white') # type: ignore
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    semi_transparent_white = mpl.colors.to_rgba(cast(Any, "white"), alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(semi_transparent_white)
        spine.set_linewidth(1.2)

def ax_view(view:str, graphics:list[str], fig:plt.Figure|None = None,
            sharex:bool = False) -> tuple[list[Axes], list[str]]:
    """
    Axes view

    Based on a str generates axes.
        Generates up to 8 axes and places them covering the entire canvas.

    Args:
        view (str): String with format: s/s/s/s each value is a graph.
        graphics (list[str]): Name of each graph, needed to process 'view'.
            If a graphic has a '/' it is replaced by ''.
        fig (Figure|None, optional): Matplotlib figure.
        sharex (bool): Shares the x-axis with all the axes.

    Returns:
        tuple[list[Axes],list[str]]: List of axes and list of view values.
    """

    graphics = [g.replace('/', '') for g in graphics]
    pview = view.lower().strip().split('/')
    pview = [i for i in pview if i in graphics]

    if len(pview) > 8 or len(pview) < 1: 
        raise exception.StatsError(utils.text_fix(f"""
            'view' allowed format: 's/s/s/s' where s is the name of the graph.
            Available graphics: {(",".join([f"'{i}'" for i in graphics]))}.
            """, newline_exclude=True))

    loc = [(0,0), (1,0), (1,4), (0,4), (1,2), (1,6), (0,2), (0,6)]
    layout_rules = {
        1: lambda i: (2, 8, 2, 8),
        2: lambda i: (2, 8, 1, 8),
        3: lambda i: (2, 8, 1, 8 if i==0 else 4),
        4: lambda i: (2, 8, 1, 4),
        5: lambda i: (2, 8, 1, 2 if i in [1,4] else 4),
        6: lambda i: (2, 8, 1, 4 if i in [0,3] else 2),
        7: lambda i: (2, 8, 1, 4 if i==3 else 2),
        8: lambda i: (2, 8, 1, 2),
    }

    fig_kw = {'fig': fig} if fig is not None else {}
    axes = []
    for i in range(len(pview)):
        sharex_=axes[-1] if axes and sharex else None
        nrows, ncols, rowspan, colspan = layout_rules[len(pview)](i)

        axes.append(
            plt.subplot2grid(
                (nrows, ncols), loc[i],
                rowspan=rowspan, colspan=colspan,
                sharex=sharex_, **fig_kw
        ))

    return axes, pview

def add_window(fig:Figure, title:str|Callable|None = None, block:bool = True, 
            anim:FuncAnimation|None = None, **kwargs) -> None:
    """
    Add window

    Add a tkinter window with the integration in tkinter.
    If tkinter is not installed, launch the window with matplotlib.

    Note:
        If you do not use tkinter integration,
        the '_anim_ref' attribute will be added to the 'fig'.

    Args:
        fig (Figure): Matplotlib Figure.
        title (str|Callable|None, optional): Window/panel title.
        block (bool, optional): Lock the thread and create 
            the window with the panels.
        anim (FuncAnimation|None, optional): Matplotlib FuncAnimation.
        **kwargs: -> tk_window.add_window
    """

    if not _cm.graph_integration or not _cm.TKINTER:
        fig_canvas = getattr(fig, 'canvas')
        if fig_canvas.manager is not None:
            fig_canvas.manager.set_window_title(
                ((lambda: title) if isinstance(title, str|None) else title)() 
                or rd.choice(_cm._random_titles if _cm._random_titles else ['BackPy']))

        setattr(fig, '_anim_ref', anim)
        return plt.show(block=block)

    from . import tk_window as tkw

    return tkw.add_window(fig=fig, title=title, block=block, anim=anim, **kwargs)

def gen_axes(num:int, pad:int=0, sharex:Axes|None=None, 
            sharey:Axes|None=None, autoget_sharex:bool = False, 
            autoget_sharey:bool = False, fig:plt.Figure|None=None) -> list:
    """
    Generate axes

    Generate 'num' of axes. 

    Args:
        num (int): Number of axes to generate.
        pad (int, optional): Number of rows to pad at the top.
        sharex (Axes|None, optional): Axes to share the x-axis.
        sharey (Axes|None, optional): Axes to share the y-axis.
        autoget_sharex (bool, optional): All new axes share the x-axis.
        autoget_sharey (bool, optional): All new axes share the y-axis.
        fig (Figure|None, optional): Matplotlib figure.

    Returns:
        list: A list of generated axes.
    """

    fig_kw = {'fig': fig} if fig is not None else {}
    axes = []
    for i in range(num):
        ax = plt.subplot2grid((num+pad,1), (i+pad,0), rowspan=1, 
            colspan=1, sharex=sharex, sharey=sharey, **fig_kw)
        axes.append(ax)

        if autoget_sharex:
            sharex = ax
        if autoget_sharey:
            sharey = ax

    return axes

def config_ax(ax:Axes, date:bool = True, bg_color:str|tuple|list = '#e5e5e5', 
            gdir:bool = False, log:bool = False) -> None:
    """
    Config axis

    Backpy default axis configuration.

    Args:
        ax (Axes): Axis to configure.
        date (bool, optional): If true, the x-axis will display dates.
        bg_color (str|tuple|list, optional): Background color of the axis,
            if it is a list or tuple a gradient will be created.
        gdir (bool, optional): If the background is a gradient, this
            determines which corner you launch from, false left, true right.
        log (bool, optional): Logarithmic y-axis.
    """

    custom_ax(ax, bg_color, edge=gdir)
    if log: ax.semilogy()

    ax.yaxis.set_major_formatter(lambda y, _: utils.float_str(utils.round_r(y.real, 2, True)))
    ax.xaxis.set_major_formatter(
        DateFormatter('%H:%M %d-%m-%Y') if date else lambda x, _: utils.float_str(utils.round_r(x.real, 2, True)))

    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
