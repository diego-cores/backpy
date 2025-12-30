"""
Custom plot module

Contains matplotlib embedding logic and colors.

Variables:
    logger (Logger): Logger variable.

Classes:
    CustomWin: Create a custom window with BackPy using 'tkinter' and 'matplotlib'.
    CustomToolbar: Inherits from the 'NavigationToolbar2Tk' class to 
        modify the toolbar buttons and change colors.

Functions:
    def_style: Define a new style to plot your graphics.
    gradient_ax: Create a diagonal background gradient on the 'ax' with 'ax.imshow'.
    custom_ax: Aesthetically configures an axis.
    ax_view: Based on a str generates axes.
    new_paneledw: Generate a window with panels using 'CustomWin'.
    add_window: Add a tkinter window with 'CustomWin'.
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore[import]
from matplotlib.animation import FuncAnimation
from typing import Callable, Never, Any, cast
from PIL import Image, ImageTk, ImageOps
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from importlib import resources
from types import MethodType
import matplotlib.colors
import matplotlib as mpl
import tkinter as tk
import random as rd
import numpy as np
import logging
import os

from . import _commons as _cm
from . import exception
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

class CustomToolbar(NavigationToolbar2Tk):
    """
    Custom Toolbar.

    Inherits from the 'NavigationToolbar2Tk' class to 
        modify the toolbar buttons and change colors.

    Attributes:
        toolitems: Buttons list.
        icon_map: Dictionary of the file name of each button logo.
        window: Window root.
        color_act: Color of the sunken buttons.
        color_btn: Color of buttons and icons.
        color_bg: Frame color.
        icon_dir: Directions to matplotlib icons.
        custom_img: Custom icons saved.

    Private Attributes:
        _org_zoom: Original 'zoom' function used to connect with other toolbars.
        _org_pan: Original 'pan' function used to connect with other toolbars.

    Methods:
        config_colors: Configure the colors of the buttons and frame.
        select: Changes the style of the button when selected.
        deselect: Changes the style of the button when deselect.
        run_zoom: Function responsible for executing 'zoom' on all toolbars.
        run_pan: Function responsible for executing 'pan' on all toolbars.
    """

    _buttons: dict[Any, Any]
    winfo_children:Callable

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

    icon_map = {
        'Home': 'home.png',
        'Back': 'back.png',
        'Forward': 'forward.png',
        'Zoom': 'zoom_to_rect.png',
        'Pan': 'move.png',
        'Save': 'filesave.png'
    }

    def __init__(self, canvas:FigureCanvasTkAgg, window:tk.Tk, 
                 color_btn:str = '#000000', color_bg:str = 'SystemButtonFace', 
                 color_act:str = '#333333', movement:bool = True, 
                 link:bool = False) -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            canvas (FigureCanvasTkAgg): Canvas containing the matplotlib figure.
            window (Tk): Window root.
            color_btn (str, optional): Button color.
            color_bg (str, optional): Frame color.
            color_act (str, optional): Color of the sunken buttons.
            movement (bool, optional): Enable user movement buttons.
            link (bool, optional): If it is True, the toolbar connects 
                to all other toolbars with a link; only the pan and 
                zoom button is connected.
        """

        if not movement:
            self.toolitems = ( # type: ignore[assignment]
                ('Home', 'Reset original view', 'home', 'home'),
                (None, None, None, None),
                ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
            self.icon_map = {
                'Home': 'home.png',
                'Save': 'filesave.png'
            }

        super().__init__(canvas, window)

        self.window = window
        self.color_act = color_act
        self.color_btn = color_btn
        self.color_bg = color_bg

        self._org_zoom = super().zoom
        self._org_pan = super().pan

        self.icon_dir = os.path.join(mpl.get_data_path(), "images")

        self.custom_img = {}
        self.config_colors()

        if link and movement:
            linked_toolbars:list = getattr(_cm, '__linked_toolbars')
            linked_toolbars.append(self)

            setattr(_cm, '__linked_toolbars', linked_toolbars)
        else:
            self.run_zoom = self._org_zoom
            self.run_pan = self._org_pan

    def run_zoom(self, *args) -> None:
        """
        Run zoom

        Execute '_org_zoom' in '__linked_toolbars'.
        """

        linked_toolbars = []
        for v in getattr(_cm, '__linked_toolbars'):
            try:
                v._org_zoom(*args)
                linked_toolbars.append(v)
            except tk.TclError:
                pass

        setattr(_cm, '__linked_toolbars', linked_toolbars)

    def run_pan(self, *args) -> None:
        """
        Run pan

        Execute '_org_pan' in '__linked_toolbars'.
        """

        linked_toolbars = []
        for v in getattr(_cm, '__linked_toolbars'):
            try:
                v._org_pan(*args)
                linked_toolbars.append(v)
            except tk.TclError:
                pass

        setattr(_cm, '__linked_toolbars', linked_toolbars)

    class _ZoomExit(Exception):
        """Internal control-flow exception."""
        pass

    def zoom(self, *args) -> Never:
        self.run_zoom(*args)
        raise SystemExit

    def pan(self, *args) -> Never:
        self.run_pan(*args)
        raise SystemExit

    def config_colors(self) -> None:
        """
        Config colors

        Configure the colors of the buttons and frame.
        """

        for key, filename in self.icon_map.items():

            path = os.path.join(self.icon_dir, filename)
            img = Image.open(path).convert("RGBA")

            gray = ImageOps.grayscale(img)
            colorized = ImageOps.colorize(gray, black=self.color_btn, white="#000000")

            colorized.putalpha(img.split()[-1])
            img_tk = ImageTk.PhotoImage(colorized, master=self.window)
            self.custom_img[key] = img_tk

            btn = self._buttons[key]

            if isinstance(btn, tk.Button):
                btn.config(activebackground=self.color_act)
            elif isinstance(btn, tk.Checkbutton):
                btn.select = MethodType(lambda b, img_tk=img_tk: self.select(b, img_tk), btn)
                btn.deselect = MethodType(lambda b: self.deselect(b), btn)
                btn.config(activebackground=self.color_act, selectcolor=self.color_act)
            btn.config(image=img_tk, bg=self.color_bg)

        list(map(lambda x: x.config(bg=self.color_bg, fg=self.color_btn), self.winfo_children()[-2:]))

    def select(self, btn:tk.Checkbutton, img:ImageTk.PhotoImage) -> None:
        """
        Select

        Changes the style of the button when selected.

        Args:
            btn (Checkbutton): Button.
            img (PhotoImage): Button icon.
        """

        btn.setvar(btn.cget('variable'), '0')
        btn.config(image=img, bg=self.color_act, offrelief="sunken", overrelief="groove")

    def deselect(self, btn:tk.Checkbutton) -> None:
        """
        Deselect

        Changes the style of the button when deselect.

        Args:
            btn (Checkbutton): Button.
        """

        btn.setvar(btn.cget('variable'), '0')
        btn.config(bg=self.color_bg, offrelief="flat", overrelief="flat")

    def set_history_buttons(self) -> None:
        """
        Set history buttons.

        This disables this function so the 'next' and 'back' buttons are not disabled.
        """

        return

class CustomWin:
    """
    Custom window.

    Create a custom window with BackPy using 'tkinter' and 'matplotlib'.

    Attributes:
        root: Tkinter window.
        icon: Window icon.
        title: Window title, can be called.
        color_frame: Frames color.
        color_buttons: Buttons color.
        color_button_act: Color of the sunken buttons.

    Private Attributes:
        _regen_title: If 'title' is callable, it is called again every 5s.
        _after_id_lift: After id that lifts the window with 'lift'.
        _main_after_id: 'root.after' id used to avoid errors by not blocking the process.
        _after_id_title: After id of title generation.

    Methods:
        gen_title: Generate and put the title to the window.
        config_icon: Put the icon on the application and change its color.
        lift: Focus on the window and jump over the others.
        mpl_canvas: Put your matplotlib figure inside the window.
        mpl_animation: Configure the window and run an animation.
        mpl_update: Updates the position of the toolbar and the canvas.
        mpl_toolbar: Put the matplotlib toolbar in the window.
        mpl_toolbar_config: Configure the toolbar, but don't 
            do this if you're going to use 'tk_panels'.
        tk_panels: Generates 'PanedWindow' panels.
        mpl_panels_config: Configure the panels.
        show: Show the window.

    Private Methods:
        _quit: Closes the window without errors.
    """

    def __init__(self, title:str|Callable|None = 'BackPy interface', 
                 frame_color:str = 'SystemButtonFace', 
                 buttons_color:str = '#000000', 
                 button_act:str = '#333333', 
                 geometry:str = '1200x600',
                 regen_title:bool = True) -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            title (str|Callable|None, optional): Window title.
            frame_color (str, optional): Color of the toolbar and other frames.
            buttons_color (str, optional): Button color.
            button_act (str, optional): Color of the sunken buttons.
            geometry (str, optional): Window geometry.
            regen_title(bool, optional): If 'title' is callable, 
                it is called again every 5s.
        """

        self.root = tk.Tk()
        self.root.geometry(geometry)
        self.root.config(bg=frame_color)

        self.icon = None
        self.title = title if title else lambda: rd.choice(
            _cm._random_titles if _cm._random_titles else ['BackPy'])

        self._main_after_id = None
        self._regen_title = regen_title

        self.color_frame = frame_color
        self.color_buttons = buttons_color
        self.color_button_act = button_act

        self.config_icon()
        self.gen_title(title=self.title)
    
        self._after_id_title = None
        self._after_id_lift = self.root.after(100, self.lift)

        self.root.minsize(int(self.root.winfo_screenwidth()*0.4), 
                          int(self.root.winfo_screenheight()*0.4))

        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    def gen_title(self, title:str|Callable|None = None):
        """
        Gen title

        Add a title to the window.
        It is generated if title is callable.

        Args:
            title (str|Callable|None, optional): Title.
        """

        if not self.root.winfo_exists():
            return

        try:
            if self._after_id_title:
                self.root.after_cancel(self._after_id_title)
                self._after_id_title = None
        except AttributeError:
            pass

        if callable(title):
            self._after_id_title = self.root.after(
                10000, lambda: self.gen_title(self.title))
            title = title()

        self.root.title(title)

    def config_icon(self) -> None:
        """
        Configure icon.

        Put the icon on the application and change its color.
        """

        with resources.as_file(resources.files('backpy.assets') / 'icon128x.png') as icon_path:
            img = Image.open(icon_path).convert("RGBA")

        gray = ImageOps.grayscale(img)
        colorized = ImageOps.colorize(gray, black=self.color_buttons, 
                                      white="#000000")
        colorized.putalpha(img.split()[-1])

        self.icon = ImageTk.PhotoImage(colorized, master=self.root)

        self.root.tk.call('wm', 'iconphoto', '.', self.icon)

    def lift(self) -> None:
        """
        Lift.

        Focus on the window and jump over the others.
        """

        self._after_id_lift = ''

        if not _cm.lift:
            return

        self.root.iconify()
        self.root.update()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _quit(self) -> None:
        """
        Quit.

        Closes the window without errors.
        """

        if self._main_after_id:
            self.root.after_cancel(self._main_after_id)
            self._main_after_id = None
        if self._after_id_lift:
            self.root.after_cancel(self._after_id_lift)
            self._after_id_lift = ''
        if self._after_id_title:
            self.root.after_cancel(self._after_id_title)
            self._after_id_title = None

        if self.root.winfo_exists():
            self.root.destroy()

    def mpl_canvas(self, fig:Figure, 
                   master:tk.PanedWindow|tk.Frame|tk.Tk|None = None) -> FigureCanvasTkAgg:
        """
        Matplotlib canvas

        Put your matplotlib figure inside the window.

        Args:
            fig (Figure): Figure from matplotlib.
            master (PanedWindow|Frame|Tk|None, optional): Where the canvas is drawn.

        Returns:
            FigureCanvasTkAgg: Resulting canvas figure.
        """

        master = master or self.root

        canvas = FigureCanvasTkAgg(fig, master=master)
        widget = canvas.get_tk_widget()
        widget.config(bg=self.color_button_act)
        widget.place(x=0, y=0, relwidth=1, relheight=1)

        if isinstance(master, tk.PanedWindow):
            master.add(canvas.get_tk_widget(), minsize=100)

        last_size = ()
        _in_rdraw = False
        re_draw = canvas.draw
        _resize_after_id:str = ''

        def on_redraw(call_rdraw:bool = True) -> None:
            """
            On redraw

            Redraws the canvas and resets the method to the original 'canvas.draw'.

            Args:
                call_rdraw (bool, optioanl): It's called 'canvas.draw' when redefining the function.
            """

            if not self.root.winfo_exists():
                return

            try:
                if call_rdraw:
                    re_draw()
                canvas.draw = re_draw
                setattr(_cm, '__anim_run', True)
            except AttributeError:
                self.root.after(1000, on_redraw)

        def on_resize(event:tk.Event|None = None) -> None:
            """
            On resize
    
            Does a '.after' to the redraw and modifies the 'canvas.draw' method.

            If the event size does not change, nothing is done.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _resize_after_id, _in_rdraw, last_size

            setattr(_cm, '__anim_run', False)
            canvas.draw = lambda: None

            if _resize_after_id:
                widget.after_cancel(_resize_after_id)

            def before_redraw() -> None:
                """
                Before redraw

                Before calling 'on_redraw', the variable '_in_rdraw' is set to False.
                """
                nonlocal _in_rdraw

                if not self.root.winfo_exists():
                    return

                _in_rdraw = False
                on_redraw()

            func = before_redraw
            size = ()

            if (event and (size:=(event.width, event.height)) == last_size 
                and not _in_rdraw):

                func = lambda: on_redraw(call_rdraw=False)
            last_size = size

            _resize_after_id = _in_rdraw = widget.after(400, func)

        last_pos = (0,0)
        last_state = 'normal'
        _state_after_id = None
        _anim_run_after_id = None

        def on_state(event:tk.Event|None = None) -> None:
            """
            On state
    
            If the window changes state it does a '.after' to redraw 
                and modifies the 'canvas.draw' method.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _anim_run_after_id, _state_after_id, _resize_after_id, last_state, last_pos

            if _state_after_id is not None:
                widget.after_cancel(_state_after_id)

            def on_state_resize():
                nonlocal last_state

                last_state = 'normal'
                on_redraw()

            if self.root.state() == 'zoomed':
                last_state = 'zoomed'
            elif self.root.state() == 'normal' and last_state == 'zoomed':
                canvas.draw = lambda: None
                setattr(_cm, '__anim_run', False)

                widget.after_cancel(_resize_after_id)
                _state_after_id = widget.after(400, on_state_resize)

            pos = (self.root.winfo_x(), self.root.winfo_y())
            if last_pos != pos:
                last_pos = pos

                if _anim_run_after_id is not None:
                    widget.after_cancel(_anim_run_after_id)

                setattr(_cm, '__anim_run', False)
                _anim_run_after_id = self.root.after(400, lambda: setattr(_cm, '__anim_run', True))

        def on_click(event:tk.Event|None = None) -> None:
            """
            On click
    
            Cancel '_anim_run_after_id' and change the state of '__anim_run' to False.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _anim_run_after_id

            if _anim_run_after_id is not None:
                widget.after_cancel(_anim_run_after_id)

            setattr(_cm, '__anim_run', False)

        def on_unclick(event:tk.Event|None = None) -> None:
            """
            On unclick
    
            When you stop hold the canvas, '__anim_run' is changed to True.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _anim_run_after_id

            _anim_run_after_id = self.root.after(400, lambda: setattr(_cm, '__anim_run', True))

        widget.bind('<Button-1>', on_click, add='+')
        widget.bind('<Button-3>', on_click, add='+')
        widget.bind('<ButtonRelease-1>', on_unclick, add='+')
        widget.bind('<ButtonRelease-3>', on_unclick, add='+')
    
        widget.bind('<Configure>', on_resize, add='+')
        self.root.bind('<Configure>', on_state, add='+')

        return canvas

    def mpl_animation(self, anim:FuncAnimation, mpl_canvas:FigureCanvasTkAgg, 
                      interval:int = 100) -> None:
        """
        Matplotlib animation

        Configure the window and run the animation.

        Args:
            anim (FuncAnimation): Matplotlib animation.
            mpl_canvas: Canvas figure.
            interval (int, optional): interval between frames, minimum 100 to prevent 
                the window from blocking when moving it, if that happens increase the interval.
        """

        anim = cast(Any, anim)

        if anim.event_source:
            anim.event_source.stop()

        def on_anim() -> None:
            """
            On animation

            Update the animation without blocking the thread.
            """
            nonlocal anim_after_id

            if not self.root.winfo_exists():
                return

            if (self.root.winfo_ismapped()
                and getattr(mpl_canvas.draw, "__func__", None) 
                    is type(mpl_canvas).draw
                and getattr(_cm, '__anim_run')):
                try:
                    anim.frame_seq = iter(anim.frame_seq)
                    frame = next(anim.frame_seq)

                    getattr(anim, '_draw_next_frame')(frame, getattr(anim, '_blit'))
                except:
                    pass

            anim_after_id = self.root.after(
                interval if interval >= 100 else 100, on_anim)

        anim_after_id = self.root.after(1000, on_anim)
        last_pos = (0,0)

        def on_resize(event:tk.Event) -> None:
            """
            On resize
    
            If the window moves, the after that 
                generates the next frame will be canceled.

            Args:
                event (Event): Tkinter event.
            """
            nonlocal anim_after_id, last_pos

            pos = (self.root.winfo_x(), self.root.winfo_y())
            if last_pos != pos:
                last_pos = pos

                self.root.after_cancel(anim_after_id)
                anim_after_id = self.root.after(400, on_anim)
        self.root.bind('<Configure>', on_resize, add='+')

    def mpl_update(self, canvas:tk.Canvas, toolbar:CustomToolbar, 
                   height:int = 32, mpl_place:bool = True) -> None:
        """
        Matplotlib update

        Updates the position of the toolbar and the canvas.

        Args:
            canvas (Canvas): Tkinter canvas.
            toolbar (CustomToolbar): Toolbar object.
            height (int, optional): Toolbar height in pixels.
            mpl_place (bool, optional): If you want 'mpl_canvas' not to change shape, 
                leave it set to False.
        """

        final_height = height/self.root.winfo_height()
        toolbar.place(relx=0, rely=1-final_height, relwidth=1, relheight=final_height)

        if mpl_place:
            canvas.place(relx=0, rely=0, relwidth=1, relheight=1-final_height)

    def mpl_toolbar(self, mpl_canvas:FigureCanvasTkAgg,
                    movement:bool = True, link:bool = False) -> CustomToolbar:
        """
        Matplotlib toolbar

        Put the matplotlib toolbar in the window.

        If you are going to use a single toolbar, 
            run the 'mpl_toolbar_config' function before 'show'.
        If you use 'tk_panels', do not run 'mpl_toolbar_config'.

        Args:
            mpl_canvas (FigureCanvasTkAgg): Canvas figure.
            movement (bool, optional): Activate the movement buttons.
            link (bool, optional): If it is True, the toolbar connects 
                to all other toolbars with a link; only the pan and 
                zoom button is connected.

        Returns:
            CustomToolbar: Toolbar.
        """

        toolbar = CustomToolbar(mpl_canvas, self.root, color_btn=self.color_buttons, 
                                color_bg=self.color_frame, color_act=self.color_button_act,
                                movement=movement, link=link)
        toolbar.config(bg=self.color_frame)

        return toolbar

    def mpl_toolbar_config(self, toolbar:CustomToolbar, mpl_canvas:FigureCanvasTkAgg, 
                           height:int = 32, mpl_place:bool = True) -> None:
        """
        Matplotlib toolbar config

        Configure the height and update of the toolbar.

        If you use 'tk_panels', do not run this.

        Args:
            mpl_canvas (FigureCanvasTkAgg): Canvas figure.
            height (int, optional): Toolbar height in pixels.
            mpl_place (bool, optional): If you want 'mpl_canvas' not to change shape, 
                leave it set to False.
        """

        self.toolbar_config = True
        self.mpl_update(mpl_canvas.get_tk_widget(), toolbar, 
                        height=height, mpl_place=mpl_place)

        self.root.bind('<Configure>', 
                       lambda event: self.mpl_update(mpl_canvas.get_tk_widget(), 
                                                toolbar, 
                                                height=height, 
                                                mpl_place=mpl_place), add='+')

    def tk_panels(self, minsize:int = 200) -> None:
        """
        Tkinter panels
    
        Create attributes with PanedWindows
        Run 'mpl_panels_config' to configure.

        Args:
            minsize (int, optional): Minimum size per panel.

        Attributes created:
            main_frame: Frame with the panels.
            main_pane: Panel in frame.
            left_pane: Panel in main panel.
            right_pane: Panel in main panel.

        To use the panels correctly, put 2 canvases in 'left_pane' and 'right_pane'.
        """

        self.main_frame = tk.Frame(self.root, background=self.color_frame)
        self.main_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.main_pane = tk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL, 
                                        border=0, sashwidth=3)
        self.left_pane = tk.PanedWindow(self.main_pane, orient=tk.VERTICAL, 
                                        border=0, sashwidth=3)
        self.right_pane = tk.PanedWindow(self.main_pane, orient=tk.VERTICAL, 
                                         border=0, sashwidth=3)

        self.main_pane.pack(fill="both", expand=True)
        self.main_pane.add(self.left_pane)
        self.main_pane.add(self.right_pane)

        self.main_pane.add(self.left_pane, minsize=minsize)
        self.main_pane.add(self.right_pane, minsize=minsize)
        self.root.update_idletasks()

    def mpl_panels_config(self, canvases:dict, titles:list = [], 
                          height:int = 32, alert:bool = True) -> None:
        """
        Matplotlib panels config

        Configure the panels, canvas, and toolbars for the panels.

        Args:
            canvases (dict): Dictionary where each key must be the 
                matplotlib canvas and the value the toolbar; if you 
                don't want a toolbar you can put None
            titles (list, optional): Title of each panel, leave 
                list empty if you don't want titles.
            height (int, optional): Toolbar height in pixels.
            alert (bool, optional): If a toolbar is configured 
                with the same instance, an alert is generated.
        """

        # Exceptions
        if titles and len(titles) != len(canvases):
            raise exception.CustomWinError(
                "'titles' can only be empty or the same size as 'canvases'.")

        try:
            if self.toolbar_config and alert:
                logger.warning(utils.text_fix("""
                    In this instance a toolbar was configured, 
                    if so you will have visual problems with the panels.
                """))
        except AttributeError:
            pass

        act_canvas = list(canvases.keys())[-1]
        def active_canvas(canvas:FigureCanvasTkAgg, graph_n:int) -> None:
            """
            Active canvas

            Change canvas focus.
            Hide the toolbar and place the new one.

            Args:
                canvas (FigureCanvasTkAgg): Focused canvas.
                graph_n (int): Graph number.
            """
            nonlocal act_canvas

            if act_canvas != canvas:
                if canvases[act_canvas]:
                    canvases[act_canvas].place_forget()
                    canvases[act_canvas].pack_forget()

                if canvases[canvas]:
                    final_height = 32/self.root.winfo_height()
                    canvases[canvas].place(relx=0, rely=1-final_height, relwidth=1, relheight=final_height)
                act_canvas = canvas

            if titles:
                if self._after_id_title:
                    self.root.after_cancel(self._after_id_title)
                    self._after_id_title = None

                self.gen_title(titles[graph_n-1])
                self._after_id_title = self.root.after(5000, lambda: self.gen_title(self.title))

        def toolbar_update(height:int) -> None:
            """
            Toolbar update

            Update the space for the toolbar.

            Args:
                height (int): Toolbar height in pixels.
            """

            final_height = height/self.root.winfo_height()
            if canvases[act_canvas] and canvases[act_canvas].winfo_manager():
                canvases[act_canvas].place(relx=0, rely=1-final_height, 
                                           relwidth=1, relheight=final_height)

            self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1-final_height)

        for i, (key, value) in enumerate(canvases.items()):
            if value and key != act_canvas:
                value.place_forget()
                value.pack_forget()

            graph_n = i + 1
            key.get_tk_widget().bind(
                "<Button-1>", lambda e, c=key, n=graph_n: active_canvas(c, n), add='+')
            key.get_tk_widget().bind(
                "<Button-3>", lambda e, c=key, n=graph_n: active_canvas(c, n), add='+')

        self.root.bind(
            '<Configure>', lambda x: toolbar_update(height=height), add='+')
        try:
            self.main_pane.sash_place(0, int(self.root.winfo_width() * 0.5), 0)
            self.left_pane.sash_place(0, 0, int(self.root.winfo_height() * 0.5))
            self.right_pane.sash_place(0, 0, int(self.root.winfo_height() * 0.5))
        except tk.TclError:
            pass

    def show(self, block:bool = True) -> None:
        """
        Show.

        Show the window.

        Args:
            block (bool, optional): Blocks the process.
        """

        if block:
            try: 
                while self.root.winfo_exists():
                    self.root.update_idletasks()
                    self.root.update()
            except tk.TclError: return
        else:
            if not self.root.winfo_exists():
                return

            self._main_after_id = self.root.after(50, lambda: self.show(block=False))

def def_style(name:str, 
              background:str | tuple[str, ...] | list[str] = '#e5e5e5', 
              frames:str = 'SystemButtonFace', 
              buttons:str = '#000000', 
              button_act:str | None = None, 
              gardient_dir:bool = True, 
              volume:str | None = None, 
              up:str | None = None, 
              down:str | None = None
              ) -> None:
    """
    Def style.

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
            'mk': {
            'u': up, 
            'd': down}

    Args:
        name (str): Name of the new style by which you will call it later.
        background (str | tuple[str, ...] | list[str], optional): 
            Background color of the axes. 
            It can be a gradient of at least 2 colors using a tuple or list.
        frames (str, optional): Background color of the frames.
        buttons (str, optional): Button color.
        button_act (str | None, optional): Color of buttons when selected or sunken.
        gardient_dir (bool, optional): The gradient direction will always 
            be top to bottom and diagonal, but you can choose whether 
            it starts from the right or left, true = right.
        volume (str | None, optional): Volume color.
        up (str | None, optional): Color when the price rises, this influences 
            the color of the candle and the bullish position indicator.
        down (str | None, optional): Color of when the price rises this influences 
            the color of the candle and the bearish position indicator.
    """
    if name in _cm.__plt_styles.keys():
        raise exception.StyleError(f"Name already in use. '{name}'")

    _cm.__plt_styles.update({
        name: {
            'bg':background or '#e5e5e5', 
            'gdir':gardient_dir, 
            'fr':frames or 'SystemButtonFace',
            'btn':buttons or '#000000', 
            'btna':button_act or '#333333', 
            'vol': volume or 'tab:orange',
            'mk': {
                'u':up or 'green',
                'd':down or 'red',
    }}})

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

    im = ax.imshow(gradient, aspect='auto', 
                   cmap=mpl.colors.LinearSegmentedColormap.from_list('custom_gradient', colors), 
                extent=(0., 1., 0., 1.), transform=ax.transAxes, zorder=-1)
    im.get_cursor_data = lambda event: None

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
    ax.title.set_color('white') # type: ignore[attr-defined]
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    semi_transparent_white = mpl.colors.to_rgba(cast(Any, "white"), alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(semi_transparent_white)
        spine.set_linewidth(1.2)

def ax_view(view:str, graphics:list[str], 
            sharex:bool = False) -> tuple[list[Axes], list[str]]:
    """
    Axes view

    Based on a str generates axes.
        Generates up to 8 axes and places them covering the entire canvas.

    Args:
        view (str): String with format: s/s/s/s each value is a graph.
        graphics (list[str]): Name of each graph, needed to process 'view'.
            If a graphic has a '/' it is replaced by ''.
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

    axes = []
    for i in range(len(pview)):
        sharex_=axes[-1] if axes and sharex else None
        nrows, ncols, rowspan, colspan = layout_rules[len(pview)](i)
        axes.append(
            plt.subplot2grid(
                (nrows, ncols), loc[i],
                rowspan=rowspan, colspan=colspan,
                sharex=sharex_
        ))

    return axes, pview

def new_paneledw(block:bool, style:dict = {}) -> None:
    """
    New paneled window

    Generate a window with panels using 'CustomWin'.

    To add a canvas, add a dictionary to the '__panel_list' list with these values:
        fig: Matplotlib Figure.
        title: Panel title.
        anim: Matplotlib FuncAnimation.
        interval: Only necessary if have an animation.
        toolbar: None, 'total' or 'limited'.
    All except fig can be None.

    Args:
        block (bool): If True, the window with the loaded panels will be displayed.
        style (dict, optional): Style used with 'fr', 'btn' and 'btna'.
    """

    # Exceptions

    if len(_cm.__panel_list) > 4:
        raise exception.CustomWinError('Maximum 4 panels')
    elif len(_cm.__panel_list) != _cm.__panel_wmax and not block:
        return
    elif len(_cm.__panel_list) <= 0:
        raise exception.CustomWinError('None panel loaded.')

    window = CustomWin(
        title=_cm.__panel_list[-1].get('title', None) if len(_cm.__panel_list) == 1 else None,
        frame_color=style.get('fr', 'SystemButtonFace'),
        buttons_color=style.get('btn', '#000000'),
        button_act=style.get('btna', '#333333'))

    window.tk_panels()
    if len(_cm.__panel_list) == 1:
        panels = [window.main_frame]
    else:
        panels = [window.left_pane, window.right_pane, 
                window.left_pane, window.right_pane]

    titles = []
    canvases = {}

    for i,panel_dict in enumerate(_cm.__panel_list):
        titles.append(panel_dict.get('title', None))
        mpl_canvas = window.mpl_canvas(fig=panel_dict['fig'], master=panels[i])

        if (anim:=panel_dict.get('anim', None)):
            window.mpl_animation(anim=anim, mpl_canvas=mpl_canvas, 
                                 interval=panel_dict.get('interval', 100))
  
        toolbar = None
        if panel_dict.get('toolbar', None):
            toolbar = window.mpl_toolbar(mpl_canvas=mpl_canvas, 
                                         movement=False 
                                            if panel_dict.get('toolbar', 'limited') == 'limited' 
                                            else True,
                                         link=True)

        canvases[mpl_canvas] = toolbar
    _cm.__panel_list = _cm.__panel_list[:-_cm.__panel_wmax]

    window.mpl_panels_config(canvases=canvases, titles=titles)
    window.show(block=block)

def add_window(fig:Figure, title:str|Callable|None = None, block:bool = True, 
              anim:FuncAnimation|None = None, interval:int|None = None, 
              style:dict|None = None, toolbar:str|None = 'total', 
              new:bool = True) -> None:
    """
    Add window

    Add a tkinter window with 'CustomWin'.

    Args:
        fig (Figure): Matplotlib Figure.
        title (str|Callable|None, optional): Window/panel title.
        block (bool, optional): Lock the thread and create 
            the window with the panels.
        anim (FuncAnimation|None, optional): Matplotlib FuncAnimation.
        interval (int|None, optional): Animation interval, 
            only necessary if there is an animation.
        style (dict|None, optional): Style to use with 'fr', 'btn' and 'btna'.
        toolbar (str|None): None, 'total' or 'limited'.
        new (bool): Create a new window or add it as a panel. True = create new.
    """

    style = style or {}
    anim = cast(Any, anim)

    if not new:
        if anim and anim.event_source: 
            anim.event_source.stop()

        _cm.__panel_list.append({
            'fig':fig,
            'title':title,
            'anim':anim,
            'interval':interval,
            'toolbar':toolbar,
        })

        plt.close(fig)
        new_paneledw(block=block, style=style)
    else:
        window = CustomWin(
            title=title,
            frame_color=style.get('fr', 'SystemButtonFace'),
            buttons_color=style.get('btn', '#000000'),
            button_act=style.get('btna', '#333333'))

        mpl_canvas = window.mpl_canvas(fig=fig)
        plt.close(fig)

        if anim:
            window.mpl_animation(anim=anim, mpl_canvas=mpl_canvas, interval=interval or 100)
        if toolbar:
            custom_toolbar = window.mpl_toolbar(
                mpl_canvas=mpl_canvas, movement=False if toolbar == 'limited' else True, link=True)
            window.mpl_toolbar_config(toolbar=custom_toolbar, mpl_canvas=mpl_canvas)

        window.show(block=block)
