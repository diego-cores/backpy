"""
Tkinter window

Contains matplotlib embedding logic.

Variables:
    logger (Logger): Logger variable.

Classes:
    CustomWin: Create a custom window with BackPy using 'tkinter' and 'matplotlib'.
    CustomToolbar: Inherits from the 'NavigationToolbar2Tk' class to 
        modify the toolbar buttons and change colors.

Functions:
    new_paneledw: Generate a window with panels using 'CustomWin'.
    add_window: Add a tkinter window with 'CustomWin'.
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
from matplotlib.backends._backend_tk import add_tooltip # type: ignore
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageTk, ImageOps
from typing import Callable, Any, cast
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from importlib import resources
from types import MethodType
from time import monotonic
import matplotlib as mpl
import tkinter as tk
import random as rd
import numpy as np
import warnings
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
        icon_map: Dictionary of the file name of each button logo. If the 
            name has separators or '/', it will be taken as the complete path.
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
        run_command: Return a wrapper that executes a method on all instances of 'CustomToolbar'.
        run_zoom: This function are overwritten if zoom is linked.
        run_pan: This function are overwritten if pan is linked.

    Private Methods:
        _ccheck_button: Create a check button and place it in the toolbar.
    """

    _buttons: dict[Any, Any]
    winfo_children:Callable

    toolitems:tuple = (
        # Name, description, default icon (handled by NavigationToolBar), method name
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

    icon_map:dict = {
        'Home': 'home.png',
        'Back': 'back.png',
        'Forward': 'forward.png',
        'Zoom': 'zoom_to_rect.png',
        'Pan': 'move.png',
        'Save': 'filesave.png'
    }

    def __init__(self, canvas:FigureCanvasTkAgg, window:tk.Tk|tk.Toplevel, 
                 color_btn:str = '#000000', color_bg:str = 'SystemButtonFace', 
                 color_act:str = '#333333', movement:bool = True, 
                 link:bool = False, buttons:list[dict]|None = None) -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            canvas (FigureCanvasTkAgg): Canvas containing the matplotlib figure.
            window (Tk|Toplevel): Window root.
            color_btn (str, optional): Button color.
            color_bg (str, optional): Frame color.
            color_act (str, optional): Color of the sunken buttons.
            movement (bool, optional): Enable user movement buttons.
            link (bool, optional): If it is True, the toolbar connects 
                to all other toolbars with a link; only the pan and 
                zoom button is connected.
            buttons (list[dict]|None, optional): Add buttons, dict: 
                'name' (str|None): Button name, if it is None, an empty space is generated,
                'desc' (str|None): Description text, 
                'icon' (str|None): Icon path,
                'func' (Callable|None): Button function, the function must accept the instance, 
                    for check buttons call 'self._buttons[name].deselect()/select()'.
                'tggl' (bool|None): True if you want the button to be a checkbutton,
                'link' (bool|None): Link the button with all the toolbars; it only works if link=True.
                'ausl' (bool|None): If it is true and link and tggl the select/deselect is not handled automatically.
        """
        self.icon_map = dict(self.icon_map)

        if not movement:
            self.toolitems = ( # type: ignore
                ('Home', 'Reset original view', 'home', 'home'),
                (None, None, None, None),
                ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
            self.icon_map = {
                'Home': 'home.png',
                'Save': 'filesave.png'
            }

        toggle_buttons = []
        linked_buttons = []
        for btn in buttons or []:
            btn_name = btn.get('name', None)

            if btn_name is None:
                self.toolitems += ((None,None,None,None),); continue
            elif not isinstance(btn_name, str):
                raise ValueError('The button name only can be string.')

            btn_name = btn_name.lower()
            if btn_name in self.icon_map:
                raise ValueError(f"The name '{btn_name}' of the button is already in use.")

            btn_desc = btn.get('desc', None)
            btn_icon = btn.get('icon', None)
            btn_func = btn.get('func', None)
            btn_tggl = btn.get('tggl', None)
            btn_link = btn.get('link', None)
            btn_ausl = btn.get('ausl', None)

            self.icon_map.update({btn_name:btn_icon or 'home.png'})

            btn_func_name = btn_name+'_btn_custom'
            setattr(self, btn_func_name, MethodType(btn_func or (lambda x: None), self))

            if btn_link and btn_func and link:
                linked_buttons.append(btn_func_name)
                setattr(self, btn_func_name+'_link', self.run_command(
                    func=btn_func_name, button=btn_name if not btn_ausl else None))            
                btn_func_name = btn_func_name+'_link'
            if btn_tggl:
                    toggle_buttons.append(
                        {'name':btn_name, 'desc':btn_desc or '', 'func':btn_func_name})

            self.toolitems += (
                    (btn_name, btn_desc, 'home', btn_func_name),
                )

        super().__init__(canvas, window)

        for btn in toggle_buttons:
            children = self.pack_slaves()
            idx = children.index(self._buttons[btn['name']])

            self._buttons[btn['name']].destroy()
            self._ccheck_button(
                button=btn, before=children[idx + 1] if idx + 1 < len(children) else None)

        self.window = window
        self.color_act = color_act
        self.color_btn = color_btn
        self.color_bg = color_bg

        self._org_zoom = super().zoom
        self._org_pan = super().pan

        self.icon_dir = os.path.join(mpl.get_data_path(), "images")

        self.custom_img = {}
        self.config_colors()

        if link:
            linked_toolbars:dict = getattr(_cm, '__linked_toolbars')
            if not self in linked_toolbars:
                linked_toolbars[self] = []

            if movement:
                linked_toolbars[self].extend(['_org_zoom', '_org_pan'])
                setattr(self, 'run_zoom', self.run_command(func='_org_zoom'))
                setattr(self, 'run_pan', self.run_command(func='_org_pan'))

            linked_toolbars[self].extend(linked_buttons)

            setattr(_cm, '__linked_toolbars', linked_toolbars)

    def _ccheck_button(self, button:dict, before:tk.Misc|None) -> None:
        """
        Custom check button

        Create a check button and place it in the toolbar.

        Args:
            button (dict): Dict of buttons to add. 
                Keys: name, desc, func.
            before (Misc|None): Neighbor widget before which to insert this 
                button in the toolbar order. If None, it is added to the end.
        """

        var = tk.IntVar(master=self)
        b = tk.Checkbutton(
            master=self, 
            text=button['name'],
            command=getattr(self, button['func']),
            indicatoron=False, 
            variable=var,
            offrelief='flat',
            overrelief='groove', 
            borderwidth=1)
        setattr(b, 'var', var)
        setattr(b, '_image_file', None)

        b.configure(font=self._label_font)

        kw:dict = {} if before is None else {'before':before,}
        b.pack(side=tk.LEFT, **kw)
        add_tooltip(b, button['desc'])

        self._buttons[button['name']] = b

    def run_command(self, func:str, button:str|None = None) -> Callable:
        """
        Run command

        Return a wrapper that executes the 'func' method on all instances of 'CustomToolbar' 
        that have the method registered in '__linked_toolbars'.

        Args:
            func (str): __name__ of the method.
            button (str|None, optional): Button name, used for select and deselect. 
                If it's None or the button isn't a Checkbutton, nothing happens.

        Returns:
            Callable: Wrapper.
        """

        def wrapper(*args, **kwargs) -> None:
            """
            Wrapper

            Executes a method on all instances of 'CustomToolbar'.
            """
            nonlocal func

            linked_toolbars = {}
            for k, v in getattr(_cm, '__linked_toolbars').items():
                try:
                    if not k.winfo_exists():
                        continue
                    if not func in v:
                        linked_toolbars[k] = v
                        continue

                    if not button is None and isinstance(
                        (command_btn:=getattr(k, '_buttons')[button]), 
                        tk.Checkbutton) and not self is k:
                        if bool(getattr(command_btn, 'var').get()):
                            command_btn.deselect()
                        else:
                            command_btn.select()

                    getattr(k, func)(*args, **kwargs)
                    linked_toolbars[k] = v
                except tk.TclError:
                    pass

            setattr(_cm, '__linked_toolbars', linked_toolbars)
        return wrapper

    def run_zoom(self, *args:list):
        """
        Run zoom

        This function are overwritten if zoom is linked.
        """
        return super().zoom(*args)

    def run_pan(self, *args:list):
        """
        Run pan

        This function are overwritten if pan is linked.
        """
        return super().pan(*args)

    def zoom(self, *args):
        return self.run_zoom(*args)

    def pan(self, *args):
        return self.run_pan(*args)

    def set_message(self, s:str) -> None:
        self.message.set(s)

    def config_colors(self) -> None:
        """
        Config colors

        Configure the colors of the buttons and frame.
        """

        for key in self._buttons.keys():
            img_tk = self.custom_img.get(key, None)
            if not img_tk:
                filename = self.icon_map[key]
                path = filename

                sep_cond = os.sep in filename or '/' in filename
                if not sep_cond: path = os.path.join(self.icon_dir, filename)

                img = Image.open(path).convert("RGBA")
                size = self.winfo_pixels('18p')
                if sep_cond: img = img.resize((size, size), Image.Resampling.LANCZOS)

                gray = ImageOps.grayscale(img)
                colorized = ImageOps.colorize(gray, black=self.color_btn, white="#000000")

                colorized.putalpha(img.split()[-1])
                img_tk = ImageTk.PhotoImage(colorized, master=self.window)
                self.custom_img[key] = img_tk

            btn = self._buttons[key]
            if isinstance(btn, tk.Button):
                btn.config(activebackground=self.color_act)
            elif isinstance(btn, tk.Checkbutton):
                btn.config(activebackground=self.color_act, 
                    selectcolor=self.color_act, selectimage=img_tk,
                    offrelief='flat', relief='sunken')

            btn.config(image=img_tk, bg=self.color_bg, height='18p', width='18p')

        list(map(
            lambda x: x.config(bg=self.color_bg, fg=self.color_btn) 
            if not isinstance(x, tk.Frame) else x.config(bg=self.color_btn, relief=tk.RIDGE), 
            self.winfo_children()
        ))

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
        master: Tk root instance.
        root: Tkinter window.
        icon: Window icon.
        title: Window title, can be called.
        color_frame: Frames color.
        color_buttons: Buttons color.
        color_button_act: Color of the sunken buttons.
        toolbar_added_buttons: Dictionary with added toolbar buttons.

    Private Attributes:
        _regen_title: If 'title' is callable, it is called again every 5s.
        _after_id_lift: After id that lifts the window with 'lift'.
        _main_after_id: 'root.after' id used to avoid errors by not blocking the process.
        _after_id_title: After id of title generation.
        _tlbar_cords: Callable to control the coordinates in the toolbar.

    Methods:
        gen_title: Generate and put the title to the window.
        focus_title: Change title for a given milliseconds.
        config_icon: Put the icon on the application and change its color.
        lift: Focus on the window and jump over the others.
        mpl_canvas: Put your matplotlib figure inside the window.
        mpl_canvas_crosshair: Get a cross on the canvas, which can be deactivated.
        mpl_animation: Configure the window and run an animation.
        mpl_update: Updates the position of the toolbar and the canvas.
        mpl_toolbar: Put the matplotlib toolbar in the window.
        mpl_toolbar_config: Configure the toolbar, but don't 
            do this if you're going to use 'tk_panels'.
        tk_panels: Generates 'PanedWindow' panels.
        mpl_panels_config: Configure the panels.
        supp_warnings: Static method for suppress ignorable matplotlib wanings.
        show: Show the window.

    Private Methods:
        _quit: Closes the window without errors.
    """

    def __init__(self, master:tk.Tk|None = None,
                 title:str|Callable|None = 'BackPy interface', 
                 frame_color:str = 'SystemButtonFace', 
                 buttons_color:str = '#000000', 
                 button_act:str = '#333333', 
                 geometry:str = '1200x600',
                 regen_title:bool = True) -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            master (Tk|None, optional): Pass your own Tk instance to create the Toplevel.
            title (str|Callable|None, optional): Window title.
            frame_color (str, optional): Color of the toolbar and other frames.
            buttons_color (str, optional): Button color.
            button_act (str, optional): Color of the sunken buttons.
            geometry (str, optional): Window geometry.
            regen_title(bool, optional): If 'title' is callable, 
                it is called again every 5s.
        """

        if not isinstance(master, tk.Tk) and not isinstance(_cm._tkinter_root, tk.Tk):
            _cm._tkinter_root = tk.Tk()
            _cm._tkinter_root.withdraw()

        root = master or _cm._tkinter_root
        assert isinstance(root, tk.Tk)

        self.master:tk.Tk = root
        self.root = tk.Toplevel(self.master)

        self.root.geometry(geometry)
        self.root.config(bg=frame_color)

        self.icon = None
        self.title = title if title else lambda: rd.choice(
            _cm._random_titles if _cm._random_titles else ['BackPy'])

        self._main_after_id = None
        self._regen_title = regen_title
        self._tlbar_cords = None

        self.color_frame = frame_color
        self.color_buttons = buttons_color
        self.color_button_act = button_act

        self.toolbar_added_buttons = {}

        self.config_icon()

        self._after_id_title = None
        self.gen_title(title=self.title)
    
        self._after_id_lift = self.root.after(100, self.lift)

        self.root.minsize(int(self.root.winfo_screenwidth()*0.4), 
                          int(self.root.winfo_screenheight()*0.4))

        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    def gen_title(self, title:str|Callable|None = None, ms:int|None = 10000):
        """
        Gen title

        Add a title to the window.
        It is generated if title is callable.

        Args:
            title (str|Callable|None, optional): Title.
            ms (int): If 'title' is callable, then it is called every 'ms' milliseconds.
                If it's None, it won't be called again.
        """

        if not self.root.winfo_exists():
            return

        if self._after_id_title:
            self.root.after_cancel(self._after_id_title)
            self._after_id_title = None

        if callable(title):
            if ms:
                self._after_id_title = self.root.after(
                    ms, lambda: self.gen_title(title=self.title, ms=ms))
            title = title()

        if title is None:
            title = 'Window from BackPy'
        self.root.title(title)

    def focus_title(self, title:str|Callable|None = None, ms:int = 5000) -> None:
        """
        Focus title

        Change title for a given milliseconds.

        Args:
            title (str|Callable|None, optional): New title.
            ms (int, optional): Milliseconds that the title will last.
        """

        if self._after_id_title:
            self.root.after_cancel(self._after_id_title)
            self._after_id_title = None

        self.gen_title(title)
        self._after_id_title = self.root.after(ms, lambda: self.gen_title(self.title))

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
        self.root.tk.call('wm', 'iconphoto', str(self.root), self.icon)

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
                   master:tk.PanedWindow|tk.Frame|tk.Tk|tk.Toplevel|None = None,
                   title:Callable|str|None = None) -> FigureCanvasTkAgg:
        """
        Matplotlib canvas

        Put your matplotlib figure inside the window.

        Note:
            Execute `plt.close(fig)` before this function.
            In matplotlib >= 3.11.0, calling it after may cause the toolbar 
            pan and zoom to stop working.

        Args:
            fig (Figure): Figure from matplotlib.
            master (PanedWindow|Frame|Tk|Toplevel|None, optional): Where the canvas is drawn.
            title (str|None, optional): When you click on the canvas, the title appears.

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
                call_rdraw (bool, optional): It's called 'canvas.draw' when redefining the function.
            """

            if not self.root.winfo_exists():
                return

            try:
                if call_rdraw:
                    re_draw()
                canvas.draw = re_draw
                setattr(_cm, '__anim_puntil', monotonic()+0.4)
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

            setattr(_cm, '__anim_puntil', None)
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

        def on_state(event:tk.Event|None = None) -> None:
            """
            On state
    
            If the window changes state it does a '.after' to redraw 
                and modifies the 'canvas.draw' method.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _state_after_id, _resize_after_id, last_state, last_pos

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
                setattr(_cm, '__anim_puntil', None)

                widget.after_cancel(_resize_after_id)
                _state_after_id = widget.after(400, on_state_resize)

            pos = (self.root.winfo_x(), self.root.winfo_y())
            if last_pos != pos:
                last_pos = pos

                setattr(_cm, '__anim_puntil', monotonic()+0.4)

        def on_click(event:tk.Event|None = None) -> None:
            """
            On click
    
            Change the state of '__anim_puntil' to None 
            and change the title.

            Args:
                event (Event | None, optional): Event.
            """

            setattr(_cm, '__anim_puntil', None)

            if title:
                self.focus_title(title=title, ms=5000)

        def on_unclick(event:tk.Event|None = None) -> None:
            """
            On unclick
    
            When you stop hold the canvas, '__anim_puntil' is set.

            Args:
                event (Event | None, optional): Event.
            """

            setattr(_cm, '__anim_puntil', monotonic()+0.4)

        widget.bind('<Button-1>', on_click, add='+')
        widget.bind('<Button-3>', on_click, add='+')
        widget.bind('<ButtonRelease-1>', on_unclick, add='+')
        widget.bind('<ButtonRelease-3>', on_unclick, add='+')
    
        widget.bind('<Configure>', on_resize, add='+')
        self.root.bind('<Configure>', on_state, add='+')

        return canvas

    def mpl_canvas_crosshair(self, fig:Figure, mpl_canvas:FigureCanvasTkAgg, 
                            dot_size:float = 2, color:str|None = None) -> None:
        """
        Matplotlib canvas crosshair

        Get a cross on the canvas, which can be deactivated.

        Args:
            fig (Figure): Figure from matplotlib.
            mpl_canvas (FigureCanvasTkAgg): Canvas containing the matplotlib figure.
            dot_size (float, optional): Size of the center point.
            color (str|None, optional): Color of the crosshair.
        """
        if dot_size <= 0:
            raise exception.CustomWinError('dot_size must be non-negative and non-zero')

        cords = None
        widget = mpl_canvas.get_tk_widget()
        _axes_vertices:dict = {}
        _get_verts = False
        _toggle_cross = False
        _toggle_snap = False

        _last_state = 0
        CTRL_STATE = 4

        def on_draw(event:tk.Event|None = None) -> None:
            """
            On draw

            It is executed after redrawing the canvas to calculate the vertices.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _get_verts

            _get_verts = True

        def on_move(event:tk.Event|None = None) -> None:
            """
            On move

            It is sent every time the mouse moves.

            Args:
                event (Event | None, optional): Event.
            """

            if event is None:
                return

            crosshair(event)
            ctrl_mng(event)

        def ctrl_mng(event:tk.Event) -> None:
            """
            Ctrl manager

            Control interaction with the 'ctrl' key.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal _toggle_snap, _toggle_cross, _last_state

            is_press = event.state == CTRL_STATE
            if is_press or _last_state == CTRL_STATE:
                if snap_mode and crss_mode:
                    _toggle_snap = not is_press
                elif crss_mode:
                    _toggle_snap = is_press
                else:
                    if not snap_mode:
                        _toggle_snap = False
                    _toggle_cross = is_press

            if event.state != _last_state:
                _last_state = event.state

        def crosshair(event:tk.Event) -> None:
            """
            Crosshair

            Draw crosshair with 'crosshair' tag.

            Args:
                event (Event | None, optional): Event.
            """
            nonlocal fig, dot_size, _get_verts, _axes_vertices, _toggle_cross, _toggle_snap, cords

            widget.delete("crosshair")
            if not _toggle_cross:
                return

            h_def = fig.get_figheight() * fig.dpi
            disp_y = h_def - event.y

            actual_ax = [
                ax for ax in fig.axes if ax.get_window_extent().contains(event.x, disp_y)]
            if not actual_ax:
                return
            actual_ax = actual_ax[0]

            data_x, data_y = actual_ax.transData.inverted().transform((event.x, disp_y))

            if _get_verts and _toggle_snap:
                _axes_vertices = {}

                for ax in fig.axes:
                    _axes_vertices[ax] = []

                    for l in ax.get_children():
                        vert = utils.get_vertices(l)

                        if vert is None or len(vert) <= 0:
                            continue

                        _axes_vertices[ax].extend(vert)
                    _axes_vertices[ax] = np.asarray(_axes_vertices[ax])
                _get_verts = False

            if _toggle_snap:
                if _axes_vertices and len(_axes_vertices[actual_ax]) == 0:
                    return

                px_mg = widget.winfo_width()*0.005
                x_lo = actual_ax.transData.inverted().transform((event.x - px_mg, 0))[0]
                x_hi = actual_ax.transData.inverted().transform((event.x + px_mg, 0))[0]

                verts = _axes_vertices[actual_ax][
                    (_axes_vertices[actual_ax][:, 0] >= x_lo) &
                    (_axes_vertices[actual_ax][:, 0] <= x_hi)
                ]
                if len(verts) <= 0:
                    verts = [[data_x, data_y]]

                xy_disp = actual_ax.transData.transform(verts)
                idx = int(np.argmin(np.abs(xy_disp[:, 1] - disp_y)))

                best = verts[idx]
                best_disp = xy_disp[idx]
  
                cords = (f'(x, y) = ({actual_ax.format_xdata(best[0])}, '
                    f'{actual_ax.format_ydata(best[1])})')

                h_y_tk = h_def - best_disp[1]
                dot_y_tk = h_y_tk
                dot_x_tk = best_disp[0]
            else:
                h_y_tk = event.y
                dot_x_tk = event.x
                dot_y_tk = event.y
                cords = (f'(x, y) = ({actual_ax.format_xdata(data_x)}, '
                    f'{actual_ax.format_ydata(data_y)})')

            kw = {'dash': (4, 4), 'tags': 'crosshair'}
            widget.create_line(event.x, 0, event.x, widget.winfo_height(), 
                            fill=color or "#e05c5c", **kw)
            widget.create_line(0, h_y_tk, widget.winfo_width(), h_y_tk, 
                            fill=color or "#5c9ee0", **kw)

            widget.create_oval(
                dot_x_tk - dot_size, dot_y_tk - dot_size, dot_x_tk + dot_size, dot_y_tk + dot_size,
                fill=color or '#5c9ee0', outline='white', tags='crosshair',
            )

        def on_leave(event:tk.Event|None = None) -> None:
            """
            On leave

            Delete the drawing with the tag 'crosshair'.

            Args:
                event (Event | None, optional): Event.
            """

            widget.delete('crosshair')

        def set_cords(self:CustomToolbar, s:str) -> None:
            """
            Set coordinates

            Modify the toolbar 'message' text label to display coordinates.

            Args:
                self (CustomToolbar): Toolbar instance.
                s (str): New coordinates.
            """
            nonlocal cords, crss_mode 

            if cords and crss_mode:
                s = cords
            self.message.set(s)
        self._tlbar_cords = set_cords

        crss_mode = False
        def cross_button(self:CustomToolbar) -> None:
            """
            Crosshair button

            Activate or deactivate the crosshair.

            Args:
                self (CustomToolbar): Toolbar instance.
            """
            nonlocal crss_mode, _toggle_cross
            crss_mode = not crss_mode

            if crss_mode:
                self._buttons['cross'].select()
                _toggle_cross = True
            else:
                self._buttons['cross'].deselect()
                _toggle_cross = False

        snap_mode = False
        def snap_button(self:CustomToolbar) -> None:
            """
            Snap button

            Activate or deactivate the snap to vertices.

            Args:
                self (CustomToolbar): Toolbar instance.
            """
            nonlocal snap_mode, crss_mode, _toggle_snap

            if crss_mode:
                snap_mode = not snap_mode

            if snap_mode:
                self._buttons['snap'].select()
                _toggle_snap = True
            else:
                _toggle_snap = False
                self._buttons['snap'].deselect()

        self.toolbar_added_buttons.update({'crosshair':[
            {},
            {
                'name':'Cross', 
                'desc':"Crosshair pointer, press 'ctrl' to snap", 
                'icon':str(resources.files('backpy.assets') / 'cross.png'),
                'func':cross_button,
                'tggl':True,
                'link':True,
            },
            {
                'name':'Snap', 
                'desc':"Snap to vertices mode, reverse the use of 'ctrl'", 
                'icon':str(resources.files('backpy.assets') / 'snap.png'),
                'func': snap_button,
                'tggl':True,
                'link':True,
                'ausl':True,
            },
        ]})

        mpl_canvas.mpl_connect('draw_event', on_draw)

        widget.bind('<Motion>', on_move, add='+')
        widget.bind('<Leave>', on_leave, add='+')

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

        anim_after_id = None
        bind_id = None
        last_pos = (0,0)

        def on_anim() -> None:
            """
            On animation

            Update the animation without blocking the thread.
            """
            nonlocal anim_after_id, bind_id

            if not self.root.winfo_exists():
                return

            if (self.root.winfo_ismapped()
                and getattr(mpl_canvas.draw, "__func__", None) 
                    is type(mpl_canvas).draw
                and not getattr(_cm, '__anim_puntil') is None
                and monotonic() >= getattr(_cm, '__anim_puntil')):
                try:
                    anim.frame_seq = iter(anim.frame_seq)
                    frame = next(anim.frame_seq)

                    getattr(anim, '_draw_next_frame')(frame, getattr(anim, '_blit'))
                except StopIteration:
                    if bind_id: self.root.unbind('<Configure>', bind_id)
                    anim_after_id = None; return

            anim_after_id = self.root.after(
                interval if interval >= 100 else 100, on_anim)

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
            if last_pos != pos and anim_after_id:
                last_pos = pos

                self.root.after_cancel(anim_after_id)
                anim_after_id = self.root.after(400, on_anim)

        anim_after_id = self.root.after(1000, on_anim)
        bind_id = self.root.bind('<Configure>', on_resize, add='+')

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
                    movement:bool = True, link:bool = False,
                    add_buttons:list[dict]|None = None) -> CustomToolbar:
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
            add_buttons (list[dict]|None, optional): Add buttons, dict: 
                'name': Button name, if it is None, an empty space is generated,
                'desc': Description text, 
                'icon': Icon path,
                'func': Button function, the function must accept the instance, 
                    for check buttons call 'self._buttons[name].deselect()/select()'.
                'tggl': True if you want the button to be a checkbutton,
                'link': Link the button with all the toolbars; it only works if link=True.

        Returns:
            CustomToolbar: Toolbar.
        """

        add_buttons = add_buttons if add_buttons else []
        for v in self.toolbar_added_buttons.values(): add_buttons.extend(v)

        toolbar = CustomToolbar(mpl_canvas, self.root, color_btn=self.color_buttons, 
                                color_bg=self.color_frame, color_act=self.color_button_act,
                                movement=movement, link=link, buttons=add_buttons)
        toolbar.config(bg=self.color_frame)

        if self._tlbar_cords:
            toolbar.set_message = MethodType(self._tlbar_cords, toolbar)

        self.toolbar_added_buttons = {}
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

    def mpl_panels_config(self, canvases:dict, height:int = 32, 
                        alert:bool = True) -> None:
        """
        Matplotlib panels config

        Configure the panels, canvas, and toolbars for the panels.

        Args:
            canvases (dict): Dictionary where each key must be the 
                matplotlib canvas and the value the toolbar; if you 
                don't want a toolbar you can put None
            height (int, optional): Toolbar height in pixels.
            alert (bool, optional): If a toolbar is configured 
                with the same instance, an alert is generated.
        """

        try:
            if self.toolbar_config and alert:
                logger.warning(utils.text_fix("""
                    In this instance a toolbar was configured, 
                    if so you will have visual problems with the panels.
                """))
        except AttributeError:
            pass

        act_canvas = list(canvases.keys())[-1]
        def active_canvas(canvas:FigureCanvasTkAgg) -> None:
            """
            Active canvas

            Change canvas focus.
            Hide the toolbar and place the new one.

            Args:
                canvas (FigureCanvasTkAgg): Focused canvas.
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

        for key, value in canvases.items():
            if value and key != act_canvas:
                value.place_forget()
                value.pack_forget()

            key.get_tk_widget().bind(
                "<Button-1>", lambda e, c=key: active_canvas(c), add='+')
            key.get_tk_widget().bind(
                "<Button-3>", lambda e, c=key: active_canvas(c), add='+')

        self.root.bind(
            '<Configure>', lambda x: toolbar_update(height=height), add='+')
        try:
            self.main_pane.sash_place(0, int(self.root.winfo_width() * 0.5), 0)
            self.left_pane.sash_place(0, 0, int(self.root.winfo_height() * 0.5))
            self.right_pane.sash_place(0, 0, int(self.root.winfo_height() * 0.5))
        except tk.TclError:
            pass

    @staticmethod
    def supp_warnings(func:Callable) -> Callable:
        """
        Suppress warnings

        Suppress ignorable matplotlib wanings.

        Args:
            func (Callable, optional): Function.

        Returns:
            Callable: Wrapper, 'wrapper' or 'func' is returned depending 
            on 'mpl_warning_supp' global variable.
        """

        def wrapper(*args, **kwargs) -> Any:
            """
            Wrapper function

            Suppress warnings.

            Returns:
                Any: Function result.
            """

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Attempt to set non-positive ylim")
                warnings.filterwarnings("ignore", message="overflow encountered in power")
                return func(*args, **kwargs)

        return wrapper if _cm.mpl_warning_supp else func

    @supp_warnings
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

def new_paneledw(block:bool, force:bool = False, style:dict = {}) -> None:
    """
    New paneled window

    Generate a window with panels using 'CustomWin'.

    To add a canvas, add a dictionary to the '__panel_list' list with these values:
        fig: Matplotlib Figure.
        title: Panel title.
        untitle: Custom unfocus titles.
        anim: Matplotlib FuncAnimation.
        interval: Only necessary if have an animation.
        toolbar: None, 'total' or 'limited'.
    All except fig can be None.

    Args:
        block (bool): If True, the window with the loaded panels will be displayed.
        force (bool, optional): If True, skips the panel count check and creates 
            the window immediately, regardless of '__panel_wmax'.
        style (dict, optional): Style used with 'fr', 'btn' and 'btna'.
    """

    # Exceptions
    if len(_cm.__panel_list) > 4:
        raise exception.CustomWinError('Maximum 4 panels')
    elif not force and len(_cm.__panel_list) != _cm.__panel_wmax and not block:
        return
    elif len(_cm.__panel_list) <= 0:
        raise exception.CustomWinError('None panel loaded.')

    custom_unfocus = []
    for i in _cm.__panel_list:
        if not i['untitle']: continue
        custom_unfocus.extend(i['untitle'])

    btn_color = style.get('btn', '#000000')
    window = CustomWin(
        title=(lambda: rd.choice(custom_unfocus)) if len(custom_unfocus) > 0 else None,
        frame_color=style.get('fr', 'SystemButtonFace'),
        buttons_color=btn_color,
        button_act=style.get('btna', '#333333'))

    window.tk_panels()
    if len(_cm.__panel_list) == 1:
        panels = [window.main_frame]
    else:
        panels = [window.left_pane, window.right_pane, 
                window.left_pane, window.right_pane]

    canvases = {}

    for i,panel_dict in enumerate(_cm.__panel_list):
        mpl_canvas = window.mpl_canvas(fig=panel_dict['fig'], master=panels[i], 
                                    title=panel_dict.get('title', None))
        window.mpl_canvas_crosshair(fig=panel_dict['fig'], mpl_canvas=mpl_canvas, 
                                color=style.get('crss', btn_color))

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

    window.mpl_panels_config(canvases=canvases)
    window.show(block=block)

def add_window(fig:Figure, title:str|Callable|None = None, block:bool = True, 
              anim:FuncAnimation|None = None, interval:int|None = None, 
              style:dict|None = None, toolbar:str|None = 'total', 
              focus_title:bool = True, custom_unfocus:list[str]|None = None, 
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
        focus_title (bool, optional): It only works if 'new' = True. 
            If true, random titles will appear after 5 seconds of clicking the canvas.
        custom_unfocus (list[str]|None, optional): List of unfocus titles; 
            titles will be chosen randomly from this list. 
            If 'None', titles will be taken from '_random_titles'.
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
            'untitle':custom_unfocus,
            'anim':anim,
            'interval':interval,
            'toolbar':toolbar,
        })

        plt.close(fig)
        new_paneledw(block=block, style=style)
    else:
        btn_color = style.get('btn', '#000000')

        window = CustomWin(
            title=((lambda: rd.choice(custom_unfocus)) if custom_unfocus else None) if focus_title else title,
            frame_color=style.get('fr', 'SystemButtonFace'),
            buttons_color=btn_color,
            button_act=style.get('btna', '#333333'))

        plt.close(fig)
        mpl_canvas = window.mpl_canvas(fig=fig, title=title if focus_title else None)

        window.mpl_canvas_crosshair(fig=fig, mpl_canvas=mpl_canvas, color=style.get('crss', btn_color))
        if anim: window.mpl_animation(anim=anim, mpl_canvas=mpl_canvas, interval=interval or 100)

        custom_toolbar = None
        if toolbar:
            custom_toolbar = window.mpl_toolbar(
                mpl_canvas=mpl_canvas, movement=False if toolbar == 'limited' else True, link=True)
            window.mpl_toolbar_config(toolbar=custom_toolbar, mpl_canvas=mpl_canvas)

        if len(_cm.__panel_list) > 0 and block:
            new_paneledw(block=False, force=True, style=style)

        window.show(block=block)
