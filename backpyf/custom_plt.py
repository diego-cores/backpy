"""
Custom plot module

Contains matplotlib embedding logic and colors.

Classes:
    CustomWin: Create a custom window with BackPy using 'tkinter' and 'matplotlib'.
    CustomToolbar: Inherits from the 'NavigationToolbar2Tk' class to 
        modify the toolbar buttons and change colors.

Functions:
    def_style: Define a new style to plot your graphics.
    gradient_ax: Create a diagonal background gradient on the 'ax' with 'ax.imshow'.
    custom_ax: Aesthetically configures an axis.
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk, ImageOps
from matplotlib.figure import Figure
from importlib import resources
import matplotlib.colors
import matplotlib as mpl
import tkinter as tk
import numpy as np
import os

from . import _commons as _cm
from . import exception

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

    Methods:
        config_colors: Configure the colors of the buttons and frame.
        select: Changes the style of the button when selected.
        deselect: Changes the style of the button when deselect.
    """

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
                 color_act:str = '#333333') -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            canvas (FigureCanvasTkAgg): Canvas containing the matplotlib figure.
            window (Tk): Window root.
            color_btn (str, optional): Button color.
            color_bg (str, optional): Frame color.
            color_act (str, optional): Color of the sunken buttons.
        """

        super().__init__(canvas, window)

        self.window = window
        self.color_act = color_act
        self.color_btn = color_btn
        self.color_bg = color_bg

        self.icon_dir = os.path.join(mpl.get_data_path(), "images")

        self.custom_img = {}
        self.config_colors()

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

            btn = self._buttons.get(key)

            if isinstance(btn, tk.Button):
                btn.config(activebackground=self.color_act)
            elif isinstance(btn, tk.Checkbutton):
                btn.select = lambda btn=btn, img=img_tk: self.select(btn, img)
                btn.deselect = lambda btn=btn: self.deselect(btn)
                btn.config(activebackground=self.color_act, selectcolor=self.color_act)
            btn.config(image=img_tk, bg=self.color_bg)

        list(map(lambda x: x.config(bg=self.color_bg, fg=self.color_btn), self.winfo_children()[-2:]))

    def select(self, btn:tk.Button, img:ImageTk.PhotoImage) -> None:
        """
        Select

        Changes the style of the button when selected.

        Args:
            btn (Button): Button.
            img (PhotoImage): Button icon.
        """

        btn.var.set(0)
        btn.config(image=img, bg=self.color_act, offrelief="sunken", overrelief="groove")

    def deselect(self, btn:tk.Button) -> None:
        """
        Deselect

        Changes the style of the button when deselect.

        Args:
            btn (Button): Button.
        """

        btn.var.set(0)
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
        color_frame: Frames color.
        color_buttons: Buttons color.
        color_button_act: Color of the sunken buttons.

    Private Attributes:
        _after_id: 'root.after' id used to avoid errors by not blocking the process.

    Methods:
        config_icon: Put the icon on the application and change its color.
        lift: Focus on the window and jump over the others.
        mpl_canvas: Put your matplotlib figure inside the window.
        mpl_update: Updates the position of the toolbar and the canvas.
        mpl_toolbar: Put the matplotlib toolbar in the window.
        show: Show the window.

    Private Methods:
        _quit: Closes the window without errors.
    """

    def __init__(self, title:str = 'BackPy interface', 
                 frame_color:str = 'SystemButtonFace', 
                 buttons_color:str = '#000000', 
                 button_act:str = '#333333', 
                 geometry:str = '1200x600') -> None:
        """
        __init__

        Builder for initializing the class.

        Args:
            title (str, optional): Window title.
            frame_color (str, optional): Color of the toolbar and other frames.
            buttons_color (str, optional): Button color.
            button_act (str, optional): Color of the sunken buttons.
            geometry (str, optional): Window geometry.
        """

        self.root = tk.Tk()
        self.root.geometry(geometry)
        self.root.config(bg=frame_color)

        self.icon = None
        self._after_id = None

        self.color_frame = frame_color
        self.color_buttons = buttons_color
        self.color_button_act = button_act

        self.config_icon()
        self.root.title(title)
        self._after_id_lift = self.root.after(100, self.lift)

        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    def config_icon(self) -> None:
        """
        Configure icon.

        Put the icon on the application and change its color.
        """

        with resources.path('backpyf.assets', 'icon128x.png') as icon_path:
            img = Image.open(icon_path).convert("RGBA")

        gray = ImageOps.grayscale(img)
        colorized = ImageOps.colorize(gray, black=self.color_buttons, 
                                      white="#000000")
        colorized.putalpha(img.split()[-1])

        self.icon = ImageTk.PhotoImage(colorized, master=self.root)

        self.root.tk.call('wm', 'iconphoto', self.root._w, self.icon)

    def lift(self) -> None:
        """
        Lift.

        Focus on the window and jump over the others.
        """

        self._after_id_lift = None

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

        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self._after_id_lift:
            self.root.after_cancel(self._after_id)
            self._after_id_lift = None

        if self.root.winfo_exists():
            self.root.destroy()

    def mpl_canvas(self, fig:Figure) -> FigureCanvasTkAgg:
        """
        Matplotlib canvas.

        Put your matplotlib figure inside the window.

        Args:
            fig (Figure): Figure from matplotlib.

        Return:
            FigureCanvasTkAgg: Resulting canvas figure.
        """

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        return canvas

    def mpl_update(self, canvas:FigureCanvasTkAgg, toolbar:CustomToolbar, 
                   height:int = 32, mpl_place:bool = True) -> None:
        """
        Matplotlib update.

        Updates the position of the toolbar and the canvas.

        Args:
            canvas (FigureCanvasTkAgg): Canvas figure.
            toolbar (CustomToolbar): Toolbar object.
            height (int, optional): Toolbar height in pixels.
            mpl_place (bool, optional): If you want 'mpl_canvas' not to change shape, 
                leave it set to False.
        """

        final_height = height/self.root.winfo_height()
        toolbar.place(relx=0, rely=1-final_height, relwidth=1, relheight=final_height)

        if mpl_place:
            canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1-final_height)

    def mpl_toolbar(self, mpl_canvas:FigureCanvasTkAgg, 
                    height:int = 32, mpl_place:bool = True) -> CustomToolbar:
        """
        Matplotlib toolbar.

        Put the matplotlib toolbar in the window.

        Args:
            mpl_canvas (FigureCanvasTkAgg): Canvas figure.
            height (int, optional): Toolbar height in pixels.
            mpl_place (bool, optional): If you want 'mpl_canvas' not to change shape, 
                leave it set to False.

        Return:
            CustomToolbar: Toolbar.
        """

        toolbar = CustomToolbar(mpl_canvas, self.root, color_btn=self.color_buttons, 
                                color_bg=self.color_frame, color_act=self.color_button_act)
        toolbar.config(bg=self.color_frame)
        self.mpl_update(mpl_canvas, toolbar, height=height, mpl_place=mpl_place)

        self.root.bind("<Configure>", 
                       lambda x: self.mpl_update(mpl_canvas, 
                                                toolbar, 
                                                height=height, 
                                                mpl_place=mpl_place))

        return toolbar

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

            self._after_id = self.root.after(50, lambda: self.show(block=False))

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

def gradient_ax(ax:matplotlib.axes._axes.Axes, 
                colors:list, right:bool=False) -> None:
    """
    Gradient axes.

    Create a diagonal background gradient on the 'ax' with 'ax.imshow'.

    Args:
        ax (Axes): Axes to draw.
        colors (list): List of the colors of the garden in order.
        right (bool, optional): Corner from which the gradient 
            starts if False starts from the top left.
    """

    if len(colors) < 1:
        colors = ['white']

    gradient = (np.linspace(0, 1, 256).reshape(-1, 1) 
                + (np.linspace(0, 1, 256) 
                   if right else -np.linspace(0, 1, 256)))

    im = ax.imshow(gradient, aspect='auto', 
                   cmap=mpl.colors.LinearSegmentedColormap.from_list("custom_gradient", colors), 
                extent=[0, 1, 0, 1], transform=ax.transAxes, zorder=-1)
    im.get_cursor_data = lambda event: None

def custom_ax(ax:matplotlib.axes._axes.Axes, 
              bg = '#e5e5e5', edge:bool = False) -> None:
    """
    Custom axes.

    Aesthetically configures an axis.

    Note:
        The gradient can change the 'ax' limits.

    Args:
        ax (Axes): Axes to config.
        bg (str, optional): Background color of the axis, 
            if it is a list or tuple a gradient will be created.
        edge (bool, optional): If the background is a gradient, this 
            determines which corner you launch from, false left, true right.
    """

    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5) 

    if (isinstance(bg, tuple) or isinstance(bg, list)) and len(bg) > 1:
        gradient_ax(ax, bg, right=edge)
    else:
        ax.set_facecolor(bg)

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    semi_transparent_white = mpl.colors.to_rgba('white', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(semi_transparent_white)
        spine.set_linewidth(1.2)
