"""
Custom plot module

Contains matplotlib embedding logic and colors.
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk, ImageOps
import matplotlib.colors
import matplotlib as mpl
import tkinter as tk
import os

from . import _commons as _cm

class CustomWin:

    def __init__(self, title, frame_color='SystemButtonFace', 
                 buttons_color='#000000', button_act = '#333333', 
                 geometry='1200x600'):
        self.root = tk.Tk()
        self.root.protocol('WM_DELETE_WINDOW', self._quit)
        self.root.geometry(geometry)

        self._after_id = None

        self.color_frame = frame_color
        self.color_buttons = buttons_color
        self.color_button_act = button_act

        self.root.title(title)
        self.root.after(100, self.lift)

    def lift(self):
        if not _cm.lift:
            return

        self.root.iconify()
        self.root.update()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _quit(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        if self.root.winfo_exists():
            self.root.destroy()

    def mpl_canvas(self, fig):
        frame = tk.Frame(self.root, bg=self.color_frame)
        frame.place(relx=0, rely=0, relwidth=1, relheight=0.95)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        return canvas

    def mpl_toolbar(self, mpl_canvas):
        frame = tk.Frame(self.root, bg=self.color_frame)
        frame.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)

        toolbar = CustomToolbar(mpl_canvas, frame, color_btn=self.color_buttons, 
                                color_bg=self.color_frame, color_act=self.color_button_act)
        toolbar.config(background=self.color_frame)
        toolbar.pack(expand=True)

    def show(self, block=True):
        if not self.root.winfo_exists():
            return

        if block:
            self.root.mainloop()
        else:
            self._after_id = self.root.after(50, lambda: self.show(block=False))

class CustomToolbar(NavigationToolbar2Tk):

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

    def __init__(self, canvas, window, color_btn = '#000000', 
                 color_bg = 'SystemButtonFace', color_act = '#333333'):
        super().__init__(canvas, window)

        self.window = window
        self.color_act = color_act
        self.color_btn = color_btn
        self.color_bg = color_bg

        self.icon_dir = os.path.join(mpl.get_data_path(), "images")

        self.custom_img = {}
        self.config_colors()

    def config_colors(self):
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

    def select(self, btn, img):
        btn.var.set(0)
        btn.config(image=img, bg=self.color_act, offrelief="sunken", overrelief="groove")
        return

    def deselect(self, btn):
        btn.var.set(0)
        btn.config(bg=self.color_bg, offrelief="flat", overrelief="flat")
        return

    def set_history_buttons(self):
        return

def custom_ax(ax, bg='#e5e5e5'):
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5) 

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

    semi_transparent_white = mpl.colors.to_rgba('white', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(semi_transparent_white)
        spine.set_linewidth(1.2)
