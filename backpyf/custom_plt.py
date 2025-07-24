
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk, ImageOps
import matplotlib as mpl
import tkinter as tk
import os

class App:
    def __init__(self, title):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.geometry("1200x600")

        self.color_bg = '#ffffff' # '#1e1e1e'
        self.color_frame = "#ffffff" # '#161616'
        self.color_buttons = '#000000' # '#ffffff'

        self.root.title(title)
        self.root.configure(bg=self.color_bg)

    def _quit(self):
        self.root.quit()
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
                                color_bg=self.color_frame)
        toolbar.config(background=self.color_frame)
        toolbar.pack(expand=True)

    def show(self, block=True):
        if block:
            self.root.mainloop()
        else:
            self.root.update()
            self.root.after(50, lambda: self.show(block=False))

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

    def __init__(self, canvas, window, color_btn = '#000000', 
                 color_bg = '#ffffff', color_act = '#333333'):
        super().__init__(canvas, window)

        icon_dir = os.path.join(mpl.get_data_path(), "images")

        icon_map = {
            'Home': 'home.png',
            'Back': 'back.png',
            'Forward': 'forward.png',
            'Zoom': 'zoom_to_rect.png',
            'Pan': 'move.png',
            'Save': 'filesave.png'
        }

        self.custom_img = {}
        for key, filename in icon_map.items():

            path = os.path.join(icon_dir, filename)
            img = Image.open(path).convert("RGBA")

            gray = ImageOps.grayscale(img)
            colorized = ImageOps.colorize(gray, black=color_btn, white="#000000")

            colorized.putalpha(img.split()[-1])
            img_tk = ImageTk.PhotoImage(colorized, master=window)
            self.custom_img[key] = img_tk

            btn = self._buttons.get(key)

            def select_personalizated(btn, img):
                btn.var.set(0)
                btn.config(image=img, bg=color_act, offrelief="sunken", overrelief="groove")
                return

            def deselect_personalizated(btn):
                btn.var.set(0)
                btn.config(bg=color_bg, offrelief="flat", overrelief="flat")
                return

            if isinstance(btn, tk.Button):
                btn.config(activebackground=color_act)
            elif isinstance(btn, tk.Checkbutton):
                btn.select = lambda btn=btn, img=img_tk: select_personalizated(btn, img)
                btn.deselect = lambda btn=btn: deselect_personalizated(btn)
                btn.config(activebackground=color_act, selectcolor=color_act)
            btn.config(image=img_tk, bg=color_bg)

        list(map(lambda x: x.config(bg=color_bg, fg=color_btn), self.winfo_children()[-2:]))
