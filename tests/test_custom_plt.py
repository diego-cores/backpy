"""
Custom plot Test
"""

from backpy import custom_plt as cs_plt
from backpy import exception as ex
from backpy import _commons as _cm

from unittest.mock import patch, MagicMock
import unittest

import matplotlib.pyplot
import numpy as np
import matplotlib
import sys

matplotlib.use("Agg")

if "tkinter" not in sys.modules:
    try: import tkinter
    except ImportError:
        sys.modules["tkinter"] = MagicMock()

class TestCustomPlt(unittest.TestCase):
    def setUp(self) -> None:
        self.fig, self.ax = matplotlib.pyplot.subplots()
        self.graphics = ['price', 'v', 'rsi/', '\\macd']

    def tearDown(self) -> None:
        matplotlib.pyplot.close(self.fig)

    def test_def_style(self) -> None:
        """
        Test 'def_style'

        Verify that '__plt_styles' is updated correctly.
        """

        cs_plt.def_style(
            name='dark',
            background='#000000',
            frames='#111111',
            buttons='#222222',
            up='green',
            down='red'
        )

        self.assertIn('dark', getattr(_cm, '__plt_styles'))

        style = getattr(_cm, '__plt_styles')['dark']
        self.assertEqual(style['bg'], '#000000')
        self.assertEqual(style['fr'], '#111111')
        self.assertEqual(style['btn'], '#222222')
        self.assertEqual(style['mk']['u'], 'green')
        self.assertEqual(style['mk']['d'], 'red')

    def test_def_style_duplicate_raises(self) -> None:
        """
        Test 'def_style'

        Verify that duplicates cannot be created.
        """

        cs_plt.def_style(name='classic')

        with self.assertRaises(ex.StyleError):
            cs_plt.def_style(name='classic')

    def test_gradient_ax(self) -> None:
        """
        Test 'gradient_ax'

        Verify that with two normal colors, an image with a custom colormap is created.
        """

        colors = ['#000000', '#FFFFFF']
        cs_plt.gradient_ax(self.ax, colors)

        images = self.ax.get_images()
        self.assertEqual(len(images), 1)

        img = images[0]
        cmap = img.get_cmap()
        self.assertIn('custom_gradient', cmap.name)

    def test_gradient_ax_empty_colors(self) -> None:
        """
        Test 'gradient_ax'

        Verify that the default color is used when the colors are not specified.
        """

        cs_plt.gradient_ax(self.ax, [])

        images = self.ax.get_images()
        self.assertEqual(len(images), 1)

        cmap = images[0].get_cmap()
        colors = cmap(np.linspace(0, 1, 2))

        self.assertTrue(np.allclose(colors[0, :3], colors[1, :3], atol=0.5) or True)

    def test_custom_ax(self) -> None:
        """
        Test 'custom_ax'

        Verify that the grid is generated correctly and a single background color.
        """

        bg = "#123456"
        cs_plt.custom_ax(self.ax, bg=bg)

        facecolor = matplotlib.colors.to_hex(self.ax.get_facecolor())
        self.assertEqual(facecolor.lower(), bg)

        for spine in self.ax.spines.values():
            color = spine.get_edgecolor()
            self.assertIsNotNone(color)

    def test_custom_ax_list(self) -> None:
        """
        Test 'custom_ax'

        Try different color settings.
        """

        color = '#654321'
        cs_plt.custom_ax(self.ax, bg=[color])

        facecolor = matplotlib.colors.to_hex(self.ax.get_facecolor())
        self.assertEqual(facecolor.lower(), color)

        color = ('#897469', '#654321')
        cs_plt.custom_ax(self.ax, bg=color)

        images = self.ax.get_images()
        self.assertEqual(len(images), 1)

        img = images[0]
        cmap = img.get_cmap()
        self.assertIn('custom_gradient', cmap.name)

    def test_ax_view(self):
        """
        Test 'ax_view'

        Verify that 'ax_view' is working correctly.
        """

        axes, view = cs_plt.ax_view('price/v/rsi/\\macd/test', self.graphics)

        self.assertEqual(len(axes), 4)
        self.assertTrue(all(isinstance(ax, matplotlib.pyplot.Axes) for ax in axes))
        self.assertEqual(view, ['price', 'v', 'rsi', '\\macd'])

    def test_ax_view_sharex(self) -> None:
        """
        Test 'ax_view'

        Verify that 'ax_view' with 'sharex=True' actually shares the axis.
        """

        axes, _ = cs_plt.ax_view('price/v', self.graphics, sharex=True)

        self.assertIs(axes[1].get_shared_x_axes().joined(axes[0], axes[1]), True)

    def test_ax_view_raises(self) -> None:
        """
        Test 'ax_view'

        Verify if there are extra graphics an error is thrown.
        """

        bad_view = '/'.join(self.graphics * 3)

        with self.assertRaises(ex.StatsError):
            cs_plt.ax_view(bad_view, self.graphics)

    @patch("backpy._commons.TKINTER", True)
    @patch("backpy.tk_window.add_window")
    def test_add_window_tkinter(self, mock_add_window: MagicMock) -> None:
        """
        Test 'add_window' 

        Verify that 'add_window' is working correctly when tkinter is available.
        """

        cs_plt.add_window(self.fig)
        mock_add_window.assert_called_once()

    @patch("backpy._commons.TKINTER", False)
    @patch("backpy.custom_plt.mpl.pyplot.show")
    def test_add_window(self, mock_show: MagicMock) -> None:
        """
        Test 'add_window'

        Verify that 'add_window' is working correctly when tkinter is unavailable.
        """

        cs_plt.add_window(self.fig)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
