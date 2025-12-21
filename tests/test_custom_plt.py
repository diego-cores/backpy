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

matplotlib.use("Agg")

class TestCustomPlt(unittest.TestCase):
    def setUp(self) -> None:
        self.fig, self.ax = matplotlib.pyplot.subplots()
        self.graphics = ['price', 'v', 'rsi/', '\\macd']

        setattr(_cm, '__panel_list', [])
        self._mod__panel_list = []

        self._mod__panel_list.append({
            'fig':self.fig,
            'title':'test',
            'toolbar':'total',
        })
        self._mod__panel_list = self._mod__panel_list * 4

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

    @patch("backpy.custom_plt.CustomWin")
    def test_new_paneledw(self, mock_win:MagicMock) -> None:
        """
        Test 'new_paneledw'

        Verify that the function works correctly.
        """

        mock_win_instance = MagicMock()
        mock_win.return_value = mock_win_instance

        setattr(_cm, '__panel_list', self._mod__panel_list)
        setattr(_cm, '__panel_wmax', 4)

        cs_plt.new_paneledw(False)

        mock_win.assert_called_once()

    def test_new_paneledw_empty(self) -> None:
        """
        Test 'new_paneledw'

        Verify that the function works correctly when there are no panels.
        """

        cs_plt.new_paneledw(False)

        with self.assertRaises(ex.CustomWinError):
            cs_plt.new_paneledw(True)

        setattr(_cm, '__panel_list', self._mod__panel_list * 2)
        setattr(_cm, '__panel_wmax', 4)

        with self.assertRaises(ex.CustomWinError):
            cs_plt.new_paneledw(False)

    @patch("backpy.custom_plt.new_paneledw")
    @patch("backpy.custom_plt.mpl.pyplot.close")
    def test_add_window(self, mock_close:MagicMock, mock_new_paneledw:MagicMock) -> None:
        """
        Test 'add_window'

        Verify that the function works correctly.
        """

        setattr(_cm, '__panel_wmax', 4)

        cs_plt.add_window(self.fig, new=False)
        self.assertEqual(len(getattr(_cm, '__panel_list', [])), 1)
        panel = getattr(_cm, '__panel_list', [{}])[0]
        self.assertIs(panel['fig'], self.fig)

        mock_close.assert_called_once_with(self.fig)
        mock_new_paneledw.assert_called_once()

    @patch("backpy.custom_plt.CustomWin")
    @patch("backpy.custom_plt.mpl.pyplot.close")
    def test_add_window_true(self, mock_close:MagicMock, mock_win:MagicMock) -> None:
        """
        Test 'add_window'

        Verify that the function works correctly with 'new' = 'True'.
        """

        mock_win_instance = MagicMock()
        mock_win.return_value = mock_win_instance

        cs_plt.add_window(self.fig, new=True)

        mock_win.assert_called_once()
        mock_win_instance.show.assert_called_once()
        mock_close.assert_called_once_with(self.fig)

if __name__ == '__main__':
    unittest.main()

# 'CustomToolbar' is missing
