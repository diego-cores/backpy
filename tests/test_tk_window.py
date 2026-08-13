"""
Tkinter window Test
"""

from backpy import exception as ex
from backpy import _commons as _cm

from unittest.mock import patch, MagicMock
import unittest

import matplotlib.pyplot
import matplotlib
import sys

matplotlib.use("Agg")

if "tkinter" not in sys.modules:
    try: import tkinter
    except ImportError:
        sys.modules["tkinter"] = MagicMock()
from backpy import tk_window as tkw

class TestTkWindow(unittest.TestCase):
    def setUp(self) -> None:
        self.fig, self.ax = matplotlib.pyplot.subplots()

        setattr(_cm, '__panel_list', [])
        self._mod__panel_list = []

        self._mod__panel_list.append({
            'fig':self.fig,
            'title':'test',
            'toolbar':'total',
        })
        self._mod__panel_list = self._mod__panel_list * 4

    @patch("backpy.tk_window.CustomWin")
    def test_new_paneledw(self, mock_win:MagicMock) -> None:
        """
        Test 'new_paneledw'

        Verify that the function works correctly.
        """

        mock_win_instance = MagicMock()
        mock_win.return_value = mock_win_instance

        setattr(_cm, '__panel_list', self._mod__panel_list)
        setattr(_cm, '__panel_wmax', 4)

        tkw.new_paneledw(False)
        mock_win.assert_called_once()

    def test_new_paneledw_empty(self) -> None:
        """
        Test 'new_paneledw'

        Verify that the function works correctly when there are no panels.
        """

        tkw.new_paneledw(False)

        with self.assertRaises(ex.CustomWinError):
            tkw.new_paneledw(True)

        setattr(_cm, '__panel_list', self._mod__panel_list * 2)
        setattr(_cm, '__panel_wmax', 4)

        with self.assertRaises(ex.CustomWinError):
            tkw.new_paneledw(False)

    @patch("backpy.tk_window.new_paneledw")
    @patch("backpy.tk_window.mpl.pyplot.close")
    def test_add_window(self, mock_close:MagicMock, mock_new_paneledw:MagicMock) -> None:
        """
        Test 'add_window'

        Verify that the function works correctly.
        """

        setattr(_cm, '__panel_wmax', 4)

        tkw.add_window(self.fig, new=False)
        self.assertEqual(len(getattr(_cm, '__panel_list', [])), 1)
        panel = getattr(_cm, '__panel_list', [{}])[0]
        self.assertIs(panel['fig'], self.fig)

        mock_close.assert_called_once_with(self.fig)
        mock_new_paneledw.assert_called_once()

    @patch("backpy.tk_window.CustomWin")
    @patch("backpy.tk_window.mpl.pyplot.close")
    def test_add_window_true(self, mock_close:MagicMock, mock_win:MagicMock) -> None:
        """
        Test 'add_window'

        Verify that the function works correctly with 'new' = 'True'.
        """

        mock_win_instance = MagicMock()
        mock_win.return_value = mock_win_instance

        tkw.add_window(self.fig, new=True)

        mock_win.assert_called_once()
        mock_win_instance.show.assert_called_once()
        mock_close.assert_called_once_with(self.fig)
