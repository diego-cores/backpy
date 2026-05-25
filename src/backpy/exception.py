"""
Exception module

Custom exceptions.
"""

class BackpyError(Exception): pass

class DataError(BackpyError):pass

class YfinanceError(BackpyError):pass

class BinanceError(BackpyError):pass

class PlotError(BackpyError):pass

class StatsError(BackpyError):pass

class RunError(BackpyError):pass

class StyleError(BackpyError):pass

class OrderError(BackpyError):pass

class CostValueError(BackpyError):pass

class CustomWinError(BackpyError):pass
