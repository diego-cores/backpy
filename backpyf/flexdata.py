"""
Flex data Module.

This module has types and classes that BackPy uses.

Classes:
    DataWrapper: Class for storing dataframes, series, ndarrays, lists, and dictionaries in a single type.
    CostsValue: Class to calculate different commissions, spreads, etc. 
        depending on the user's input and whether they are a maker or taker.
"""

from typing import TypeVar, Generic, List, Union, Dict, Tuple
from collections.abc import ItemsView as dict_items
from collections.abc import MutableSequence
import pandas as pd
import random as rd
import numpy as np

from . import exception

T = TypeVar('T')

class DataWrapper(MutableSequence, Generic[T]):
    """
    DataWrapper.

    Datawrapper unifies pd.dataframe, pd.series, np.ndarray, lists, and dictionaries.

    Private Attributes:
        _data: The stored data in np.ndarray type.
        _index: Pandas index
        _columns: Name of columns.

    Methods:
        insert: Inserts a value into the data list.
        to_dataframe: Returns what is stored in Pandas Dataframe.
        to_series: Returns what is stored in Pandas Series.
        to_dict: Return the value in Python dict format.
        to_list: Return the value in Python list format.
        unwrap: Returns self._data in its np.ndarray format.

    Private Methods:
        __init__: Constructor method.
        __set_convertible: Return the data to list.
        __set_index: Return the data index if it has.
        __get_columns: Convert 'columns' to np.ndarray.
        __set_columns: Returns the names of the columns in 'data' if it has.
        __valid_index: Return the index if it is correct.
        __valid_columns: Returns the correct column names.
    """

    def __init__(self, data: Union[List[T], Dict[str, List[T]]] = None, 
                 columns:np.ndarray = None) -> None:
        """
        Builder method.

        Args:
            data (Union[List[T], Dict[str, List[T]]]): Value to store.
            columns (np.ndarray, optional): Column names.
        """
        if type(columns) != np.ndarray and not columns is None:
            columns = self.__get_columns(columns)

        self._data = self.__set_convertible(data)
        self._index = self.__set_index(data)
        self._columns = (self.__set_columns(data) if columns is None 
                         else columns)

        super().__init__()

    def __get_columns(self, columns) -> np.ndarray:
        """
        Get columns.

        This function converts its 'columns' argument to np.ndarray.

        Returns:
            list: 'columns' in np.ndarray type.
        """

        if type(columns) is DataWrapper:
            return columns._columns
        elif type(columns) is pd.Index:
            return columns.to_numpy()
        else:
            return np.array(columns, ndmin=1)

    def __set_convertible(self, data) -> np.ndarray:
        """
        Set convertible.

        Returns 'data' in list type.

        Returns:
            list: 'data' in list type.
        """
        if data is None:
            return np.array([])
        elif type(data) is DataWrapper:
            return data.unwrap()
        elif type(data) is list:
            return np.array(data)
        elif type(data) is dict:
            return np.array(list(data.values())).T

        match type(data):
            case pd.DataFrame:
                return data.values
            case pd.Series:
                return data.to_numpy()
            case pd.Index:
                return data.to_numpy()
            case np.ndarray:
                return data

        return data

    def __set_index(self, data) -> np.ndarray:
        """
        Set index.

        Returns the Pandas index if 'data' has one.

        Returns:
            list: Index in np.ndarray type.
        """

        if type(data) is DataWrapper:
            return data._index
        elif type(data) is pd.DataFrame or type(data) is pd.Series:
            return data.index.to_numpy()

        return None

    def __set_columns(self, data) -> np.ndarray:
        """
        Set index.

        Returns the names of columns if 'data' has columns.

        Returns:
            list: Columns in np.ndarray type.
        """

        if type(data) is DataWrapper:
            return data._columns
        elif type(data) is pd.DataFrame:
            return data.columns.to_numpy()
        elif type(data) is dict:
            return np.array(list(data.keys()))

        return None

    def __valid_index(self, flatten:bool = False) -> list:
        """
        Valid index.

        Returns the index if it is suitable.

        Args:
            flatten (bool, optional): The length of self._data.flatten 
                is calculated instead of self._data.

        Returns:
            list: Index in list type.
        """

        return (self._index.tolist() 
                if isinstance(self._index, np.ndarray) 
                and len(self._index) == (len(self._data.flatten()) 
                                         if flatten else len(self._data)) 
                else None)

    def __valid_columns(self) -> list:
        """
        Valid columns.

        Returns the correct column names.

        Returns:
            list: Columns in list type.
        """
        n_cols = (self._data.shape[1] if self._data.ndim == 2 
                  else 1)

        return (self._columns.tolist() 
                if not self._columns is None
                    and n_cols == len(self._columns) 
                else list(range(n_cols)))

    def insert(self, idx:int, value:any) -> None:
        """
        Insert

        This is like: np.insert.
        Inserts a value into the data list.

        Args:
            idx (bool): Index where it will be inserted.
            value (any): Value to insert.
        """

        self._data = np.insert(self._data, idx, value)

    def unwrap(self) -> np.ndarray:
        """
        Unwrap

        Returns self._data in its np.ndarray format.
        
        Returns:
            np.ndarray: self._data.
        """

        return self._data

    def to_dataframe(self) -> pd.DataFrame:
        """
        To Pandas Dataframe.

        Return the value in pd.DataFrame format

        Returns:
            pd.DataFrame: Data.
        """

        try: 
            return pd.DataFrame(self._data, index=self.__valid_index(), 
                                columns=self.__valid_columns())
        except ValueError as e: 
            raise exception.ConvWrapperError(f"Dataframe conversion error.")

    def to_series(self) -> pd.Series:
        """
        To Pandas Series

        Return the value in pd.Series format.

        Returns:
            pd.Series: Data.
        """

        try: 
            return pd.Series(self._data.flatten(), index=self.__valid_index(True))
        except ValueError as e: 
            raise exception.ConvWrapperError(f"Series conversion error.")

    def to_dict(self) -> dict:
        """
        To Python dict

        Return the value in Python dict format.

        Returns:
            dict: Data.
        """

        try:
            if self._data.ndim == 2:
                columns = self.__valid_columns()
                return {columns[i]: list(self._data.T[i]) 
                            for i in range(len(self._data.T))}
            else:
                return {i: [val] for i, val in enumerate(self._data)}
        except ValueError as e:
            raise exception.ConvWrapperError(f"Dict conversion error: {e}")

    def to_list(self) -> list:
        """
        To Python list

        Return the value in Python list format.

        Returns:
            list: Data.
        """

        return self._data.tolist()

    def __getattr__(self, name):
        attr = getattr(self._data, name, None)
        if callable(attr):
            def wrapper(*args, **kwargs):
                try:
                    result = attr(*args, **kwargs)

                    return DataWrapper(result) if isinstance(result, np.ndarray) else result
                except Exception as e:
                    raise exception.ConvWrapperError(
                        f"Error when calling '{name}': {e}")
            return wrapper
        elif attr is not None:
            return attr
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    def __delitem__(self, idx):
        del self._data[idx]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return str(self._data)

    def __add__(self, other):
        return DataWrapper(
            self._data + (other.unwrap() if isinstance(other, DataWrapper) else other))

    def __sub__(self, other):
        return DataWrapper(
            self._data - (other.unwrap() if isinstance(other, DataWrapper) else other))

    def __mul__(self, other):
        return DataWrapper(
            self._data * (other.unwrap() if isinstance(other, DataWrapper) else other))

    def __truediv__(self, other):
        return DataWrapper(
            self._data / (other.unwrap() if isinstance(other, DataWrapper) else other))

class CostsValue:
    """
    Costs value.

    This class measures user input to give different values between maker and taker.

    Format:
        (maker, taker) may have an additional tuple indicating 
        that it may be a random number between two numbers.

    Private Attributes:
        _value: Given value.
        _error: Custom message displayed at the end of an error.
        _rand_supp: Whether random values can be generated or not.
        __maker: function that returns maker value.  
        __taker: function that returns taker value. 

    Methods:
        get_maker: Return '__maker()'.
        get_taker: Return '__taker()'.

    Private Methods:
        __init__: Constructor method.
        __process_value: Returns the random or fixed value.
    """

    def __init__(self, value: Union[
        float, Tuple[float, float], 
        Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]]], 
        supp_random:bool = True, supp_double:bool = False,
        cust_error:str = None) -> None:
        """
        Builder method.

        Args:
            value: Data tuple with this format: (maker, taker).
            supp_random (bool, optional): If it supports random values.
            supp_double (bool, optional): False if there is only one side (maker, taker).
            cust_error (str, optional): If an error occurs, 
                you can add custom text at the end of the error.
        """

        self._value = value
        self._rand_supp = supp_random
        self._error = ' ' + (cust_error or '')

        if isinstance(value, tuple):
            if (
                (len(value) == 1 or (len(value) == 2 and supp_random)) 
                and not supp_double
                ):
                self.__taker = self.__maker = self.__process_value(value)
            elif len(value) == 2 and supp_double:
                self.__maker = self.__process_value(value[0])
                self.__taker = self.__process_value(value[1])
            else:
                raise exception.CostValueError(
                    f"Tuple must have 1 or 2 elements: (maker, taker).{self._error}")
        else:
            self.__maker = self.__taker = self.__process_value(value)

    def __process_value(self, val) -> callable:
        """
        Process value.

        This function evaluates 'val' to determine 
        if it matches a 'random.uniform' or returns the fixed value.

        Args:
            val: Value to evaluate.

        Return:
            callable: The function that will return the random or fixed value.
        """

        if isinstance(val, tuple) and len(val) == 2 and self._rand_supp:
            if min(*val) < 0:
                raise exception.CostValueError(
                    f"No value can be less than 0.{self._error}")

            return lambda: rd.uniform(*val)
        elif isinstance(val, (int, float)):
            if val < 0:
                raise exception.CostValueError(
                    f"No value can be less than 0.{self._error}")

            return lambda: val
        else:
            raise exception.CostValueError(
                f"Invalid value format.{self._error}")

    def get_maker(self) -> float:
        """
        Get maker.

        Return:
            float: '__maker()'.
        """

        return self.__maker()

    def get_taker(self) -> float:
        """
        Get taker.
        
        Return:
            float: '__taker()'.
        """

        return self.__taker()
