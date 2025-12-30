"""
Flex data module

This module has types and classes that BackPy uses.

Variables:
    logger (Logger): Logger variable.

Classes:
    DataWrapper: Class for storing dataframes, series, ndarrays, lists, and dictionaries in a single type.
    CostsValue: Class to calculate different commissions, spreads, etc. 
        depending on the user's input and whether they are a maker or taker.
    ChunkWrapper: Class to preload data into your ndarray in chunks and save attributes.
"""

from __future__ import annotations
from typing import Self, Any, cast, overload, Callable
import logging

from pandas.api.extensions import ExtensionArray
from pandas._typing import SequenceNotStr

import pandas as pd
import random as rd
import numpy as np

from . import exception
from . import utils

logger:logging.Logger = logging.getLogger(__name__)

class DataWrapper():
    """
    Data wrapper.

    Datawrapper unifies dataframe, series, ndarray, lists, and dictionaries.

    Private Attributes:
        _data: The stored data in np.ndarray type.
        _index: Pandas index
        _columns: Name of columns.
        _alert: Alert about performance.

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

    _columns:np.ndarray | range | None
    _index:np.ndarray | range | None

    def __init__(
            self, data: (list | dict[str, list] | ExtensionArray | np.ndarray | pd.DataFrame
                          | pd.Series | pd.Index | DataWrapper | None) = None, 
            columns: np.ndarray | pd.Index | range | list | tuple | "DataWrapper" | None = None,
            index: np.ndarray | pd.Index | range | list | tuple | "DataWrapper" | None = None,
            alert: bool = False
        ) -> None:
        """
        Builder method.

        Args:
            data (list | dict[str, list] | ExtensionArray | ndarray | DataFrame 
                | Series | Index | DataWrapper | None, optional): Value to store.
            columns (ndarray | Index | range | list | tuple | 
                "DataWrapper" | None, optional): Column names.
            index (ndarray | Index | list | tuple | 
                "DataWrapper" | None, optional): Data index.
            alert (bool): Alert about performance when running conversions that may take a long time.
        """

        columns_nd = None if columns is None else self.__get_columns(columns)            

        if not index is None:
            index = self.__get_columns(index, index=True)

        self._data = self.__set_convertible(data)
        self._index = (self.__set_index(data) if index is None else index)
        self._columns = (self.__set_columns(data) 
                        if columns_nd is None else columns_nd)
        self._alert = bool(alert or False)

        super().__init__()

    def __get_columns(self, 
            columns:(np.ndarray | pd.Index | range
                | list | tuple | "DataWrapper" | None), 
            index:bool = False) -> np.ndarray|range|None:
        """
        Get columns.

        This function converts its 'columns' argument to ndarray or range.

        Args:
            columns (ndarray | Index 
                | list | tuple | "DataWrapper" | None, optional): 
                Columns to convert.
            index (bool, optional): Instead of searching for 
                the '_columns' attribute in 'DataWrapper' search for '_index'.

        Returns:
            ndarray|range|None: 'columns' in ndarray type.
        """

        if isinstance(columns, (np.ndarray, range)):
            return columns
        elif isinstance(columns, DataWrapper):
            if index: return columns._index

            return columns._columns
        elif isinstance(columns, pd.Index):
            return columns.to_numpy()
        else:
            return np.array(columns, ndmin=1)

    def __set_convertible(self, data:list | dict[str, list] | ExtensionArray | np.ndarray | pd.DataFrame 
                | pd.Series | pd.Index | "DataWrapper" | None) -> np.ndarray:
        """
        Set convertible.

        Returns 'data' in ndarray type.

        Args:
            data (list | dict[str, list] | np.ndarray | pd.DataFrame 
                | pd.Series | pd.Index | DataWrapper | None, optional): Value to store.

        Returns:
            ndarray: 'data' in ndarray type.
        """
        if data is None:
            return np.array([])
        elif isinstance(data, DataWrapper):
            return data.unwrap()
        elif isinstance(data, list):
            return np.array(data, dtype=object)
        elif isinstance(data, dict):
            return np.array(list(data.values()), 
                            dtype=object).T
        elif isinstance(data, pd.DataFrame):
            return data.to_records(index=False)
        elif isinstance(data, (pd.Series, pd.Index)):
            return np.asarray(data.values)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, ExtensionArray):
            return np.asarray(data)

        return data

    def __set_index(self, data:list | dict[str, list] | ExtensionArray | np.ndarray | pd.DataFrame 
                | pd.Series | pd.Index | "DataWrapper" | None) -> range|np.ndarray|None:
        """
        Set index.

        Returns the Pandas index if 'data' has one.

        Args:
            data (list | dict[str, list] | ExtensionArray | np.ndarray | DataFrame 
                | Series | Index | DataWrapper | None, optional): Value with index.

        Returns:
            range|np.ndarray|None: Index in ndarray or range type.
        """

        if type(data) is DataWrapper:
            return data._index
        elif type(data) is pd.DataFrame or type(data) is pd.Series:
            return data.index.to_numpy()
        elif type(data) is np.ndarray or type(data) is ExtensionArray:
            return range(len(data))

        return None

    def __set_columns(self, data:list | dict[str, list] | ExtensionArray | np.ndarray | pd.DataFrame 
                | pd.Series | pd.Index | "DataWrapper" | None) -> np.ndarray|range|None:
        """
        Set index.

        Returns the names of columns if 'data' has columns.

        Args:
            data (list | dict[str, list] | ExtensionArray | np.ndarray | DataFrame 
                | Series | Index | DataWrapper | None, optional): Value with columns.

        Returns:
            ndarray|range|None: Columns in ndarray type.
        """

        if type(data) is DataWrapper:
            return data._columns
        elif type(data) is pd.DataFrame:
            return data.columns.to_numpy()
        elif type(data) is dict:
            return np.array(list(data.keys()))

        return None

    def __valid_index(self, flatten:bool = False) -> None|list:
        """
        Valid index.

        Returns the index if it is suitable.

        Args:
            flatten (bool, optional): The length of self._data.flatten 
                is calculated instead of self._data.

        Returns:
            None|list: Index in list type.
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
                    and not isinstance(self._columns, range)
                    and n_cols == len(self._columns) 
                else list(range(n_cols)))

    def insert(self, index:int, value:Any) -> None:
        """
        Insert

        This is like: np.insert.
        Inserts a value into the data list.

        Args:
            index (bool): Index where it will be inserted.
            value (Any): Value to insert.
        """

        self._data = np.insert(self._data, index, value)

    def unwrap(self) -> np.ndarray:
        """
        Unwrap

        Returns _data in its ndarray format.
        
        Returns:
            ndarray: _data.
        """

        return self._data

    def to_dataframe(self) -> pd.DataFrame:
        """
        To Pandas Dataframe.

        Return the value in DataFrame format

        Returns:
            DataFrame: Data.
        """

        if self._alert:
            logger.warning(utils.text_fix("""
            Converting data types takes a long time; 
            if you're looking for good performance, 
            you should use the default type: 'np.ndarray'.
            """, newline_exclude=False))
        try: 
            if (hasattr(self._data.dtype, "names") 
                and self._data.dtype.names is not None):

                return pd.DataFrame(self._data, index=cast(
                                        SequenceNotStr[int], 
                                        self.__valid_index() or []))
            else:
                return pd.DataFrame(self._data, index=cast(
                                        SequenceNotStr[int], 
                                        self.__valid_index() or []), 
                                    columns=cast(SequenceNotStr, self.__valid_columns()))
        except ValueError as e:
            raise exception.ConvWrapperError(f"Dataframe conversion error.")

    def to_series(self) -> pd.Series:
        """
        To Pandas Series

        Return the value in Series format.

        Returns:
            Series: Data.
        """

        if self._alert:
            logger.warning(utils.text_fix("""
            Converting data types takes a long time; 
            if you're looking for good performance, 
            you should use the default type: 'np.ndarray'.
            """, newline_exclude=False))
        try: 
            return pd.Series(self._data.flatten(), 
                             index=self.__valid_index(True))
        except ValueError as e: 
            raise exception.ConvWrapperError(f"Series conversion error.")

    def to_dict(self) -> dict:
        """
        To Python dict

        Return the value in Python dict format.

        Returns:
            dict: Data.
        """

        if self._alert:
            logger.warning(utils.text_fix("""
            Converting data types takes a long time; 
            if you're looking for good performance, 
            you should use the default type: 'np.ndarray'.
            """, newline_exclude=False))
        try:
            if self._data.ndim == 2:
                columns = self.__valid_columns()
                return {columns[i]: list(self._data.T[i]) 
                            for i in range(len(self._data.T))}
            else:
                return {i: [val] for i, val in enumerate(self._data)}
        except ValueError as e:
            raise exception.ConvWrapperError(f"Dict conversion error.")

    def to_list(self) -> list:
        """
        To Python list

        Return the _data in Python list format.

        Returns:
            list: Data.
        """

        if self._alert:
            logger.warning(utils.text_fix("""
            Converting data types takes a long time; 
            if you're looking for good performance, 
            you should use the default type: 'np.ndarray'.
            """, newline_exclude=False))
        return self._data.tolist()

    def __getattr__(self, name):
        attr:Any|None = getattr(self._data, name, None)
        if callable(attr):
            c_attr = attr
            def wrapper(*args, **kwargs):
                try:
                    result = c_attr(*args, **kwargs)

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

    @overload
    def __getitem__(self, index: int | slice) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: str) -> np.ndarray: ...

    def __getitem__(self, index:str|int|slice) -> np.ndarray:
        if len(self._data) == 0: return self._data

        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __len__(self):
        if (hasattr(self._data.dtype, "names") 
            and self._data.dtype.names is not None):
            return self._data.size

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

    def __init__(
            self, value: (float | tuple[float, float] | tuple[float 
            | tuple[float, float], float | tuple[float, float]]), 
            supp_random:bool = True, supp_double:bool = False, 
            cust_error:str | None = None
        ) -> None:
        """
        Builder method.

        Args:
            value (float | tuple[float, float] | tuple[float | tuple[float, float], float | tuple[float, float]]): 
                Data tuple with this format: (maker, taker).
            supp_random (bool, optional): If it supports random values.
            supp_double (bool, optional): False if there is only one side (maker, taker).
            cust_error (str | None, optional): If an error occurs, 
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

    def __process_value(self, val: tuple | int | float) -> Callable:
        """
        Process value.

        This function evaluates 'val' to determine 
        if it matches a 'random.uniform' or returns the fixed value.

        Args:
            val (tuple | int | float): Value to evaluate.

        Returns:
            Callable: The function that will return the random or fixed value.
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

        Returns:
            float: Maker value.
        """

        return self.__maker()

    def get_taker(self) -> float:
        """
        Get taker.
        
        Returns:
            float: Taker value.
        """

        return self.__taker()

class ChunkWrapper(np.ndarray):
    """
    Chunk wrapper.

    Preload memory space into your ndarray and save attributes.

    Private Attributes:
        _pos: Index where data will be added.
        _chunk_size: Size of each chunk.

    Methods:
        get: Get value from your structured array with default.
        delete: Removes an index from the array.
        values: Returns the filled data.
        append: Add data and load chunks if needed.
        chunk_loader: Load space to save more data.

    Private Methods:
        __new__: Constructor method.
        __rewrap: Wraps an ndarray with this instance.
    """

    _chunk_size:int
    _pos:int

    def __new__(cls, data:Any = np.array([]), 
                dtype:str | type | np.dtype | None = None,
                chunk_size: int | None = None) -> Self:
        """
        Constructor method.

        Args:
            data (Any): Value to store.
            dtype (str | type | dtype | None, optional): 
                Numpy dtype, None sets the dtype already set.
            chunk_size (int, optional): Size of each chunk.
        """

        # exceptions
        if chunk_size and (not isinstance(chunk_size, int) or chunk_size <= 0):
            raise exception.ChunkWrapperError(
                "'chunk_size' can only be 'int' greater than 0.")

        obj = np.asarray(data, dtype=dtype).view(cls)
        obj._pos = len(data)
        obj._chunk_size = chunk_size or 10_000

        return obj

    def __rewrap(self, new:np.ndarray) -> ChunkWrapper:
        """
        Rewrap

        Wraps an ndarray with this instance.

        Args:
            new (ndarray): ndarray to rewrap.

        Returns:
            ChunkWrapper: Rewrapped array. 
        """

        new_data = new.view(ChunkWrapper)

        new_data._pos = self._pos
        new_data._chunk_size = self._chunk_size

        return new_data

    def chunk_loader(
            self, dtype:str | type | tuple | list | np.dtype | None = None) -> ChunkWrapper:
        """
        Chunk loader

        Load space to save more data.

        Args:
            dtype (str | type | tuple | list | dtype | None, optional): 
                Numpy dtype, None sets the dtype already set.

        Returns:
            ChunkWrapper: Returns the ChunkWrapper with the new spaces.
        """

        if dtype is None:
            dtype = self.dtype

        setattr(self, 'dtype', dtype)
        new_data = np.concat(
            [self, np.empty(self._chunk_size, dtype=dtype)]
        , dtype=dtype)

        return self.__rewrap(new_data)

    def append(self, data:np.ndarray, 
                dtype:str | type | tuple | list | np.dtype | None = None) -> ChunkWrapper:
        """
        Append

        Add data and load chunks if needed.

        Args:
            data (ndarray): Data to add.
            dtype (str | type | tuple | list | dtype | None, optional): 
                Numpy dtype, None sets the dtype already set.

        Returns:
            ChunkWrapper: Returns the ChunkWrapper with the new data.
        """

        if self._pos + len(data) > len(self):
            new_val = self.chunk_loader(dtype=dtype)
        else:
            new_val = self

        new_val[new_val._pos:new_val._pos + len(data)] = data

        new_val._pos += len(data)

        return new_val

    def values(self) -> np.ndarray:
        """
        Values

        Returns the filled data, equivalent to 'self[:self._pos]'
        
        Returns:
            ndarray: The array without the empty data.
        """

        return self[:self._pos]

    def get(self, name:str, 
            default:Any | None = None) -> type | np.ndarray | None:
        """
        Get

        Get a column from your structured array 
            with a default if it doesn't exist.

        Args:
            name (str): Column name.
            default (Any | None): Default value.

        Returns:
            type|ndarray|None: Value.
        """
        
        if not hasattr(self.dtype, "names") or self.dtype.names is None:
            raise exception.ChunkWrapperError(
                'Only support structured ndarray.')

        if name in self.dtype.names:
            return self[name]
        else:
            return default

    def delete(self, index:int) -> None:
        """
        Delete

        Removes an element from the array.
            Method inplace.

        Args:
            index (int): Index to delete.

        Note:
            This function messes up the array because to 'eliminate' 
            the value it is replaced by the last one.
        """

        self._pos -= 1

        self[index] = self[self._pos] 
