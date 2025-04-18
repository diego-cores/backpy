"""
Flex data Module.

This module contains the class that unifies various types of lists and dictionaries.

Classes:
    DataWrapper: Class for storing dataframes, series, ndarrays, lists, and dictionaries in a single type.
"""

from typing import TypeVar, Generic, List, Union, Dict
from collections.abc import ItemsView as dict_items
from collections.abc import MutableSequence
import numpy as np
import pandas as pd

from . import exception

T = TypeVar('T')

class DataWrapper(MutableSequence, Generic[T]):
    """
    DataWrapper.

    Datawrapper unifies pd.dataframe, pd.series, np.ndarray, lists, and dictionaries.

    Attributes:
        data: The stored data in np.ndarray type.

    Private Attributes:
        _index: Pandas index

    Methods:
        insert: Inserts a value into the data list.
        to_dataframe: Returns what is stored in Pandas Dataframe.
        to_series: Returns what is stored in Pandas Series.
        to_dict: Return the value in Python dict format.
        to_list: Return the value in Python list format.
        unwrap: Returns self.data in its np.ndarray format.

    Private Methods:
        __init__: Constructor method.
        __set_convertible: Return the data to list.
        __set_index: Return the data index if it has.
        __valid_index: Return the index if it is correct.
    """

    def __init__(self, data: Union[List[T], Dict[str, List[T]]] = None) -> None:
        """
        Builder method.

        Args:
            data (Union[List[T], Dict[str, List[T]]]): Value to store.
        """

        self.data = self.__set_convertible(data)
        self._index = self.__set_index(data)

        try: 
            super().__init__()
        except TypeError as e:
            raise exception.DataFlexError(f"Error converting data to Numpy array. {e}")

    def __set_convertible(self, data) -> np.ndarray:
        """
        Set convertible.

        Returns 'data' in list type.

        Returns:
            list: 'data' in list type.
        """
        if type(data) is DataWrapper or type(data) is np.ndarray:
            return data
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

        raise exception.DataFlexError("Unsupported data format.")

    def __set_index(self, data) -> np.ndarray:
        """
        Set index.

        Returns the Pandas index if 'data' has one.

        Returns:
            list: Index in list type.
        """

        if type(data) is DataWrapper:
            return data._index
        elif type(data) is pd.DataFrame or type(data) is pd.Series:
            return data.index.to_numpy()

        return None

    def __valid_index(self, flatten:bool = False) -> list:
        """
        Valid index.

        Returns the index if it is suitable.

        Args:
            flatten (bool, optional): The length of self.data.fallten 
                is calculated instead of self.data.

        Returns:
            list: Index in list type.
        """

        return self._index.tolist() if isinstance(self._index, np.ndarray) and len(self._index) == (len(self.data.flatten()) if flatten else len(self.data)) else None

    def insert(self, idx:int, value:any) -> None:
        """
        Insert

        This is like: np.insert.
        Inserts a value into the data list.

        Args:
            idx (bool): Index where it will be inserted.
            value (any): Value to insert.
        """

        self.data = np.insert(self.data, idx, value)

    def unwrap(self) -> np.ndarray:
        """
        Unwrap

        Returns self.data in its np.ndarray format.
        
        Returns:
            np.ndarray: self.data.
        """

        return self.data

    def to_dataframe(self) -> pd.DataFrame:
        """
        To Pandas Dataframe.

        Return the value in pd.DataFrame format

        Returns:
            pd.DataFrame: Data.
        """

        try: 
            return pd.DataFrame(self.data, index=self.__valid_index())
        except ValueError as e: 
            raise exception.ConvFlexError(f"Dataframe conversion error.")

    def to_series(self) -> pd.Series:
        """
        To Pandas Series

        Return the value in pd.Series format.

        Returns:
            pd.Series: Data.
        """

        try: 
            return pd.Series(self.data.flatten(), index=self.__valid_index(True))
        except ValueError as e: 
            raise exception.ConvFlexError(f"Series conversion error.")

    def to_dict(self) -> dict:
        """
        To Python dict

        Return the value in Python dict format.

        Returns:
            dict: Data.
        """

        try:
            if self.data.ndim == 2:
                return {i: list(col) for i, col in enumerate(self.data.T)}
            else:
                return {i: [val] for i, val in enumerate(self.data)}
        except ValueError as e:
            raise exception.ConvFlexError(f"Dict conversion error: {e}")

    def to_list(self) -> list:
        """
        To Python list

        Return the value in Python list format.

        Returns:
            list: Data.
        """

        return self.data.tolist()

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def __getattr__(self, attr):
        return getattr(self.data, attr)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __delitem__(self, idx):
        del self.data[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __str__(self):
        return str(self.data)
