"""
Flex data Module.

This module contains the class that unifies various types of lists and dictionaries.

Classes:
    DataWrapper: Class for storing dataframes, series, ndarrays, lists, and dictionaries in a single type.
"""

from typing import TypeVar, Generic, List, Union, Dict
from collections import UserList
import numpy as np
import pandas as pd

from . import exception

T = TypeVar('T')

class DataWrapper(UserList, Generic[T]):
    """
    DataWrapper.

    Datawrapper unifies pd.dataframe, pd.series, np.ndarray, lists, and dictionaries.

    Attributes:
        value: The stored data in list type.

    Private Attributes:
        _index: Pandas index
        __init_type: Saves the data type before converting to DataWrapper.

    Methods:
        to_dataframe: Returns what is stored in Pandas Dataframe.
        to_series: Returns what is stored in Pandas Series.
        to_ndarray: Returns what is stored in Numpy Array.

    Private Methods:
        __init__: Constructor method.
        __set_convertible: Return the data to list.
        __set_index: Return the data index if it has.
        __set_initialtype: Return the value type list or dict.
        __valid_index: Return the index if it is correct.
    """

    def __init__(self, data: Union[List[T], Dict[str, List[T]]] = None) -> None:
        """
        Builder method.

        Args:
            data (Union[List[T], Dict[str, List[T]]]): Value to store.
        """

        self._index = self.__set_index(data)
        value = self.__set_convertible(data)
        self.__init_type = self.__set_initialtype(value)

        try: super().__init__(value)
        except TypeError as e:
            raise exception.DataFlexError(f"Error converting data to list. {e}")

    def __set_convertible(self, data) -> list:
        """
        Set convertible.

        Returns 'data' in list type.

        Returns:
            list: 'data' in list type.
        """

        if isinstance(data, DataWrapper):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient='list').items()
        elif isinstance(data, pd.Series):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return data.items()
        
        return data
    
    def __set_index(self, data) -> list:
        """
        Set index.

        Returns the Pandas index if 'data' has one.

        Returns:
            list: Index in list type.
        """

        if isinstance(data, DataWrapper):
            return data._index
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return list(data.index)
        
        return None

    def __set_initialtype(self, value) -> type:
        """
        Set initialtype.

        Returns the main type of 'value': list or dictionary.

        Returns:
            type: Main type.
        """

        if isinstance(value, DataWrapper):
            return value._DataWrapper__init_type
        elif isinstance(value, type({1: 1}.items())):
            return dict
        
        return type(value)

    def __valid_index(self, lst:bool = False) -> list:
        """
        Valid index.

        Returns the index if it is suitable.

        Args:
            lst (bool, optional)

        Returns:
            list: Index in list type.
        """

        if lst: 
            return self._index if self._index and len(self._index) == len(self.data) else None

        if (self.__init_type == dict 
              and all(len(v) == 2 for v in self.data) 
              and self._index 
              and max(len(value[1]) 
                      if isinstance(value[1], list) 
                      else 1 for value in self.data) == len(self._index)):
                return self._index
        
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """
        To Pandas Dataframe.

        Return the value in pd.DataFrame format

        Returns:
            pd.DataFrame: Data.
        """

        if self.__init_type == dict and all(len(v) == 2 for v in self.data):

            return pd.DataFrame({key: value for key, value in self.data},
                                index=self.__valid_index())
        else:
            try: return pd.DataFrame(self.data, index=self.__valid_index(True))
            except ValueError as e: 
                raise exception.ConvFlexError(f"Dataframe conversion error.")

    def to_series(self) -> pd.Series:
        """
        To Pandas Series

        Return the value in pd.Series format.

        Returns:
            pd.Series: Data.
        """
        try: return pd.Series(self.data, index=self.__valid_index(True))
        except ValueError as e: 
            raise exception.ConvFlexError(f"Series conversion error.")
    
    def to_ndarray(self) -> np.ndarray:
        """
        To ndarray

        Return the value in np.ndarray format.

        Returns:
            np.ndarray: Data.
        """
        try: return np.array(self.data, dtype=object)
        except ValueError as e:
            raise exception.ConvFlexError(f"ndarray conversion error.")
