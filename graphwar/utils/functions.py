import itertools
import numpy as np
from collections import namedtuple
from numbers import Number
from typing import Any, Optional

topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])


def topk(array: np.ndarray, k: int, largest: bool = True) -> topk_namedtuple:
    """Returns the k largest/smallest elements and corresponding indices 
    from an array-like input.

    Parameters
    ----------
    array : np.ndarray or list
        the array-like input
    k : int
        the k in "top-k" 
    largest ： bool, optional
        controls whether to return largest or smallest elements        

    Returns
    -------
    namedtuple[values, indices]
        Returns the :attr:`k` largest/smallest elements and corresponding indices 
        of the given :attr:`array`

    Example
    -------
    >>> array = [5, 3, 7, 2, 1]
    >>> topk(array, 2)
    topk_namedtuple(values=array([7, 5]), indices=array([2, 0], dtype=int64))

    >>> topk(array, 2, largest=False)
    topk_namedtuple(values=array([1, 2]), indices=array([4, 3], dtype=int64))

    >>> array = [[1, 2], [3, 4], [5, 6]]
    >>> topk(array, 2)
    topk_namedtuple(values=array([6, 5]), indices=(array([2, 2], dtype=int64), array([1, 0], dtype=int64)))
    """

    array = np.asarray(array)
    flat = array.ravel()

    if largest:
        indices = np.argpartition(flat, -k)[-k:]
        argsort = np.argsort(-flat[indices])
    else:
        indices = np.argpartition(flat, k)[:k]
        argsort = np.argsort(flat[indices])

    indices = indices[argsort]
    values = flat[indices]
    indices = np.unravel_index(indices, array.shape)
    if len(indices) == 1:
        indices, = indices
    return topk_namedtuple(values=values, indices=indices)


def repeat(src: Any, length: Optional[int] = None) -> Any:
    """repeat any objects and return iterable ones.

    Parameters
    ----------
    src : Any
        any objects
    length : Optional[int], optional
        the length to be repeated. If `None`,
        it would return the iterable object itself, by default None

    Returns
    -------
    Any
        the iterable repeated object


    Example
    -------
    >>> from graphwar.utils import repeat
    # repeat for single non-iterable object
    >>> repeat(1)
    [1]
    >>> repeat(1, 3)
    [1, 1, 1]
    >>> repeat('relu', 2)
    ['relu', 'relu']
    >>> repeat(None, 2)
    [None, None]
    # repeat for iterable object
    >>> repeat([1, 2, 3], 2)
    [1, 2]
    >>> repeat([1, 2, 3], 5)
    [1, 2, 3, 3, 3]

    """
    if src == [] or src == ():
        return []
    if length is None:
        length = get_length(src)
    if any((isinstance(src, Number), isinstance(src, str), src is None)):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src


def get_length(obj: Any) -> int:
    if isinstance(obj, (list, tuple)):
        length = len(obj)
    else:
        length = 1
    return length
