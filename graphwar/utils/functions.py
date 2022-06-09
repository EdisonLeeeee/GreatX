import functools
import inspect
import itertools
from collections import namedtuple
from numbers import Number
from typing import Any, Callable, Optional

import numpy as np

topk_values_indices = namedtuple('topk_values_indices', ['values', 'indices'])


def topk(array: np.ndarray, k: int, largest: bool = True) -> topk_values_indices:
    """Returns the k largest/smallest elements and corresponding indices 
    from an array-like input.

    Parameters
    ----------
    array : np.ndarray or list
        the array-like input
    k : int
        the k in "top-k" 
    largest : bool, optional
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
    topk_values_indices(values=array([7, 5]), indices=array([2, 0], dtype=int64))

    >>> topk(array, 2, largest=False)
    topk_values_indices(values=array([1, 2]), indices=array([4, 3], dtype=int64))

    >>> array = [[1, 2], [3, 4], [5, 6]]
    >>> topk(array, 2)
    topk_values_indices(values=array([6, 5]), indices=(array([2, 2], dtype=int64), array([1, 0], dtype=int64)))
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
    return topk_values_indices(values=values, indices=indices)


def repeat(src: Any, length: Optional[int] = None) -> Any:
    """Repeat any objects and return iterable ones.

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

    >>> # repeat for iterable object
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


def wrapper(func: Callable) -> Callable:
    """Wrap a function to make some arguments 
    have the same length. By default, the arguments
    to be modified are `hids` and `acts`. 

    Uses can custom these arguments by setting argument 

    * `includes` : to includes custom arguments
    * `excludes` : to excludes custom arguments
    * `length_as` : to make the length of the arguments the same as `length_as`, by default, it is `hids`.

    Parameters
    ----------
    func : Callable
        a function to be wrapped.

    Returns
    -------
    Callable
        a wrapped function.

    Raises
    ------
    TypeError
        if the required arguments of the function is missing.


    Example
    -------

    >>> @wrapper
    ... def func(hids=[16], acts=None):
    ...     print(locals())

    >>> func(100)
    {'hids': [100], 'acts': [None]}

    >>> func([100, 64])
    {'hids': [100, 64], 'acts': [None, None]}

    >>> func([100, 64], excludes=['acts'])
    {'hids': [100, 64], 'acts': None}


    >>> @wrapper
    ... def func(self, hids=[16], acts=None):
    ...     print(locals())    

    >>> func()
    TypeError: The decorated function 'func' missing required argument 'self'.

    >>> func('class_itself')
    {'self': 'class_itself', 'hids': [16], 'acts': [None]}

    >>> func('class_itself', hids=[])
    {'self': 'class_itself', 'hids': [], 'acts': []}


    >>> @wrapper
    ... def func(self, hids=[16], acts=None, heads=8):
    ...     print(locals())    

    >>> func('class_itself', hids=[100, 200])
    {'self': 'class_itself', 'hids': [100, 200], 'acts': [None, None], 'heads': 8}

    >>> func('class_itself', hids=[100, 200], includes=['heads'])
    {'self': 'class_itself', 'hids': [100, 200], 'acts': [None, None], 'heads': [8, 8]}

    """

    @functools.wraps(func)
    def decorate(*args, **kwargs) -> Any:
        inspect_paras = inspect.signature(func).parameters
        inspect_paras = list(inspect_paras.values())

        paras = {}
        unspecified = []
        i = 0
        max_length = len(args)
        for p in inspect_paras:
            if p.kind == inspect._ParameterKind.VAR_KEYWORD:
                # arguments like `**kwargs`
                continue
            if i < max_length:
                paras[p.name] = args[i]
                i += 1
                continue
            if p.default == inspect._empty:
                if p.name in kwargs:
                    paras[p.name] = kwargs[p.name]
                    continue

                if i >= max_length:
                    raise TypeError(
                        f"The decorated function '{func.__name__}' missing required argument '{p.name}'.")
            else:
                paras[p.name] = p.default

        for k, v in kwargs.items():
            paras[k] = v

        includes = paras.get("includes", [])
        excludes = paras.get("excludes", [])
        length_as = paras.get("length_as", "hids")

        assert isinstance(includes, list)
        assert isinstance(excludes, list)
        assert isinstance(length_as, str)

        accepted_vars = includes + ['hids', 'acts']
        accepted_vars = list(set(accepted_vars) - set(excludes))

        assert length_as in accepted_vars

        repeated = get_length(paras.get(length_as, 0))
        for var in accepted_vars:
            if var in paras:
                val = paras[var]
                paras[var] = repeat(val, repeated)

        paras.pop('includes', None)
        paras.pop('excludes', None)
        paras.pop('length_as', None)

        return func(**paras)

    return decorate
