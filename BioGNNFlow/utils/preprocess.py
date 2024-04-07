from typing import Callable

import numpy as _np


def to_numpy(keywords: list[str] = None, position: list[int] = None) -> Callable:
    """
    A decorator for functions, change array-like to numpy arrays.

    Parameters
    ----------
    keywords : list[str], optional
        Keywords to convert to numpy arrays,
        when None, all **kwargs are converted to numpy arrays if ``hasattr(val, "__iter__")``, by default None.
    position : list[int], optional
        Position of the positional args to convert to numpy arrays,
        when None, all *args are converted to numpy arrays if ``hasattr(arg, "__iter__")``, by default None.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if keywords is None:
                kwargs = {
                    key: (
                        _np.array(val)
                        if (hasattr(val, "__iter__") and (not isinstance(val, str)))
                        else val
                    )
                    for key, val in kwargs.items()
                }
            else:
                for key in keywords:
                    kwargs[key] = _np.array(kwargs[key])

            if position is None:
                args = [
                    (
                        _np.array(arg)
                        if (hasattr(arg, "__iter__") and (not isinstance(arg, str)))
                        else arg
                    )
                    for arg in args
                ]
            else:
                args = [_np.array(args[index]) for index in position]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_shape(iterable, str_check=True) -> list[int]:
    """
    Get the shape of an iterable item.

    Parameters
    ----------
    iterable : iterable
        From which to get the shape, following must be true,

        - ``hasattr(iterable, "__iter__")``,
        - ``hasattr(iterable,"__len__")``,
        - ``hasattr(iterable, "__getitem__")``.
    str_check : bool, optional
        Determine whether to check if the iterable is str.

    Returns
    -------
    list[int] : shape of iterable object.
    """
    if isinstance(iterable, str) and str_check:
        raise TypeError("The iterable shall not be a string")

    if not hasattr(iterable, "__iter__") or isinstance(iterable, str):
        return []

    if not hasattr(iterable, "__len__"):
        raise TypeError("Iterators do not support '__len__'")

    if not hasattr(iterable, "__getitem__"):
        raise TypeError(f"Object {type(iterable)} does not support '__getitem__'")

    try:
        next(iter(iterable))
    except StopIteration:
        return [0]  # return [0] if StopIteration is raised

    return [len(iterable)] + get_shape(iterable[0], str_check=False)
