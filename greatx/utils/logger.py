import functools
import logging
import os
import sys
from typing import Optional

try:
    # For colored outputs when printing
    from termcolor import colored
except ImportError:
    colored = None

__all__ = ["setup_logger", "get_logger"]


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def setup_logger(output: Optional[str] = None, name: str = "GreatX", *,
                 distributed_rank: int = 0, mode: str = 'w',
                 color: bool = True,
                 abbrev_name: Optional[str] = None) -> logging.Logger:
    """Initialize the GreatX logger and set its verbosity level to "DEBUG".

    Parameters
    ----------
    output : Optional[str], optional
        a file name or a directory to save log. If None,
        will not save log file. If ends with ".txt" or ".log",
        assumed to be a file name.
        Otherwise, logs will be saved to `{output}/log.txt`.
    name : str, optional
        the root module name of this logger, by default "GreatX"
    distributed_rank : int, optional
        used for distributed training, by default 0
    mode : str, optional
        mode for the output file (if output is given), by default 'w'.
    color : bool, optional
        whether to use color when printing, the `termcolor` package
        is required, by default True
    abbrev_name : Optional[str], optional
        an abbreviation of the module, to avoid long names in logs.
        Set to "" to not log the root module in logs.
        By default, None.

    Returns
    -------
    logging.Logger
        a logger

    Example
    -------
    >>> logger = setup_logger(name='my exp')

    >>> logger.info('message')
    [12/19 17:01:43 my exp]: message

    >>> logger.error('message')
    ERROR [12/19 17:02:22 my exp]: message

    >>> logger.warning('message')
    WARNING [12/19 17:02:32 my exp]: message

    >>> # specify output files
    >>> logger = setup_logger(output='log.txt', name='my exp')
    # additive, by default mode='w'
    >>> logger = setup_logger(output='log.txt', name='my exp', mode='a')

    # once you logger is set, you can call it by
    >>> logger = get_logger(name='my exp')
    """
    if color and colored is None:
        raise RuntimeError("Please install 'termcolor' to use colored outputs"
                           " when printing.")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)

        dirs = os.path.dirname(filename)
        if dirs:
            if not os.path.isdir(dirs):
                os.makedirs(dirs)
        file_handle = logging.FileHandler(filename=filename, mode=mode)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(plain_formatter)
        logger.addHandler(file_handle)

    return logger


def get_logger(name: str = "GreatX") -> logging.Logger:
    """Get a logger for a given name.

    Parameters
    ----------
    name : str, optional
        name of the logger, by default "GreatX"

    Returns
    -------
    a logger for the given name
    """
    return logging.getLogger(name)


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:  # noqa
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
