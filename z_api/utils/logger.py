"""Module for setting up loggers."""

import tqdm
import logging
import os
import colorlog


class TqdmStreamHandler(logging.StreamHandler):
    """Handles writing log to the terminal streams."""

    def __init__(self, level = logging.NOTSET):
        """Initialise class variables."""
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        """Emit messages.

        :param record: Message to be logged.
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.handleError(record)


class TqdmFileHandler(logging.FileHandler):
    """Handles writing log to the file."""

    def __init__(self, fp, level = logging.NOTSET):
        """Initialise class variables."""
        super(self.__class__, self).__init__(fp)
        self.logfile = open(fp, 'a', encoding = 'utf-8')

    @property
    def log(self):
        """Get and set log file."""
        return self.logfile

    @log.setter
    def log(self, val: str):
        self.logfile = open(val, 'a', encoding = 'utf-8')

    def emit(self, record):
        """Emit messages.

        :param record: Message to be logged.
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file = self.logfile)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.handleError(record)


def initialise_loggers(logger, logfile):
    """Initialise loggers."""
    logdir = os.environ['PYLOGS']

    # Set formatting
    strformat   = '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s'
    dateformat  = '%d-%m-%Y %H:%M:%S'
    colorformat = {
        'DEBUG': 'cyan',
        'INFO': 'white',
        'SUCCESS:': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bold_red'}

    # Initialise loggers and level
    logger = colorlog.getLogger(logger)
    logger.setLevel(logging.DEBUG)

    success = 5
    logging.addLevelName(success, 'SUCCESS')

    # Set up stream logger
    streamhandler = TqdmStreamHandler()
    streamhandler.setFormatter(colorlog.ColoredFormatter(
                               strformat,
                               datefmt = dateformat,
                               log_colors = colorformat))

    # Set up file logger
    unified = TqdmFileHandler(logdir + 'api.log')
    unified.setFormatter(colorlog.ColoredFormatter(
                         strformat,
                         datefmt = dateformat,
                         log_colors = colorformat))

    individual = TqdmFileHandler(logdir + '{0}.log'.format(logfile))
    individual.setFormatter(colorlog.ColoredFormatter(
                            strformat,
                            datefmt = dateformat,
                            log_colors = colorformat))
    logger.addHandler(streamhandler)
    logger.addHandler(unified)
    logger.addHandler(individual)

    return logger
