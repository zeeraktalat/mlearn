import tqdm, logging, os, colorlog

class TqdmStreamHandler(logging.StreamHandler):
    def __init__(self, level = logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class TqdmFileHandler(logging.FileHandler):
    def __init__(self, fp, level = logging.NOTSET):
        super(self.__class__, self).__init__(fp)
        self.logfile = open(fp, 'a', encoding = 'utf-8')

    @property
    def log(self):
        return self.logfile

    @log.setter
    def log(self, val):
        self.logfile = val

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file = self.logfile)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def initialise_loggers(logger, logfile):
    logdir = os.environ['PYLOGS']
    """ Initialise loggers """

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
    streamhandler.setFormatter(
            colorlog.ColoredFormatter(
                strformat,
                datefmt = dateformat,
                log_colors = colorformat))

    # Set up file logger
    filehandler = TqdmFileHandler(logdir + 'newsModel.log')
    filehandler.setFormatter(
            colorlog.ColoredFormatter(
                strformat,
                datefmt = dateformat,
                log_colors = colorformat))

    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)

    return logger

