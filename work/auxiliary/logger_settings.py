import logging
import logging.config
import os
from datetime import  datetime


DEFUALT_PATH = 'D:\Clarifruit\cherry_stem\work\logs\log_file.log'
ERROR_FORMAT = "%(levelname)s:%(asctime)s in %(funcName)s in %(filename)" \
               " at line %(lineno)d: %(message)s"
DEBUG_FORMAT = '%(asctime)s:%(name)s:%(message)s'
FORM = "%(asctime)s-6d  %(levelname)-8s %(name)s: %(message)s"


def configure_logger(name, console_level='INFO', file_level='ERROR',
                     out_path=DEFUALT_PATH):
    """
    A method to configure a logger for console logging and logging to file,
     with different levels
    :param name: name the of file logs dest path
    :param console_level: the logger level for the console handler
    :param file_level: the logger level for the file writer handler
    :param out_path: the destination path for the log files
    :return: a configured logger
    """

    curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(out_path, name + curr_time +"_log.log")

    log_config = {'version': 1,
                  'disable_existing_loggers': False,
                  'formatters': {'error': {'format': FORM},
                                 'debug': {'format': FORM}},

                  'handlers': {'console': {'class': 'logging.StreamHandler',
                                           'formatter': 'debug',
                                           'level': console_level,
                                           'stream': 'ext://sys.stdout'},

                               'file': {'class': 'logging.FileHandler',
                                        'filename': log_path,
                                        'formatter': 'error',
                                        'level': file_level}},
                  'root': {
                      'level': 'DEBUG',
                      'handlers': ('console', 'file')},

                  }

    logging.config.dictConfig(log_config)
    logging.getLogger('PIL.Image').setLevel('ERROR')
    # otherwise will get:
    # PIL.Image: Error closing: 'Image' object has no attribute 'fp'

