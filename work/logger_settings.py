import logging
import logging.config

log_file = 'D:\Clarifruit\cherry_stem\work\logs\log_file.log'
ERROR_FORMAT = "%(levelname)s:%(asctime)s in %(funcName)s in %(filename) at line %(lineno)d: %(message)s"
DEBUG_FORMAT = '%(asctime)s:%(name)s:%(message)s'
form = "%(asctime)s-6d  %(levelname)-8s %(name)s: %(message)s"
LEVEL = 'DEBUG'
LOG_CONFIG = {'version': 1,
              'disable_existing_loggers': False,
              'formatters': {'error': {'format': form},
                             'debug': {'format': form}},

              'handlers': {'console': {'class': 'logging.StreamHandler',
                                       'formatter': 'debug',
                                       'level': logging.DEBUG,
                                       'stream': 'ext://sys.stdout'},

                           'file': {'class': 'logging.FileHandler',
                                    'filename': log_file,
                                    'formatter': 'error',
                                    'level': logging.ERROR}},
              'root': {
                  'level': LEVEL,
                  'handlers': ('console', 'file')},

              'segmentation': {
                  'level': LEVEL,
                  'handlers': ('console', 'file')}
              }


def configure_logger():
    logging.config.dictConfig(LOG_CONFIG)
