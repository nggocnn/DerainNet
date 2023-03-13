import logging
from colorlog import ColoredFormatter

class DeRainLogger:
    logger = None

    @staticmethod
    def get_logger():
        if DeRainLogger.logger is None:
            LOG_LEVEL = logging.DEBUG
            LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOG_FORMAT)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)

            DeRainLogger.logger = logging.getLogger('pythonConfig')
            DeRainLogger.logger.setLevel(LOG_LEVEL)
            DeRainLogger.logger.addHandler(stream)

            DeRainLogger.logger.info('Logger initialized!')

            # DeRainLogger.logger.debug("A quirky message only developers care about")
            # DeRainLogger.logger.info("Curious users might want to know this")
            # DeRainLogger.logger.warning("Something is wrong and any user should be informed")
            # DeRainLogger.logger.error("Serious stuff, this is red for a reason")
            # DeRainLogger.logger.critical("OH NO everything is on fire")

        return DeRainLogger.logger
