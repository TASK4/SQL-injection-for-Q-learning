import logging
import logging.config
import configparser

class CustomLogger:
    def __init__(self, config_file='config/logging_config.ini'):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        # Simple setup based on ini (expand if use dictConfig later)
        logging.basicConfig(
            level=getattr(logging, config['DEFAULT']['LogLevel']),
            format=config['DEFAULT']['LogFormat']
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)