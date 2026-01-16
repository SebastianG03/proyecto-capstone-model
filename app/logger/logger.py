import logging
from logging.handlers import RotatingFileHandler

from app.core.config import DEBUG

FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)

level = logging.DEBUG if DEBUG else logging.INFO

info_logger = logging.getLogger('app_info')
info_logger.setLevel(level)
info_handler = RotatingFileHandler('info.log', maxBytes=100_000_000, backupCount=3)
info_handler.setLevel(level)
info_logger.addHandler(info_handler)

debug_logger = logging.getLogger('app_debug')
debug_logger.setLevel(level)
debug_handler = RotatingFileHandler('debug.log', maxBytes=100_000_000, backupCount=3)
debug_handler.setLevel(level)
debug_logger.addHandler(debug_handler)

error_logger = logging.getLogger('app_error')
error_logger.setLevel(logging.ERROR)
err_handler = RotatingFileHandler('error.log', maxBytes=100_000_000, backupCount=3)
err_handler.setLevel(logging.ERROR)
error_logger.addHandler(err_handler)

