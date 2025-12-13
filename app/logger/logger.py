import logging
from logging.handlers import RotatingFileHandler


info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.INFO)
handler = RotatingFileHandler("info.log", maxBytes=1000000, backupCount=3)
handler.setLevel(logging.INFO)
info_logger.addHandler(handler)

debug_logger = logging.getLogger("debug_logger")
debug_logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("debug.log", maxBytes=1000000, backupCount=3)
handler.setLevel(logging.DEBUG)
debug_logger.addHandler(handler)

error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("error.log", maxBytes=1000000, backupCount=3)
handler.setLevel(logging.ERROR)
error_logger.addHandler(handler)
