import os
import logging
import logging.handlers


def get_logger(
    log_dir: str, 
    log_name: str = 'app_log.log',
    when: str = 'midnight',      
    interval: int = 1,           
    backup_count: int = 24      
) -> logging.Logger:    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(levelname)s>>>%(asctime)s>>>[%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s"
    )

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path, 
        when=when, 
        interval=interval, 
        backupCount=backup_count
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # first log
    logger.info('Logger is ready')
    logger.info(f"Log file: {log_dir}")

    return logger