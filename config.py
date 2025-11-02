import logging


class Config:
    """config for from_scratch nn project"""

    # logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = "logs/training.log"
    LOG_TO_CONSOLE = True
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    RANDOM_SEED = 42
    EPOCHS = 30
    LEARNING_RATE = 0.01

    # model
    NUM_FILTERS = 3
    FILTER_SIZE = 3
    STRIDE = 1
    POOLING_KERNEL_SIZE = 2


LOG_LEVEL = Config.LOG_LEVEL
LOG_FILE = Config.LOG_FILE
