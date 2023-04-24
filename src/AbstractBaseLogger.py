from datetime import datetime
from logging import StreamHandler, INFO, DEBUG, FileHandler, Logger

from .utils import prepend_dataset_root


class AbstractBaseLogger(Logger):
    def __init__(self, name: str):
        super().__init__(name)
        file_handler = FileHandler(prepend_dataset_root(f"logs/{name}.log"))
        file_handler.setLevel(DEBUG)
        self.addHandler(file_handler)
        stream_handler = StreamHandler()
        stream_handler.setLevel(INFO)
        self.addHandler(stream_handler)

    @classmethod
    def get_log_file_name(cls):
        curr_time = datetime.now().strftime("%d-%m-%YT%H-%M-%S")
        log_file_name = f"{cls.__name__}-{curr_time}"
        return log_file_name
