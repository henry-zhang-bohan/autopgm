import pandas
import numpy as np


class FileParser(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.parsed_data_frames = []
