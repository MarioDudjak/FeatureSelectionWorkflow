import csv
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from src.utils.meta import Metadata


class FileProcessor(metaclass=ABCMeta):
    _current_directory = Path(__file__).resolve()
    _reports_directory = _current_directory.parent.parent.parent.parent.joinpath('reports')
    _synthetic_directory = _current_directory.parent.parent.parent.parent.joinpath('data').joinpath('synthetic')
    _results_directory = _current_directory.parent.parent.parent.parent.joinpath('reports').joinpath('results')
    _summary_directory = _current_directory.parent.parent.parent.parent.joinpath('reports').joinpath('summary')
    _logs_directory = _current_directory.parent.parent.parent.parent.joinpath('reports').joinpath('logs')

    @abstractmethod
    def get_dataset(self, filename):
        pass

    @classmethod
    def get_synthetic_datasets_folder(cls):
        if cls._synthetic_directory.exists() and cls._synthetic_directory.is_dir():
            return cls._synthetic_directory

    @classmethod
    def get_reports_folder(cls):
        if cls._reports_directory.exists() and cls._reports_directory.is_dir():
            return cls._reports_directory

    @classmethod
    def get_results_folder(cls):
        if cls._results_directory.exists() and cls._results_directory.is_dir():
            return cls._results_directory

    @classmethod
    def get_summary_folder(cls):
        if cls._summary_directory.exists() and cls._summary_directory.is_dir():
            return cls._summary_directory

    @classmethod
    def get_logs_folder(cls):
        if cls._logs_directory.exists() and cls._logs_directory.is_dir():
            return cls._logs_directory


class CsvProcessor(FileProcessor):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CsvProcessor, cls).__new__(cls)
        return cls.instance

    @classmethod
    def get_dataset(cls, filename):
        dataset = pd.read_csv(filename, header=0, index_col=0)
        meta = cls._extract_metadata(dataset, filename)
        return dataset, meta

    @classmethod
    def _extract_metadata(cls, data, filename):

        IR = (len(data.iloc[:, -1]) - np.count_nonzero(data.iloc[:, -1])) / np.count_nonzero(data.iloc[:, -1])
        if IR < 1:
            IR = 1 / IR
        return Metadata(filename.stem, filename, data.shape[0], data.shape[1] - 1, [], data.iloc[:, -1].nunique(), IR)

    @classmethod
    def _get_csv_path(cls, path):
        return path.joinpath('files').joinpath('csv')

    @classmethod
    def write_file(cls, filename, header, data):
        with open(filename, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            if os.path.getsize(filename) == 0:
                writer.writerow(header)

            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            dimensionality = np.array(data).ndim

            if dimensionality == 2 or (isinstance(data, list) and any(isinstance(row, list) for row in data)):
                writer.writerows(data)
                writer.writerow([])

            else:
                writer.writerow(data)

    @classmethod
    def _write_2d_list(cls, writer, data):
        data = list(data)
        if not all(len(row) == len(data[0]) for row in data):
            for row in data:
                writer.writerow(row)
        else:
            writer.writerows(data)


    @classmethod
    def read_file(cls, filename):
        filepath = super().get_reports_folder().joinpath(filename + '.csv')
        if filepath.exists():
            with open(filepath.resolve(), 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                data = []
                i = 0
                for row in reader:
                    if i == 0:
                        header = row
                    else:
                        data.append(row)
                    i = i + 1
            return header, data

        return None, None

    @classmethod
    def save_raw_results(cls, filename, header, data):
        filepath = super().get_results_folder().joinpath(filename + '.csv')
        cls.write_file(filepath.resolve(), header, data)

    @classmethod
    def save_log_results(cls, filename, header, data):
        filepath = super().get_logs_folder().joinpath(filename + '.csv')
        cls.write_file(filepath.resolve(), header, data)

    @classmethod
    def save_summary_results(cls, filename, header, data):
        filepath = super().get_summary_folder().joinpath(filename + '.csv')
        cls.write_file(filepath.resolve(), header, data)

    @classmethod
    def save_synthetic_datasets(cls, filename, header, data):
        filepath = super().get_synthetic_datasets_folder().joinpath(filename + '.csv')
        cls.write_file(filepath.resolve(), header, data)


class ArffProcessor(FileProcessor):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ArffProcessor, cls).__new__(cls)
        return cls.instance

    @classmethod
    def get_dataset(cls, filename):
        print(filename)
        if filename.exists():
            data, meta = arff.loadarff(filename.resolve())
            meta = cls._extract_metadata(data, meta, file_path=filename)
            return data, meta
        else:
            print("File with path {0} does not exist".format(filename.resolve()))
            return None

    @classmethod
    def _extract_metadata(cls, data, meta, file_path):
        labels = [row[-1] for row in data]
        IR = len([elem for elem in labels if elem == list(set(labels))[0]]) / (
            len([elem for elem in labels if elem == list(set(labels))[1]]))
        if IR < 1:
            IR = 1 / IR
        # list(meta._attributes)[-1]
        return Metadata(meta.name.capitalize(), file_path, len(data), len(meta.names()) - 1,
                        list({key for (key, value) in meta._attributes.items() if
                              value.type_name == 'nominal' and [val for val in value.values if
                                                                not val.isdigit()] and key != 'Class' and key != 'class'}),
                        len(set(labels)), IR)

    @classmethod
    def _get_arff_path(cls, path):
        return path.joinpath('files').joinpath('arff')


class DatProcessor(FileProcessor):
    arff_processor = ArffProcessor()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DatProcessor, cls).__new__(cls)
            cls.arff_processor = ArffProcessor()
        return cls.instance

    @classmethod
    def get_dataset(cls, filename):
        if filename.exists():
            cls._prepare_arff_file(filename)
            return cls.arff_processor.get_dataset(filename)

    @classmethod
    def _prepare_arff_file(cls, filename):
        file = open(filename, "r")
        lines = file.readlines()
        file.close()

        new_file = open(filename, "w")
        lines = [line for line in lines if
                 '<null>' not in line and "@inputs" not in line and "@outputs" not in line and "@input" not in line and "@output" not in line]
        for line in lines:
            new_file.write(line)

        new_file.close()

    @classmethod
    def _get_dat_path(cls, path):
        return path.joinpath('files').joinpath('dat')


class FileProcessorFactory:

    def __init__(self):
        self._creators = {
            ".csv": CsvProcessor(),
            ".arff": ArffProcessor(),
            ".dat": DatProcessor()
        }

    def register_form(self, format, creator):
        self._creators[format] = creator

    def get_file_processor(self, file_path):
        creator = self._creators.get(file_path.suffix)
        if not creator:
            raise ValueError(file_path)

        return creator
