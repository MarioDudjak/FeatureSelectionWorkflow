from dataclasses import dataclass

import pandas as pd

from src.data.preparation.transformation.processors import FeatureNormalizer, ClassLabelEncoder, \
    MissingValuesHandler, FeatureEncoding, FeatureCleaner
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import FileProcessorFactory


@dataclass
class ETLPipeline(object):
    processors: list

    def run(self, datasets):
        """Iterate over all raw datasets, apply processors and save to processed directory as pandas Dataframes"""
        for file in datasets:
            print('Processing file {}...'.format(file.path))
            file_processor = FileProcessorFactory().get_file_processor(file.path)
            dataset = pd.DataFrame(file_processor.get_dataset(file.path)[0])

            for processor in self.processors:
                dataset = dataset.pipe(processor.process)

            dataset.to_csv(DatasetProvider.PROCESSED_DATASET_FOLDER.joinpath(
                file.name + '.csv'))  # save dataset to processed folder


processors = [
    MissingValuesHandler(),
    # FeatureCleaner(),
    # FeatureEncoding(),
    FeatureNormalizer(),
    ClassLabelEncoder()
]

pipeline = ETLPipeline(processors)
pipeline.run(datasets=DatasetProvider().get_raw_dataset_list())
