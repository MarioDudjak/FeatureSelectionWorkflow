from pathlib import Path

from src.utils.file_handling.processors import FileProcessorFactory


class DatasetProvider:
    RAW_DATASET_FOLDER = ''
    PROCESSED_DATASET_FOLDER = ''
    SYNTHETIC_DATASET_FOLDER = ''

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DatasetProvider, cls).__new__(cls)
            cls.RAW_DATASET_FOLDER = cls._get_raw_dataset_folder()
            cls.PROCESSED_DATASET_FOLDER = cls._get_processed_dataset_folder()
            cls.SYNTHETIC_DATASET_FOLDER = cls._get_synthetic_dataset_folder()
        return cls.instance

    @classmethod
    def _get_raw_dataset_folder(cls):
        return cls._get_folder('raw/files/arff')

    @classmethod
    def _get_processed_dataset_folder(cls):
        return cls._get_folder('processed')

    @classmethod
    def _get_synthetic_dataset_folder(cls):
        return cls._get_folder('synthetic')

    @classmethod
    def _get_folder(cls, folder_name):
        current_directory = Path(__file__).resolve()
        data_directory = current_directory.parent.parent.parent.joinpath('data')

        if data_directory.exists() and data_directory.is_dir():
            return data_directory.joinpath(folder_name)
        else:
            # Exception
            return

    @classmethod
    def get_raw_dataset_list(cls):
        return cls._get_dataset_list(cls.RAW_DATASET_FOLDER)

    @classmethod
    def get_processed_dataset_list(cls):
        return cls._get_dataset_list(cls.PROCESSED_DATASET_FOLDER)

    @classmethod
    def get_synthetic_dataset_list(cls):
        return cls._get_dataset_list(cls.SYNTHETIC_DATASET_FOLDER)

    @classmethod
    def _get_dataset_list(cls, dataset_folder):
        if dataset_folder is not None:
            dataset_list = []
            file_processor_factory = FileProcessorFactory()
            for file_path in dataset_folder.iterdir():
                file_processor = file_processor_factory.get_file_processor(file_path=file_path)
                data, meta = file_processor.get_dataset(filename=file_path)
                dataset_list.append(meta)
            return dataset_list
        else:
            return


def main():
    # raw_datasets = DatasetProvider().get_raw_dataset_list()
    processed_datasets = DatasetProvider().get_processed_dataset_list()
    # synthetic_datasets = DatasetProvider().get_synthetic_dataset_list()

    print("Names:")
    for dataset in processed_datasets:
        print(dataset.name)

    print("Instances:")
    for dataset in processed_datasets:
        print(dataset.instances)

    print("Features:")
    for dataset in processed_datasets:
        print(dataset.features)

    print("Classes:")
    for dataset in processed_datasets:
        print(dataset.classes)

    print("IR:")
    for dataset in processed_datasets:
        print(dataset.IR)


if __name__ == "__main__":
    main()
