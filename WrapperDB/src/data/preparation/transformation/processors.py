from abc import abstractmethod, ABC

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd


class Processor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, dataset):
        pass

    @classmethod
    def extract_data(cls, dataset):
        features = dataset.iloc[:, :-1]
        labels = dataset.iloc[:, -1]
        return features, labels

    @classmethod
    def join_data(cls, features, labels):
        return pd.concat([features, labels], axis=1)


class MissingValuesHandler(Processor):
    def process(self, dataset):
        dataset = dataset.dropna(axis=0)
        return dataset


class FeatureCleaner(Processor):
    def process(self, dataset):
        dataset.iloc[:, :-1].drop(columns=[col for col in list(dataset.iloc[:, :-1]) if dataset[col].nunique() <= 1])
        return dataset


class FeatureEncoding(Processor):

    def process(self, dataset):
        nominal_columns = [col for col in dataset.columns[0:-1] if [val for val in dataset[col] if
                                                                    not isinstance(val, int) and not isinstance(val,
                                                                                                                float) and not val.isdigit()]]
        if nominal_columns:
            new_dataset = dataset.iloc[:, :-1]
            new_dataset.drop(nominal_columns, inplace=True, axis=1)
            dataset = pd.concat(
                [new_dataset, pd.get_dummies(dataset.loc[:, nominal_columns], columns=nominal_columns),
                 dataset.iloc[:, -1]], axis=1)

        return dataset


class FeatureNormalizer(Processor):

    def process(self, dataset):
        dataset.iloc[:, :-1] = MinMaxScaler().fit_transform(dataset.iloc[:, :-1])
        return dataset


class ClassLabelEncoder(Processor):

    def process(self, dataset):
        dataset.iloc[:, -1] = LabelEncoder().fit_transform(dataset.iloc[:, -1])
        return dataset


class DuplicateInstanceHandler(Processor):
    def process(self, dataset):
        pass
