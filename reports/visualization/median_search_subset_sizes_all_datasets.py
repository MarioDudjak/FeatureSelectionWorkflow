import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from src.models.setup import classifiers
from src.utils.file_handling.processors import CsvProcessor
from src.utils.datasets import DatasetProvider
font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

data_point_labels = ["0.01", "0.02", "0.03", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
experiment_setups = ['fixedSplit', 'fixedSplits_strongDEinequality', 'variableSplits', 'crossValidation']
datasets_dimensions = []
optimiser_name = 'DE'
binariser_name = 'FixedBinarizer'

for dataset in DatasetProvider().get_processed_dataset_list():
    datasets_dimensions.append(dataset.features)

for experiment_setup in experiment_setups:
    plot_data = {}
    for classifier in classifiers:
        plot_data[classifier.name] = {}
        filename = classifier.name + "_fss_search"
        print("Processing file {0}".format(filename))

        header, data = CsvProcessor().read_file(
            filename='summary/' + experiment_setup + "/" + filename)

        if header is not None and data is not None:
            for i, row in enumerate(data):
                if row:
                    dataset = row[0]
                    plot_data[classifier.name][dataset] = [float(size) / datasets_dimensions[i] for size in row[1:]]

    for classifier, datasets_results in plot_data.items():
        box_plot_data = []
        medians = []
        for i in range(len(data_point_labels)):
            box_plot_data.append([dataset_scores[i] for dataset_scores in list(datasets_results.values())])
            medians.append(np.median(box_plot_data[i]))

        plt.title(str(experiment_setup.replace('_', '')) + "-" + classifier.replace('_', ''))
        plt.ylabel(r'Median feature subset sizes', fontsize=13)
        plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)

        plt.boxplot(box_plot_data)
        plt.plot(range(1, 15), medians, lw=1.5, ms=4)
        plt.xticks(range(1, 15), data_point_labels)
        plt.tick_params()

        # plt.legend(labels=labels, loc="center", ncol=2, fancybox=False, framealpha=0.9, bbox_to_anchor=(0.65, 0.125))
        plt.tight_layout()
        plt.grid(b=True, linestyle=':')
        plt.show()

        # plt.savefig(experiment_setup + "/" + '_'.join([optimiser_name, classifier]) + ".pdf",
        #             format='pdf', dpi=300)
        plt.close()
