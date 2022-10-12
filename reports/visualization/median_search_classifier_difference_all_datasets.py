import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from src.models.setup import classifiers
from src.utils.file_handling.processors import CsvProcessor

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

data_point_labels = ["0.01", "0.02", "0.03", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
experiment_setups = ['fixedSplit', 'fixedSplits_strongDEinequality', 'variableSplits', 'crossValidation']

optimiser_name = 'DE'
binariser_name = 'FixedBinarizer'

for experiment_setup in experiment_setups:
    plot_data = {}
    for classifier in classifiers:
        plot_data[classifier.name] = {}
        val_filename = classifier.name + "_search"
        test_filename = classifier.name + "_search_test"
        print("Processing files {0} and {1}".format(val_filename, test_filename))

        header, val_data = CsvProcessor().read_file(
            filename='summary/' + experiment_setup + "/" + val_filename)
        header, test_data = CsvProcessor().read_file(
            filename='summary/' + experiment_setup + "/" + test_filename)

        if header is not None and val_data is not None and test_data is not None:
            for row in zip(val_data,
                           test_data):  # row is tuple of lists (val and test scores at search points for dataset)
                if all(row):
                    dataset = row[0][0]
                    plot_data[classifier.name][dataset] = []
                    for i, scores in enumerate(row):
                        plot_data[classifier.name][dataset].append([float(score) for score in scores[1:]])

    for classifier, datasets_results in plot_data.items():
        val_results = [dataset_score[0] for dataset_score in datasets_results.values()]
        test_results = [dataset_score[1] for dataset_score in datasets_results.values()]
        data = []
        medians = []
        for i in range(len(data_point_labels)):
            data.append([float(val_results[j][i]) - float(test_results[j][i]) for j in range(len(val_results))])
            medians.append(np.median(data[i]))

        plt.title(str(experiment_setup.replace('_', '')) + "-" + classifier.replace('_', ''))
        plt.ylabel(r'Difference between validation and test quality (F1-score)', fontsize=13)
        plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)

        plt.boxplot(data)
        plt.plot(range(1, 15), medians, lw=1.5, ms=4)
        plt.xticks(range(1, 15), data_point_labels)
        plt.tick_params()

        # plt.legend(labels=labels, loc="center", ncol=2, fancybox=False, framealpha=0.9, bbox_to_anchor=(0.65, 0.125))
        plt.tight_layout()
        plt.grid(b=True, linestyle=':')
        #plt.show()

        plt.savefig(experiment_setup + "/" + '_'.join([optimiser_name, classifier]) + ".pdf",
                    format='pdf', dpi=300)
        plt.close()
