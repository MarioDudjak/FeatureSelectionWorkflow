import matplotlib.pyplot as plt
from matplotlib import rc

from src.experiments.fs_study.setup import classifiers
from src.features.selection.wrappers import fs_wrappers
from src.experiments.fs_study.setup import experiment_setups
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

data_point_labels = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
no_of_metrics = 17

for experiment_name in experiment_setups.keys():
    fitness_metric = experiment_name.split('_')[2]
    for optimiser in fs_wrappers:
        scores = {}  # keys=classifiers, values=dictionary
        for classifier in classifiers:
            scores[classifier.name] = {}
            test_filename = optimiser.name + "_" + classifier.name + "_search_test"
            print("Processing file {0}".format(test_filename))

            header, test_data = CsvProcessor().read_file(
                filename='summary/' + experiment_name + "/" + test_filename)

            if header is not None and test_data is not None:
                for i, row in enumerate(test_data):
                    if all(row) and i % no_of_metrics == 0:
                        dataset = row[0].split('_')[0]
                        scores[classifier.name][dataset] = [float(r) for r in row[1:]]

        for dataset, results in scores[classifier.name].items():
            legend = []
            markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']
            colors = ['blue', 'green', 'red', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink']
            plt.ylabel(fitness_metric, fontsize=13)
            plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)
            plt.xlim([0.01, 1])
            plt.xticks(range(0, 14), data_point_labels)
            plt.tick_params()
            plt.title(experiment_name.replace('_', '-') + "-" + optimiser.name + "-" + dataset)
            for j, classifier in enumerate(classifiers):
                plt.plot(scores[classifier.name][dataset], lw=0.75, ms=4, marker=markers[j], color=colors[j])
                legend.append(classifier.name + r"-test")

            plt.legend(labels=legend, fancybox=False, framealpha=0.9)
            plt.tight_layout()
            plt.grid(b=True, linestyle=':')
            plt.show()

            # plt.savefig(experiment_setup + "/" + '_'.join([classifier.name, optimiser_name, dataset]) + ".pdf", format='pdf', dpi=300)
            plt.close()
