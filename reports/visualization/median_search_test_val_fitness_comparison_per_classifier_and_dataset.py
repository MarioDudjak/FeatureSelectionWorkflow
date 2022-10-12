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
    for optimiser in fs_wrappers:
        for classifier in classifiers:
            val_filename = optimiser.name + "_" + classifier.name + "_search"
            test_filename = optimiser.name + "_" + classifier.name + "_search_test"
            print("Processing files {0} and {1}".format(val_filename, test_filename))

            header, val_data = CsvProcessor().read_file(
                filename='summary/' + experiment_name + "/" + val_filename)
            header, test_data = CsvProcessor().read_file(
                filename='summary/' + experiment_name + "/" + test_filename)

            if header is not None and val_data is not None and test_data is not None:
                for i, row in enumerate(zip(val_data, test_data)):    # row is tuple of lists (val and test scores at search points for dataset)
                    if all(row) and i % no_of_metrics == 0:
                        dataset = row[0][0].split('_')[0]
                        fitness_metric = '-'.join(row[0][0].split('_')[1:-1])

                        plt.ylabel(fitness_metric, fontsize=13)
                        plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)
                        plt.xlim([0.01, 1])
                        plt.xticks(range(0, 14), data_point_labels)
                        plt.tick_params()

                        markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']

                        for j, scores in enumerate(row):
                            plt.plot([float(score) for score in scores[1:]], lw=0.75, ms=4, marker=markers[j])

                        plt.legend(labels=[optimiser.name + "\_" + classifier.name + "\_val", optimiser.name + "\_" + classifier.name + "\_test"], loc="center", bbox_to_anchor=(0.725, 0.79), ncol=1, fancybox=False,
                                   framealpha=0.9)
                        plt.tight_layout()
                        plt.grid(b=True, linestyle=':')
                        plt.show()

                        #plt.savefig(experiment_setup + "/" + '_'.join([classifier.name, optimiser_name, dataset]) + ".pdf", format='pdf', dpi=300)
                        plt.close()







