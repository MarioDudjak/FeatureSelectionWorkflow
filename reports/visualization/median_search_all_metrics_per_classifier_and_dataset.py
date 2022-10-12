import matplotlib.pyplot as plt
from matplotlib import rc

from src.experiment.fs_study.setup import classifiers, experiment_setups
from src.features.selection.wrappers import fs_wrappers
from src.utils.file_handling.processors import CsvProcessor

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

data_point_labels = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
metric_positions = [0, 3, 5, 7, 13, 15]
markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']

for experiment_name in experiment_setups.keys():
    plot_data = {}
    for wrapper in fs_wrappers:
        for classifier in classifiers:
            plot_data[classifier.name] = {}
            val_filename = '_'.join([wrapper.name, classifier.name, "search"])
            test_filename = '_'.join([wrapper.name, classifier.name, "search", "test"])
            print("Processing files {0} and {1}".format(val_filename, test_filename))

            val_header, val_data = CsvProcessor().read_file(
                filename='summary/' + experiment_name + "/" + val_filename)
            test_header, test_data = CsvProcessor().read_file(
                filename='summary/' + experiment_name + "/" + test_filename)

            if val_header is not None and val_data is not None:
                dataset = ''
                labels = []
                pos = 0
                for i, row in enumerate(val_data):
                    if row[0].split('_')[0] != dataset:
                        if i == 0:
                            dataset = row[0].split('_')[0]
                        else:
                            plt.title('-'.join([experiment_name.replace('_', '-'), wrapper.name, classifier.name,
                                                dataset.replace('_', '')]))
                            plt.ylabel(r'Quality metrics', fontsize=13)
                            plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)
                            plt.xlim([data_point_labels[0], data_point_labels[-1]])
                            plt.ylim(top=1)
                            plt.xticks(range(0, 14), data_point_labels)
                            plt.tick_params()
                            plt.legend(labels=labels, fancybox=False, framealpha=0.9)
                            plt.tight_layout()
                            plt.grid(b=True, linestyle=':')
                            plt.show()
                            # plt.savefig('-'.join([experiment_name.replace('_', '-'), wrapper.name, classifier.name,
                            #                     dataset.replace('_', '')]) + ".pdf",
                            #             format='pdf', dpi=300)
                            plt.close()
                            labels = []
                            dataset = row[0].split('_')[0]
                            pos = 0

                    if pos in metric_positions:
                        scores = [float(score) for score in row[1:]]
                        plt.plot(scores, lw=0.75, ms=4, alpha=0.5, marker=markers[metric_positions.index(pos)])
                        metric = row[0][row[0].find(dataset) + len(dataset) + 1:].replace('_', '')
                        labels.append(metric)
                    pos += 1
            plt.close()
