import matplotlib.pyplot as plt
from matplotlib import rc

from src.experiment.setup import classifiers, fitnesses
from src.features.wrappers import fs_wrappers
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
no_of_metrics = 4

evaluation_procedures = ['_holdout', '3Fold', '5Fold']
labels = [r'best', r'mean', r'worst']
for fitness_metric in fitnesses:
    for classifier in classifiers:
        scores = {}
        for procedure in evaluation_procedures:
            for optimiser in fs_wrappers:
                scores[optimiser.name + procedure] = {}
                val_filename = fitness_metric + "_" + procedure + "_" + optimiser.name + "_" + classifier.name + "_search"
                test_filename = val_filename + "_test"
                print("Processing files {0}".format(val_filename))

                header, val_data = CsvProcessor().read_file(
                    filename='summary/' + val_filename)
                header, test_data = CsvProcessor().read_file(
                    filename='summary/' + test_filename)

                if header is not None and val_data is not None and test_data is not None:
                    for i, row in enumerate(zip(val_data,
                                                test_data)):  # row is tuple of lists (val and test scores at search points for dataset)
                        if all(row) and i % no_of_metrics == 0:
                            dataset = row[0][0].split('_')[0]
                            scores[optimiser.name + procedure][dataset] = []

                            for results in row:
                                scores[optimiser.name + procedure][dataset].append([float(r) for r in results[1:]])

        for dataset, results in scores['GAMu_10000_holdout'].items():
            legend = []
            markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']
            colors = ['blue', 'green', 'red', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink']

            for optimiser in fs_wrappers:
                for j, procedure in enumerate(evaluation_procedures):
                    fitness_name = '-'.join(procedure.split('_'))
                    optimiser_name = '-'.join(optimiser.name.split('_'))
                    if bool(scores[optimiser.name + procedure]):
                        plt.plot(scores[optimiser.name + procedure][dataset][0], lw=0.75, ms=4, marker=markers[0],
                                 color=colors[j])
                        legend.append('-'.join([optimiser_name + "-" + fitness_name, r'val']))
                        plt.plot(scores[optimiser.name + procedure][dataset][1], lw=0.75, ms=4, marker=markers[1],
                                 color=colors[j])
                        legend.append('-'.join([optimiser_name + "-", fitness_name, r'test']))

                plt.ylabel(fitness_metric.split('_')[0], fontsize=13)
                plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)
                plt.xlim([0.01, 1])
                plt.xticks(range(0, 14), data_point_labels)
                plt.tick_params()
                plt.title(classifier.name + "-" + dataset)
                plt.legend(labels=legend, fancybox=False, framealpha=0.9)
                plt.tight_layout()
                plt.grid(b=True, linestyle=':')
                # plt.show()

                plt.savefig('_'.join([optimiser.name, classifier.name, dataset]) + "_overfitting.pdf", format='pdf', dpi=300)
                plt.close()

# for fitness_metric in fitnesses:
#     for classifier in classifiers:
#         scores = {}
#         for procedure in evaluation_procedures:
#             for optimiser in fs_wrappers:
#                 scores[optimiser.name + procedure] = {}
#                 val_filename = fitness_metric + "_" + procedure + "_" + optimiser.name + "_" + classifier.name + "_search"
#                 test_filename = val_filename + "_test"
#                 print("Processing files {0}".format(val_filename))
#
#                 header, val_data = CsvProcessor().read_file(
#                     filename='summary/' + val_filename)
#                 # header, test_data = CsvProcessor().read_file(
#                 #     filename='summary/' + test_filename)
#
#                 if header is not None and val_data is not None:
#                     for i, row in enumerate(val_data):
#                         if all(row) and i % no_of_metrics == 0:
#                             dataset = row[0].split('_')[0]
#                             scores[optimiser.name + procedure][dataset] = []
#
#                         if i % no_of_metrics == 0 or i % no_of_metrics == 1 or i % no_of_metrics == 2:
#                             scores[optimiser.name + procedure][dataset].append([float(r) for r in row[1:]])
#
#         for dataset, results in scores[optimiser.name + procedure].items():
#
#             for j, procedure in enumerate(evaluation_procedures):
#                 fitness_name = '-'.join(procedure.split('_'))
#                 legend = []
#                 markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']
#                 colors = ['blue', 'green', 'red', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink']
#                 plt.ylabel(fitness_metric.split('_')[0], fontsize=13)
#                 plt.xlabel(r'NFEs\textsubscript{max}', fontsize=13)
#                 plt.xlim([0.01, 1])
#                 plt.xticks(range(0, 14), data_point_labels)
#                 plt.tick_params()
#                 plt.title(classifier.name + "-" + dataset)
#                 for optimiser in fs_wrappers:
#                     for idx, res in enumerate(scores[optimiser.name + procedure][dataset]):
#                         plt.plot(scores[optimiser.name + procedure][dataset][idx], lw=0.75, ms=4, marker=markers[idx],
#                                  color=colors[idx])
#                         legend.append('-'.join([optimiser.name + fitness_metric.split('_')[0], labels[idx]]))
#                     # plt.plot(scores[optimiser.name + procedure][dataset][1], lw=0.75, ms=4, marker=markers[1],
#                     #          color=colors[j])
#                     # legend.append('-'.join([optimiser.name + fitness_name, r'test']))
#
#                     plt.legend(labels=legend, fancybox=False, framealpha=0.9)
#                     plt.tight_layout()
#                     plt.grid(b=True, linestyle=':')
#                     # plt.show()
#
#                     plt.savefig('_'.join([optimiser.name + procedure, classifier.name, dataset]) + "_convergence.pdf", format='pdf', dpi=300)
#                     plt.close()
