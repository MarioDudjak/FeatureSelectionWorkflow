from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor
from src.experiment.setup import classifiers
from src.features.wrappers import fs_wrappers

import matplotlib.pyplot as plt
from matplotlib import rc

from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor
from src.experiment.setup import classifiers
from src.features.wrappers import fs_wrappers

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

imbalance_indices = [2, 3, 4, 8, 9, 10]
colors = ['royalblue', 'blueviolet', 'orange', 'chocolate', 'limegreen', 'forestgreen']
datasets = DatasetProvider().get_processed_dataset_list()
runs = 30
i = 0
for classifier in classifiers:
    i = 0
    for wrapper in fs_wrappers:
        x1 = []
        y1= []
        x2 = []
        y2 = []
        for dataset in datasets:
            filename = '_'.join(
                ['Full', classifier.name, dataset.name])
            print("Processing file {0}".format(filename))

            full_header, full_data = CsvProcessor().read_file(filename='logs/Full/' + filename)

            filename = '_'.join(['', wrapper.name, classifier.name, dataset.name])
            wrapper_header, wrapper_data = CsvProcessor().read_file(filename='logs/outputQuality/' + filename)
            archive_header, archive_data = CsvProcessor().read_file(
                filename='summary/' + wrapper.name + "_" + classifier.name + "__archive_combinations")
            try:
                wrapper_test_idx1 = wrapper_header.index("f1_macro")
            except:
                a = 5
            wrapper_test_idx2 = wrapper_header.index("recall")

            archive_test_idx1 = archive_header.index("sfs_intersection_test_F1")
            archive_test_idx2 = archive_header.index("sfs_intersection_test_sensitivity")

            if full_data:
                full_scores_F1 = [float(row[0]) for row in full_data if row]
                full_scores_TPR = [float(row[6]) for row in full_data if row]

            if wrapper_data:
                wrapper_scores_F1 = [float(row[wrapper_test_idx1]) for row in wrapper_data if row]
                wrapper_scores_TPR = [float(row[wrapper_test_idx2]) for row in wrapper_data if row]

                x1.append(sum([x1 - x2 for (x1, x2) in zip(wrapper_scores_F1, full_scores_F1)])/runs)
                y1.append(sum([x1 - x2 for (x1, x2) in zip(wrapper_scores_TPR, full_scores_TPR)])/runs)

            if archive_data:
                archive_scores_F1 = [float(row[archive_test_idx1]) for row in archive_data if row[0]==dataset.name]
                archive_scores_TPR = [float(row[archive_test_idx2]) for row in archive_data if row[0]==dataset.name]

                x2.append(sum([x1 - x2 for (x1, x2) in zip(archive_scores_F1, full_scores_F1)]) / runs)
                y2.append(sum([x1 - x2 for (x1, x2) in zip(archive_scores_TPR, full_scores_TPR)]) / runs)


        plt.scatter([x1[i] for i in range(len(x1)) if i not in imbalance_indices], [y1[i] for i in range(len(y1)) if i not in imbalance_indices], alpha=0.7, marker='o', c=colors[i], s=150)
        plt.scatter([x1[i] for i in range(len(x1)) if i in imbalance_indices],
                    [y1[i] for i in range(len(y1)) if i in imbalance_indices], alpha=0.7, marker='X', c=colors[i],
                    label='_nolegend_', s=150)

        plt.scatter([x2[i] for i in range(len(x2)) if i not in imbalance_indices],
                    [y2[i] for i in range(len(y2)) if i not in imbalance_indices], alpha=0.7, c=colors[i+1], s=150)
        plt.scatter([x2[i] for i in range(len(x2)) if i in imbalance_indices],
                    [y2[i] for i in range(len(y2)) if i in imbalance_indices], alpha=0.7, marker='X', c=colors[i+1], label='_nolegend_' ,s=150)

        i+=2

    plt.xlim([-0.1, 0.4])
    plt.ylim([-0.1, 0.4])
    plt.xlabel(r'$\Delta$F1', fontsize=30)
    plt.ylabel(r'$\Delta$TPR', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tick_params()
    plt.tight_layout()
    plt.grid(b=True, which='both', linewidth=0.9, linestyle='dashed', alpha=0.6)
    plt.legend(labels=[r'GA', r'GA+A',  r'DE', r'DE+A', r'PSO', r'PSO+A'], fancybox=False, framealpha=0.9, ncol=3)
    ax = plt.gca()
    ax.axhline(0, linestyle='-', linewidth=1.2, alpha=0.8, color='grey')  # horizontal lines
    ax.axvline(0, linestyle='-', linewidth=1.2, alpha=0.8, color='grey')  # vertical lines
    # plt.show()
    plt.savefig('F1_TPR_' + classifier.name + '.pdf', format='pdf', dpi=300)
    plt.close()

