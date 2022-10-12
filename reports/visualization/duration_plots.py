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

feature_counts = [13, 34, 61, 166, 310, 500]
colors = ['royalblue', 'orange', 'limegreen', 'crimson']
markers = ['o','o','o','o',  's', 's', 's', 's',  'D',  'D',  'D',  'D']

datasets = DatasetProvider().get_processed_dataset_list()
runs = 30
wrappers = ['GAMu', 'DE', 'PSO']
i = 0
labels = [r'GA+$1$-NN', r'GA+$5$-NN', r'GA+GNB', r'GA+SVM', r'DE+$1$-NN', r'DE+$5$-NN', r'DE+GNB', r'DE+SVM', r'PSO+$1$-NN', r'PSO+$5$-NN', r'PSO+GNB', r'PSO+SVM']
for classifier in classifiers:
    for wrapper in wrappers:
        filename = '_'.join(
            [wrapper, classifier.name, 'duration'])
        print("Processing file {0}".format(filename))

        header, data = CsvProcessor().read_file(filename='logs/outputQuality/duration/' + filename)
        duration_values = []
        duration_values.append(100*[(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
                           row[0] == 'Wine'][0])

        duration_values.append(100*
            [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
            row[0] == 'Ionosphere'][0])

        duration_values.append(100*
            [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
            row[0] == 'German_credit'][0])

        duration_values.append(100*
            [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
             row[0] == 'Clean2'][0])

        duration_values.append(100*
            [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
            row[0] == 'Uci_voice'][0])

        duration_values.append(100*
            [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
            row[0] == 'Madelon'][0])

        plt.plot(duration_values, '-o', lw=0.75, ms=4, alpha=0.5, marker=markers[i], color=colors[i % 4])



        i += 1

plt.xlabel(r'$d$', fontsize=18)
plt.ylabel(r'$\Delta t$', fontsize=18)
plt.xlim([0, 5])
plt.xticks(range(0, 6), feature_counts, fontsize=13)
plt.yticks(fontsize=13)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':')
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
#plt.show()
plt.savefig('arhiva_trajanje.pdf', format='pdf', dpi=300)
plt.close()

markers = ['o',  's',  'D']

datasets = DatasetProvider().get_processed_dataset_list()
runs = 30
wrappers = [r'PSOd_1NN', r'PSO(4-2))_5NN', r'EGAFS_GNB']
i = 0
labels = [r'PSO\textsubscript{D}', r'PSO(4-2)', r'EGAFS']
for wrapper in wrappers:
    filename = '_'.join(
        [wrapper, 'duration'])
    print("Processing file {0}".format(filename))

    header, data = CsvProcessor().read_file(filename='logs/outputQuality/duration/' + filename)
    duration_values = []
    duration_values.append(100*[(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
                       row[0] == 'Wine'][0])

    duration_values.append(100*
        [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
        row[0] == 'Ionosphere'][0])

    duration_values.append(100*
        [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
        row[0] == 'German_credit'][0])

    duration_values.append(100*
        [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
         row[0] == 'Clean2'][0])

    duration_values.append(100*
        [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
        row[0] == 'Uci_voice'][0])

    duration_values.append(100*
        [(float(row[2]) + float(row[3]) - float(row[1])) / float(row[1]) for row in data if row and
        row[0] == 'Madelon'][0])

    plt.plot(duration_values, '-o', lw=0.75, ms=4, alpha=0.5, marker=markers[i], color=colors[i])



    i += 1

plt.xlabel(r'$d$', fontsize=18)
plt.ylabel(r'$\Delta t$', fontsize=18)
plt.xlim([0, 5])
plt.xticks(range(0, 6), feature_counts, fontsize=13)
plt.yticks(fontsize=13)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':')
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
#plt.show()
plt.savefig('arhiva_trajanje2.pdf', format='pdf', dpi=300)
plt.close()