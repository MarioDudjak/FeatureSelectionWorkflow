import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from src.utils.file_handling.processors import CsvProcessor
from src.utils.datasets import DatasetProvider

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

wrapper = "PSO"
wrapper_name = 'PSO'
dir = "C:/Users/MDudjak/Dropbox/Doktorski studij/Disertacija/FS-Doprinos/Experiment/reports/summary/"
save_dir = "C:/Users/MDudjak/Dropbox/Doktorski studij/Disertacija/FS-Doprinos/Results/Priprema/"
classifier = "SVM"
runs = 30
filename = '_'.join([wrapper, classifier, "_archive_combinations"])
archive_header, archive_data = CsvProcessor().read_file(filename=dir + filename)

wrapper_best_idx = 5
archive_best_idx = 18
datasets = DatasetProvider().get_processed_dataset_list()
sns.set_theme(style="whitegrid")
imbl_datasets = ['Clean2', 'Climate', 'German_credit', 'Uci_parkinsons', 'Uci_voice', 'Urbanland']
if archive_header is not None and archive_data is not None:
    results = {'skup podataka': [], 'F1': [], 'omotač': []}
    for i, dataset in enumerate(datasets):
        if dataset.name in imbl_datasets:
            best_fitnesses = [float(row[wrapper_best_idx]) for row in archive_data if row[0] == dataset.name]
            archive_fitnesses = [float(row[archive_best_idx]) for row in archive_data if row[0] == dataset.name]

            for f in best_fitnesses:
                results['skup podataka'].append(r"$\mathcal{D}_{" + str(i + 1) + "}$")
                results['F1'].append(f)
                results['omotač'].append(wrapper_name)

            for f in archive_fitnesses:
                results['skup podataka'].append(r"$\mathcal{D}_{" + str(i + 1) + "}$")
                results['F1'].append(f)
                results['omotač'].append(wrapper_name+"+A")

    df = pd.DataFrame(data=results)
    vp = sns.violinplot(x='skup podataka', y="F1", hue="omotač", palette="RdBu", data=df, split=True, scale='count',
                   inner='quartile')
    plt.ylim(bottom=0.3, top=1.02)
    legend = plt.legend(fancybox=False, framealpha=0.9, frameon=True, loc='lower left')
    legend.get_frame().set_linewidth(1.5)
    vp.set_xlabel('skup podataka',fontsize=15)
    vp.set_ylabel('F1', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params()
    plt.grid(b=True, linewidth=1.5, linestyle=':', alpha=0.8)
    plt.tight_layout()
    #plt.show()
    plt.savefig(wrapper_name + "_" + classifier + '_vp.pdf', format='pdf', dpi=300)
    plt.close()
