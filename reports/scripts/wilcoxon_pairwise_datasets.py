from src.utils.file_handling.processors import CsvProcessor
from src.utils.datasets import DatasetProvider

wrapper = "GAMu_variable"
dir = "C:/Users/MDudjak/Dropbox/Doktorski studij/Disertacija/FS-Doprinos/Experiment/reports/summary/"
save_dir = "C:/Users/MDudjak/Dropbox/Doktorski studij/Disertacija/FS-Doprinos/Results/Priprema/"
classifier = "SVM"
runs = 30
filename = '_'.join([wrapper, classifier, "_archive_combinations"])
archive_header, archive_data = CsvProcessor().read_file(filename=dir + filename)

wrapper_best_idx = 5
archive_best_idx = 18
datasets = DatasetProvider().get_processed_dataset_list()

if archive_header is not None and archive_data is not None:
    for dataset in datasets:
        best_fitnesses = [float(row[wrapper_best_idx]) for row in archive_data if row[0] == dataset.name]
        archive_fitnesses = [float(row[archive_best_idx]) for row in archive_data if row[0] == dataset.name]

        with open(save_dir + wrapper + "_" + classifier + "/" + dataset.name + ".txt", "w") as file:
            for idx, f in enumerate(best_fitnesses):
                if idx == runs - 1:
                    file.write(str(f))
                else:
                    file.write(str(f) + '\n')

        with open(save_dir + wrapper + "_" + classifier + "_Arhiva/" + dataset.name + ".txt", "w") as file:
            for idx, f in enumerate(archive_fitnesses):
                if idx == runs - 1:
                    file.write(str(f))
                else:
                    file.write(str(f) + '\n')


