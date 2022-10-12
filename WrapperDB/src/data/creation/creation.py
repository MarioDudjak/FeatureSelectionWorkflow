import matplotlib.pyplot as plt
import numpy as np

from src.data.creation.setup import training_size, imbalance_degree, noise_level, small_disjuncts_complexity, \
    class_overlapping_degree

from src.utils.file_handling.processors import CsvProcessor

np.random.seed(42)
figures_path = r"C:\Users\MarioPC\PycharmProjects\DataScienceWorkflow\reports\figures\\"

for ir_degree, ir in imbalance_degree.items():
    for ts_level, ts in training_size.items():
        no_minority_instances = int(ts * (1 - ir))
        no_majority_instances = int(ts * ir)

        for sd_degree, sd in small_disjuncts_complexity.items():
            for co_degree, co in class_overlapping_degree.items():
                for noise_degree, noise in noise_level.items():
                    x1 = []
                    y1 = []
                    x2 = []
                    y2 = []
                    filename = 'IR-' + str(ir) + '-DS-' + str(ts) + '-SD-' + str(sd) + '-CO-' + str(co) + '-N-' + str(
                        noise)
                    print("Processing file {0}".format(filename))
                    fig, ax = plt.subplots()

                    x_boundaries = np.sort(np.random.rand(2 * sd))
                    y_boundaries = np.sort(np.random.rand(2 * sd))

                    low = 0
                    for i, point in enumerate(np.concatenate([x_boundaries, [1]])):
                        if i % 2 == 0:
                            x1 = np.concatenate(
                                [x1,
                                 np.random.uniform(low=low, high=point, size=int(no_majority_instances // (sd + 1)))])
                        else:
                            x2 = np.concatenate([x2, np.random.uniform(low=low - (co / (2 * sd)),
                                                                       high=point + (co / (2 * sd)),
                                                                       size=int(no_minority_instances // sd))])
                        low = point

                    low = 0
                    for i, point in enumerate(np.concatenate([y_boundaries, [1]])):
                        if i % 2 == 0:
                            y1 = np.concatenate(
                                [y1,
                                 np.random.uniform(low=low, high=point, size=int(no_majority_instances // (sd + 1)))])
                        else:
                            y2 = np.concatenate([y2, np.random.uniform(low=low - (co / (2 * sd)),
                                                                       high=point + (co / (2 * sd)),
                                                                       size=int(no_minority_instances // sd))])
                        low = point

                    # Injecting noise - changing %instances to different class label
                    random_indexes = np.random.randint(low=0, high=ts, size=int(noise * ts))
                    for idx in random_indexes:
                        if idx >= no_majority_instances and len(x2) > 0 and len(y2) > 0:
                            i = np.random.randint(low=0, high=len(x2))
                            x1 = np.append(x1, x2[i])
                            x2 = np.delete(x2, i)
                            y1 = np.append(y1, y2[i])
                            y2 = np.delete(y2, i)
                        else:
                            i = np.random.randint(low=0, high=len(x1))
                            x2 = np.append(x2, x1[i])
                            x1 = np.delete(x1, i)
                            y2 = np.append(y2, y1[i])
                            y1 = np.delete(y1, i)

                    ax.scatter(x1, y1, c='red', label='negative', alpha=0.7, edgecolor='none', marker='.')
                    ax.scatter(x2, y2, c='blue', label='positive', alpha=0.7, edgecolor='none', marker='.')

                    ax.legend(fontsize='medium', bbox_to_anchor=(0.212, 0.975), ncol=1, handletextpad=0.5,
                              handlelength=1.35)
                    ax.grid(True)
                    ax.set(xlim=(0, 1), ylim=(0, 1))
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    #plt.show()
                    plt.savefig(figures_path + filename + ".pdf", dpi=300)
                    plt.close()

                    X = np.column_stack((x1, y1))
                    X = np.append(X, np.column_stack((x2, y2)), axis=0)

                    y = np.full((len(x1)), 0, dtype=int)
                    y = np.append(y, np.full((len(x2)), 1, dtype=int))

                    data = np.column_stack((X, y))
                    CsvProcessor().save_synthetic_datasets(filename=filename, header=['feature1', 'feature2', 'label'], data=data)

