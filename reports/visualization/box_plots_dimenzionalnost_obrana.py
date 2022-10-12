import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

red_1NN = [
    [0.530894309,0.507777778,0.547991968,0.548148148,0.485245902,0.623529412,0.504466667,0.605263158,0.548484848,0.514301075,0.532879819,0.41025641],
    [0.591056911,0.635555556,0.702610442,0.664814815,0.701092896,0.744117647,0.971133333,0.670175439,0.634848485,0.94655914,0.83446712,0.505128205],
    [0.495121951,0.501111111,0.556024096,0.546296296,0.495081967,0.620588235,0.504466667,0.603508772,0.543939394,0.509784946,0.532879819,0.407692308],
    [0.536585366,0.612222222,0.647590361,0.659259259,0.520765027,0.739215686,0.829333333,0.657894737,0.657575758,0.891397849,0.685034014,0.435897436],
    [0.500813008,0.501111111,0.521084337,0.52037037,0.492349727,0.537254902,0.500133333,0.529824561,0.507575758,0.502043011,0.514512472,0.384615385],
    [0.508943089,0.593333333,0.536144578,0.653703704,0.510928962,0.675490196,0.513266667,0.631578947,0.622727273,0.636774194,0.53968254,0.487179487]

]

for i in range(len(red_1NN)):
    red_1NN[i] = [value*100 for value in red_1NN[i]]

box = plt.boxplot(red_1NN,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([30, 100])
plt.ylabel(r'red. (\%)', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('1NN_red.pdf', format='pdf', dpi=300)
plt.close()

red_5NN = [
    [0.482926829,0.457777778,0.509036145,0.525925926,0.490163934,0.719607843,0.504933333,0.535087719,0.527272727,0.510107527,0.532426304,0.420512821],
    [0.568292683,0.654444444,0.711044177,0.688888889,0.759562842,0.816666667,0.972933333,0.61754386,0.703030303,0.954086022,0.861904762,0.507692308],
    [0.469918699,0.491111111,0.5,0.503703704,0.503825137,0.687254902,0.515333333,0.515789474,0.509090909,0.509032258,0.515873016,0.441025641],
    [0.522764228,0.68,0.656425703,0.687037037,0.584153005,0.785294118,0.872266667,0.626315789,0.681818182,0.918494624,0.739002268,0.507692308],
    [0.456097561,0.455555556,0.496586345,0.477777778,0.476502732,0.618627451,0.500466667,0.485964912,0.518181818,0.505376344,0.502721088,0.441025641],
    [0.487804878,0.582222222,0.506425703,0.655555556,0.52295082,0.715686275,0.518266667,0.587719298,0.663636364,0.688709677,0.526303855,0.471794872]
]

for i in range(len(red_5NN)):
    red_5NN[i] = [value*100 for value in red_5NN[i]]

box = plt.boxplot(red_5NN,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([30, 100])
plt.ylabel(r'red. (\%)', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('5NN_red.pdf', format='pdf', dpi=300)
plt.close()

red_GNB = [
    [0.582926829,0.622222222,0.563453815,0.45,0.491256831,0.62254902,0.512866667,0.610526316,0.774242424,0.529032258,0.563945578,0.453846154],
    [0.662601626,0.742222222,0.771285141,0.646296296,0.720765027,0.720588235,0.953266667,0.638596491,0.834848485,0.945483871,0.845578231,0.469230769],
    [0.568292683,0.597777778,0.552811245,0.425925926,0.506557377,0.616666667,0.506333333,0.6,0.734848485,0.519032258,0.570975057,0.451282051],
    [0.618699187,0.712222222,0.663855422,0.62037037,0.589071038,0.729411765,0.806666667,0.626315789,0.813636364,0.924408602,0.778231293,0.494871795],
    [0.492682927,0.552222222,0.514257028,0.451851852,0.492896175,0.548039216,0.510466667,0.58245614,0.674242424,0.508709677,0.531519274,0.456410256],
    [0.521138211,0.658888889,0.530923695,0.601851852,0.533333333,0.65,0.523266667,0.614035088,0.777272727,0.872150538,0.578684807,0.456410256]
]

for i in range(len(red_GNB)):
    red_GNB[i] = [value*100 for value in red_GNB[i]]

box = plt.boxplot(red_GNB,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([30, 100])
plt.ylabel(r'red. (\%)', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('GNB_red.pdf', format='pdf', dpi=300)
plt.close()

red_SVM = [
    [0.529268293,0.504444444,0.497791165,0.57962963,0.51420765,0.618627451,0.511933333,0.531578947,0.593939394,0.539784946,0.563265306,0.38974359],
    [0.634146341,0.668888889,0.719879518,0.662962963,0.740983607,0.694117647,0.9546,0.578947368,0.796969697,0.962258065,0.850340136,0.438461538],
    [0.525203252,0.514444444,0.500200803,0.572222222,0.497814208,0.605882353,0.5108,0.529824561,0.590909091,0.530537634,0.560090703,0.438461538],
    [0.591869919,0.665555556,0.641967871,0.638888889,0.568852459,0.697058824,0.811333333,0.566666667,0.786363636,0.933225806,0.763038549,0.441025641],
    [0.48699187,0.503333333,0.463855422,0.524074074,0.482513661,0.544117647,0.511066667,0.456140351,0.531818182,0.517096774,0.53015873,0.461538462],
    [0.547154472,0.617777778,0.501004016,0.646296296,0.538797814,0.633333333,0.527266667,0.543859649,0.766666667,0.822473118,0.567120181,0.448717949]
]

for i in range(len(red_SVM)):
    red_SVM[i] = [value*100 for value in red_SVM[i]]


box = plt.boxplot(red_SVM,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([30, 100])
plt.ylabel(r'red. (\%)', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('SVM_red.pdf', format='pdf', dpi=300)
plt.close()


red_napredni = [
[0.42601626,0.425555556,0.840361446,0.5,0.430054645,0.640196078,0.396333333,0.58245614,0.471212121,0.490107527,0.493197279,0.341025641],
[0.465853659,0.56,0.886546185,0.661111111,0.485245902,0.739215686,0.6056,0.652631579,0.648484848,0.68172043,0.539909297,0.456410256],
[0.313821138,0.423333333,0.276907631,0.396296296,0.368852459,0.85,0.431,0.464912281,0.481818182,0.604946237,0.447619048,0.335897436],
[0.363414634,0.524444444,0.290763052,0.616666667,0.410382514,0.857843137,0.455,0.547368421,0.589393939,0.656451613,0.470521542,0.502564103],
[0.57398374,0.63,0.513253012,0.740740741,0.528961749,0.620588235,0.504266667,0.650877193,0.765151515,0.619784946,0.553741497,0.643589744],
[0.637398374,0.765555556,0.937550201,0.672222222,0.853551913,0.779411765,0.971,0.654385965,0.865151515,0.948064516,0.86984127,0.523076923]
]

for i in range(len(red_napredni)):
    red_napredni[i] = [value*100 for value in red_napredni[i]]


box = plt.boxplot(red_napredni,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([25, 100])
plt.ylabel(r'red. (\%)', fontsize=25)
plt.xticks(range(1,7), [r'PSO\textsubscript{D}', r'PSO\textsubscript{D}+A', r'PSO(4-2)', r'PSO(4-2)+A', r'EGAFS', r'EGAFS+A'], fontsize=12)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('red_napredni.pdf', format='pdf', dpi=300)
plt.close()
