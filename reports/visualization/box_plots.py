import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

ASM_1NN = [
    [0.394927954,0.288023862,0.168669105,0.311586967,0.255885696,0.39622702,0.089886409,0.396401425,0.320688859,0.105005187,0.153134934,0.212321542],
    [0.468458223,0.329989251,0.41696324,0.304579786,0.471559353,0.414854318,0.275117518,0.389174969,0.344626625,0.244092775,0.389327198,0.084606543],
    [0.491222414,0.42688882,0.517554562,0.368092349,0.515786748,0.464878849,0.512661761,0.408033529,0.413216041,0.46033074,0.513792268,0.282319874]
]
box = plt.boxplot(ASM_1NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'ASM', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('ASM-1NN.pdf', format='pdf', dpi=300)
plt.close()

Presjek_1NN = [
    [7.8,2.466666667,2.033333333,2.266666667,3.433333333,2.366666667,0.766666667,3.333333333,2.466666667,1.233333333,0.6,0.533333333],
    [14.26666667,4.266666667,36.4,2.1,22.9,3.8,61.36666667,2.966666667,2.633333333,21.1,29.56666667,0.566666667],
    [14.86666667,6.866666667,71.9,2.566666667,23.53333333,7.166666667,237.2333333,3.366666667,4.333333333,107.3666667,62.46666667,1.166666667]
]
box = plt.boxplot(Presjek_1NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (presjek)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Presjek-1NN.pdf', format='pdf', dpi=300)
plt.close()

Nule_1NN = [
    [10.76666667,3.8,10.53333333,0.533333333,3.466666667,7.666666667,0.066666667,6.066666667,4.2,2.933333333,3.333333333,0.033333333],
    [14.9,4.6,50.03333333,0.366666667,21.76666667,8.8,62,5.533333333,3.8,23.8,40.23333333,0],
    [14.83333333,6.7,78.56666667,1.966666667,23.36666667,10.46666667,237.8333333,4.033333333,4.366666667,108.3666667,66.93333333,0.233333333]
]
box = plt.boxplot(Nule_1NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (otpisane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Nule_1NN.pdf', format='pdf', dpi=300)
plt.close()

Dodavanje_1NN = [
    [8.966666667,8.466666667,47.33333333,3.766666667,14.8,6.333333333,13.66666667,2.933333333,5.566666667,15.33333333,23.73333333,5.9],
    [4.733333333,7.366666667,22.1,4.033333333,6.333333333,5.066666667,23.96666667,3.533333333,4.9,12.56666667,16.73333333,6.766666667],
    [5.266666667,5.333333333,5.1,3.666666667,6.3,3.866666667,6.133333333,3.633333333,3.966666667,5.233333333,5.2,5.5]
]
box = plt.boxplot(Dodavanje_1NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (dodane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Dodavanje_1NN.pdf', format='pdf', dpi=300)
plt.close()

F1_1NN = [
    [0.815098876,0.966790461,0.946108409,0.727999751,0.619998454,0.911585452,0.568258213,0.886720796,0.882393167,0.76117546,0.772260808,0.950799677],
    [0.813344616,0.96349414,0.948204536,0.713375653,0.619830432,0.919019737,0.578557834,0.886546753,0.885873073,0.763528425,0.775069453,0.950173839],
    [0.812329742,0.955444356,0.921654574,0.693862002,0.604484695,0.915112406,0.571550445,0.85962948,0.876686021,0.766193338,0.76683369,0.950080257]
]
box = plt.boxplot(F1_1NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'F1', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('F1_1NN.pdf', format='pdf', dpi=300)
plt.close()

'''5-NN'''

ASM_5NN = [
    [0.382332707,0.284120368,0.147122137,0.317702473,0.237121411,0.431486045,0.095079292,0.346130523,0.328749838,0.102835058,0.146080993,0.212321542],
    [0.435848747,0.28853334,0.359391094,0.358406144,0.463697556,0.392310579,0.287254428,0.304749512,0.345718752,0.207551242,0.377860628,0.096867079],
    [0.500373124,0.429640184,0.51161856,0.375249164,0.498703679,0.491787409,0.509555856,0.412678683,0.405118547,0.445790102,0.511144174,0.286424993]
]
box = plt.boxplot(ASM_5NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'ASM', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('ASM-5NN.pdf', format='pdf', dpi=300)
plt.close()

Presjek_5NN = [
    [9.3,2.266666667,1.866666667,2.2,2.533333333,1.6,2,3.966666667,3.466666667,0.9,0.833333333,0.533333333],
    [15.1,3.066666667,32.76666667,3.066666667,18.76666667,3.233333333,43.96666667,3.633333333,3.166666667,13.56666667,24.06666667,0.666666667],
    [15.9,7.1,74.6,3.133333333,23.76666667,5.633333333,235.1333333,4.5,4.133333333,90.6,64.63333333,1.233333333]
]
box = plt.boxplot(Presjek_5NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (presjek)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Presjek-5NN.pdf', format='pdf', dpi=300)
plt.close()

Nule_5NN = [
    [8.8,2.866666667,3.9,0.533333333,2.333333333,7.366666667,0.2,3.766666667,4.8,1.333333333,2.466666667,0.1],
    [13.86666667,3.666666667,37.53333333,0.766666667,20.33333333,10.5,43.86666667,3.033333333,3.966666667,15.96666667,29.13333333,0.1],
    [12.66666667,5.2,74.23333333,1.866666667,22.53333333,11.76666667,237.2666667,3.7,4.533333333,96.76666667,65.3,0.333333333]
]
box = plt.boxplot(Nule_5NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (otpisane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Nule_5NN.pdf', format='pdf', dpi=300)
plt.close()

Dodavanje_5NN = [
    [8.4,8.1,46.1,3.4,12.13333333,4.633333333,11.53333333,3.3,3.066666667,13.33333333,19.46666667,5.866666667],
    [4.466666667,6.533333333,24.26666667,2.566666667,6.6,4.066666667,19.9,3.466666667,3.833333333,11.7,14.3,5.733333333],
    [5.1,5.433333333,7.333333333,3.066666667,5.333333333,4.033333333,5.733333333,3.333333333,3.266666667,5.9,5,5.633333333]
]
box = plt.boxplot(Dodavanje_5NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (dodane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Dodavanje_5NN.pdf', format='pdf', dpi=300)
plt.close()

F1_5NN = [
    [0.847850281,0.970278403,0.939175547,0.693813881,0.65420939,0.880587751,0.597641362,0.829044163,0.860729089,0.795145845,0.772811323,0.942277372],
    [0.847170174,0.970341566,0.942302655,0.692071308,0.656160038,0.87539049,0.604511884,0.828047766,0.85135938,0.790688391,0.774740394,0.941262401],
    [0.842266669,0.969578842,0.933068592,0.644147874,0.638389751,0.85779915,0.585637832,0.819517205,0.841732837,0.786702322,0.763157249,0.943107834]
]
box = plt.boxplot(F1_5NN,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'F1', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('F1_5NN.pdf', format='pdf', dpi=300)
plt.close()

'''GNB'''

ASM_GNB = [
    [0.392054715,0.388066201,0.165462201,0.331804611,0.288575983,0.397630959,0.106568615,0.37267746,0.426895654,0.091360216,0.167525113,0.232747916],
    [0.435848747,0.28853334,0.359391094,0.358406144,0.463697556,0.392310579,0.287254428,0.304749512,0.345718752,0.207551242,0.377860628,0.096867079],
    [0.496555466,0.450083661,0.509176925,0.401597184,0.503304236,0.494430753,0.514820345,0.423736085,0.449980619,0.290721407,0.514168612,0.293179684]
]
box = plt.boxplot(ASM_GNB,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'ASM', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('ASM-GNB.pdf', format='pdf', dpi=300)
plt.close()

Presjek_GNB = [
    [4.4,2.766666667,3.933333333,3.766666667,5.6,3.933333333,0.9,2.5,0.466666667,1.8,1.733333333,0.566666667],
    [15.10,3.07,32.77,3.07,18.77,3.23,43.97,3.63,3.17,13.57,24.07,0.67],
    [15.13333333,5.8,71.43333333,4.233333333,23.83333333,8.733333333,233.5666667,3.2,1.933333333,29.86666667,57.43333333,0.733333333]
]
box = plt.boxplot(Presjek_GNB,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (presjek)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Presjek-GNB.pdf', format='pdf', dpi=300)
plt.close()

Nule_GNB = [
    [11.36666667,6.933333333,2.5,1.7,6.666666667,6.533333333,0.833333333,4.8,4.7,1.4,1.8,0.833333333],
    [13.87,3.67,37.53,0.77,20.33,10.50,43.87,3.03,3.97,15.97,29.13,0.10],
    [13.8,9.066666667,74.7,2.133333333,22.23333333,10.56666667,243.8666667,4.566666667,5.666666667,35.13333333,65.96666667,0.9]
]
box = plt.boxplot(Nule_GNB,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (otpisane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Nule_GNB.pdf', format='pdf', dpi=300)
plt.close()

Dodavanje_GNB = [
    [9.433333333,4.966666667,34.03333333,2.6,11.43333333,5.566666667,22.46666667,4.366666667,3.166666667,15.1,20.96666667,6.333333333],
    [4.47,6.53,24.27,2.57,6.60,4.07,19.90,3.47,3.83,11.70,14.30,5.73],
    [4.5,4.433333333,6.433333333,2.933333333,4.633333333,3.166666667,4.8,4.133333333,2.966666667,9.766666667,4.5,6.333333333]
]
box = plt.boxplot(Dodavanje_GNB,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (dodane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Dodavanje_GNB.pdf', format='pdf', dpi=300)
plt.close()

F1_GNB= [
    [0.764798928,0.964717621,0.774618448,0.718888107,0.663977817,0.897652223,0.597590426,0.840910461,0.701973456,0.631390805,0.822966166,0.955706683],
    [0.847170174,0.970341566,0.942302655,0.692071308,0.656160038,0.87539049,0.604511884,0.828047766,0.85135938,0.790688391,0.774740394,0.941262401],
    [0.749153198,0.961295329,0.762787738,0.713548855,0.641307363,0.890583974,0.591915258,0.822737232,0.691847516,0.615830394,0.822604818,0.953583998]
]
box = plt.boxplot(F1_GNB,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'F1', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('F1_GNB.pdf', format='pdf', dpi=300)
plt.close()

'''SVM'''

ASM_SVM = [
    [0.357880481,0.247370526,0.160826372,0.350677454,0.285308732,0.350696421,0.108852929,0.307100575,0.332558923,0.126420482,0.170069703,0.087232538],
    [0.44,0.29,0.36,0.36,0.46,0.39,0.29,0.30,0.35,0.21,0.38,0.05],
    [0.489606883,0.380793367,0.518570629,0.389581159,0.501701878,0.439720954,0.512845251,0.402370017,0.331718617,0.363294598,0.519871217,0.243042539]
]
box = plt.boxplot(ASM_SVM,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'ASM', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('ASM-SVM.pdf', format='pdf', dpi=300)
plt.close()

Presjek_SVM = [
    [7.6,1.633333333,2.333333333,3.666666667,4.766666667,2.833333333,0.5,3.433333333,2.266666667,2.166666667,1.133333333,0.1],
    [12,3.2,35.2,3.6,19.96666667,3.8,65.96666667,3.666666667,2.433333333,13.43333333,22.53333333,0.333333333],
    [13.93333333,5.333333333,77.53333333,3.833333333,23.56666667,6.966666667,232.2,4.3,2.533333333,48.56666667,59.13333333,0.733333333]
]
box = plt.boxplot(Presjek_SVM,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (presjek)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Presjek-SVM.pdf', format='pdf', dpi=300)
plt.close()

Nule_SVM = [
    [8.133,3,1.7,1.6,5.3,3.133,0.5,1.833,4.133,1.1677,2.533,0.266666667],
    [12,3.733,33.0667,1.7,20.533,6.8,70.4,2.033,4.133,17.0666,33.3667,0.06667],
    [12.533,5.1667,67.366,1.966,21.8,7.833,243.5667,2.4,3.1,53.9,67.33,0.4]
]
box = plt.boxplot(Nule_SVM,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (otpisane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Nule_SVM.pdf', format='pdf', dpi=300)
plt.close()

Dodavanje_SVM = [
    [7.4,8.3,44.16666667,2.4,11.03333333,7.566666667,22.2,4.566666667,2.2,9.533333333,20.86666667,7.2],
    [5,6.833333333,24.23333333,2.9,6.333333333,6.5,28.36666667,4.566666667,2.266666667,7.266666667,12.3,6.933333333],
    [4.633333333,6.133333333,5.3,2.533333333,4.566666667,5.5,4.166666667,4.366666667,2.6,6.466666667,4.5,6.433333333]
]
box = plt.boxplot(Dodavanje_SVM,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'br.\ zn.\ (dodane)', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('Dodavanje_SVM.pdf', format='pdf', dpi=300)
plt.close()

F1_SVM =  [
    [0.840475381,0.964858726,0.923691515,0.731696979,0.649842058,0.944561976,0.597189127,0.862786017,0.716460339,0.813758367,0.789072298,0.960924488],
    [0.840361406,0.965241225,0.925804386,0.726739311,0.652443941,0.946067413,0.601015483,0.86126651,0.708711015,0.811051079,0.790590325,0.959660448],
    [0.840227383,0.963707191,0.919136383,0.677400659,0.641387651,0.946030877,0.591010322,0.852686718,0.715093918,0.795085752,0.778877983,0.957336972]
]
box = plt.boxplot(F1_SVM,   patch_artist=True)
colors = ['lightblue', 'navajowhite', 'lightgray']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.ylabel(r'F1', fontsize=30)
plt.xticks(range(1,4), [r'GA', r'DE', r'PSO'], fontsize=30)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('F1_SVM.pdf', format='pdf', dpi=300)
plt.close()