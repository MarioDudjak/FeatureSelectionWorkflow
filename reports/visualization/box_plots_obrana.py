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
    [0.06034998,0.052267362,0.095965009,0.190973363,0.050943372,0.065342855,0.013159273,0.14482557,0.050368297,0.007868754,0.03606115,0.061355732],
    [0.087158751,0.117281907,0.168675863,0.409038497,0.084181391,0.1360474,0.647365273,0.209403057,0.149214311,0.076322486,0.163240229,0.194017094],
    [0.036758023,0.0420164,0.097284037,0.201707718,0.036062469,0.065965115,0.015938303,0.128949241,0.045855434,0.00630848,0.039598051,0.084606543],
    [0.049927763,0.084463614,0.146805484,0.358638935,0.048682875,0.158669199,0.168651056,0.213631012,0.148680815,0.070773022,0.053433481,0.16495137],
    [0.010550438,0.01591443,0.025474596,0.153895275,0.022386369,0.030275231,0.006677004,0.087627834,0.036717006,-0.00051286,0.01233307,0.057297377],
    [0.029876357,0.024375873,0.029688744,0.335800036,0.021664988,0.102703311,0.007248539,0.185409069,0.087879617,0.007298487,0.016581244,0.141429413]
]
box = plt.boxplot(ASM_1NN,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([0, 0.7])
plt.ylabel(r'ASM', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('1NN_ASM.pdf', format='pdf', dpi=300)
plt.close()

ASM_5NN = [
    [0.058617984,0.031400617,0.058270659,0.250893997,0.091728102,0.131636939,0.020648256,0.172545828,0.037470186,0.001852617,0.039221566,0.109554966],
    [0.088259932,0.079113237,0.13033937,0.468791279,0.146547115,0.221851047,0.646276924,0.285085895,0.100928165,0.113174498,0.177281783,0.195558503],
    [0.045919862,0.028637773,0.060254205,0.252993979,0.080803661,0.096296967,0.017772432,0.158423165,0.0464124,0.00413501,0.038672514,0.096867079],
    [0.072668561,0.090025173,0.08846726,0.437968436,0.093092996,0.180241652,0.228267206,0.237889311,0.099992287,0.056550636,0.087722256,0.213330386],
    [0.021166272,0.020814282,0.021783987,0.157557927,0.035261128,0.063852085,0.006994109,0.129595733,0.008508401,0.002930689,0.015789067,0.106324786],
    [0.042796443,0.034226368,0.02526213,0.336976829,0.040187605,0.101570735,0.007564004,0.190601888,0.046638138,0.012219966,0.016694335,0.174921898]

]
box = plt.boxplot(ASM_5NN,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([0, 0.7])
plt.ylabel(r'ASM', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('5NN_ASM.pdf', format='pdf', dpi=300)
plt.close()

ASM_GNB = [
    [0.187323292,0.171183331,0.087676498,0.091207809,0.068939842,0.132527647,0.011831549,0.251812385,0.276353768,0.006520817,0.044320392,0.142455055],
    [0.230293587,0.27535113,0.166691674,0.346359241,0.154454712,0.262407517,0.070441019,0.392779986,0.354120515,0.131355299,0.151740024,0.304017094],
    [0.198525971,0.191371387,0.107064955,0.094103266,0.070122773,0.129619084,0.010777369,0.244994911,0.249949578,0.006309513,0.050296224,0.15423519],
    [0.233815752,0.300843824,0.140700402,0.313482941,0.109353046,0.230053509,0.03492998,0.344418518,0.265531671,0.117589698,0.123209208,0.289537283],
    [0.101810317,0.125053165,0.04181644,0.102988506,0.04396654,0.075818272,0.004120763,0.198215942,0.179631371,-0.000492459,0.023351451,0.124509284],
    [0.109747383,0.188221025,0.047467016,0.241580916,0.061255349,0.142130989,0.005004181,0.318497297,0.20484749,0.053286709,0.028577132,0.305452402]

]
box = plt.boxplot(ASM_GNB,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([0, 0.7])
plt.ylabel(r'ASM', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('GNB_ASM.pdf', format='pdf', dpi=300)
plt.close()

ASM_SVM = [
    [0.087744884,0.042588752,0.079295248,0.285148695,0.078066075,0.10764956,0.008264699,0.172649296,0.09417948,0.019052577,0.03360192,0.087232538],
    [0.150438587,0.079588438,0.117753857,0.461913884,0.200981571,0.159110122,0.499646176,0.278659196,0.303419416,0.242748514,0.171275743,0.22930445],
    [0.07344939,0.050535384,0.086903959,0.283760263,0.075771223,0.083197528,0.016067539,0.164695743,0.102669553,0.016082145,0.048700897,0.050400825],
    [0.121686497,0.099626159,0.126280203,0.423898924,0.094937447,0.140711817,0.072354031,0.255088391,0.290982236,0.174582853,0.118716462,0.226607722],
    [0.03647385,0.015845802,0.019454412,0.19954844,0.034273648,0.03602766,0.0068189,0.104992846,0.049229902,0.003174083,0.019950852,0.051140584],
    [0.065406884,0.047776011,0.026107426,0.412923737,0.047826427,0.082234568,0.007687841,0.191386992,0.213683386,0.05843302,0.031575763,0.212180961]
]
box = plt.boxplot(ASM_SVM,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([0, 0.7])
plt.ylabel(r'ASM', fontsize=25)
plt.xticks(range(1,7), [r'GA', r'GA+A', r'DE', r'DE+A', r'PSO', r'PSO+A'], fontsize=20)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('SVM_ASM.pdf', format='pdf', dpi=300)
plt.close()


napredni_ASM = [
   [0.04070127,0.066737828,0.027539538,0.174412516,0.065494581,0.055872015,0.019128057,0.140218266,0.097195767,0.022144165,0.045068546,0.118408488],
    [0.045824431,0.091600328,0.038637807,0.357413793,0.040819665,0.11199588,0.014680699,0.244055973,0.134750129,0.026462035,0.043591293,0.15709107],
    [0.045242158, 0.066063132, 0.036192255, 0.19473545, 0.03389903, 0.144140185, 0.01515504, 0.10277072, 0.068704284,
     0.007857395, 0.02234709, 0.201343943],
    [0.050205561, 0.062833431, 0.032274283, 0.300426017, 0.034259806, 0.169985536, 0.015879215, 0.131860542,
     0.142204475, 0.015580845, 0.026813401, 0.208944887],
    [0.109649455, 0.13908548, 0.036917718, 0.142047072, 0.048514692, 0.095483907, 0.007696966, 0.228989188, 0.170928497, 0.002885497, 0.016915286, 0.192888299],
    [0.208272523, 0.31661102, 0.126233626, 0.357189381, 0.196691256, 0.270847473, 0.080492402, 0.388978289, 0.336408917,
     0.12158983, 0.112060376, 0.324571176],

]
box = plt.boxplot(napredni_ASM,   patch_artist=True)
#colors = ['lightblue', 'lightblue', 'navajowhite', 'navajowhite', 'lightgray', 'lightgray']
colors = ['lightblue', 'navajowhite', 'lightblue', 'navajowhite', 'lightblue', 'navajowhite']


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#plt.xlim([-0.1, 0.4])
plt.ylim([0, 0.7])
plt.ylabel(r'ASM', fontsize=25)
plt.xticks(range(1,7), [r'PSO\textsubscript{D}', r'PSO\textsubscript{D}+A', r'PSO(4-2)', r'PSO(4-2)+A', r'EGAFS', r'EGAFS+A'], fontsize=12)
plt.yticks(fontsize=25)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, linestyle=':', alpha=0.6)
#plt.legend(labels=[r'GA', r'DE', r'PSO'], fancybox=False, framealpha=0.9, ncol=3)

#plt.show()
plt.savefig('napredni_ASM.pdf', format='pdf', dpi=300)
plt.close()