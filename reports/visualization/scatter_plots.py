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


GA_1NN_F1 = [-0.004451376, 0.005174053, 0.044373444, 0.146331906, -0.035717429, 0.044891776, 0.312982012, 0.03252797,
     0.006807331, -0.066823238, 0.036344617, 0.018921927]
GA_1NN_TPR = [-0.006046014, 0.00453529, 0.033430254, 0.12806696, -0.030444444, 0.050669643, 0.311025641, 0.032142857, 0.01021021,
     -0.068903319, 0.035701653, 0.013580247]

plt.scatter(GA_1NN_F1, GA_1NN_TPR, alpha=0.7, marker='o', color='royalblue')

DE_1NN_F1 = [-0.001370727,0.001323421,0.042802068,0.152724854,-0.002823569,0.049614372,0.109757649,0.028525807,0.001833809,-0.023181609,0.02383085,0.015876289]
DE_1NN_TPR = [-0.00176458,0.00033543,0.033483462,0.141926742,0.001619048,0.054985119,0.108102564,0.027636054,0.007057057,-0.02027417,0.021823113,0.012283951]
plt.scatter(DE_1NN_F1, DE_1NN_TPR, alpha=0.7, marker='s', color='orange')

PSO_1NN_F1 = [-0.004265108,-0.001850276,0.01589695,0.13048069,-0.011980591,0.046928734,0.020394824,0.019252184,-0.003253579,-0.012511245,0.007242735,0.005528019]
PSO_1NN_TPR = [-0.004684858,-0.001481481,0.00831895,0.121356464,-0.008412698,0.051041667,0.018974359,0.017857143,0.002927928,-0.01024531,0.007534877,0.001419753]

plt.scatter(PSO_1NN_F1, PSO_1NN_TPR, alpha=0.7, marker='D', color='limegreen')

labels = [r'GA', r'DE', r'PSO']
plt.xlim([-0.1, 0.4])
plt.ylim([-0.1, 0.4])
plt.xlabel(r'F1\textsubscript{R} - F1\textsubscript{O}', fontsize=15)
plt.ylabel(r'$TPR\textsubscript{R} - TPR\textsubscript{O}$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')

ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # horizontal lines
ax.axvline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # vertical lines

#plt.show()
plt.savefig('F1_TPR_1NN.pdf', format='pdf', dpi=300)
plt.close()


GA_5NN_F1 = [0.001675908,-0.006109205,0.019584001,0.087454286,0.015072231,0.065047182,0.323668271,0.011966508,0.027334764,0.045664388,0.028063305,-0.000551858]
GA_5NN_TPR = [0.000333868,-0.006254368,0.015524295,0.078594064,0.018222222,0.067782738,0.320615385,0.009013605,0.031043544,0.051370851,0.035707497,-0.002283951]

plt.scatter(GA_5NN_F1, GA_5NN_TPR, alpha=0.7, marker='o', color='royalblue')

DE_5NN_F1 = [0.002619194,-0.00834077,0.019835236,0.104404336,0.020580732,0.06397229,0.159308505,0.009607801,0.037538321,0.02690406,0.016299439,-0.001009475]
DE_5NN_TPR = [0.001276619,-0.008179595,0.015118843,0.097548161,0.021333333,0.065699405,0.156512821,0.006802721,0.038813814,0.032323232,0.020653667,-0.00191358]
plt.scatter(DE_5NN_F1, DE_5NN_TPR, alpha=0.7, marker='s', color='orange')

PSO_5NN_F1 = [-0.000282604,-0.015234916,0.010891698,0.094515156,0.024454632,0.043228027,0.025284614,0.017203049,0.022212582,-0.00315333,-0.007491504,0.00504109]
PSO_5NN_TPR = [-0.001373997,-0.015377358,0.008382781,0.082796593,0.023873016,0.043675595,0.022769231,0.013860544,0.025525526,-0.001154401,-0.003627778,0.003395062]

plt.scatter(PSO_5NN_F1, PSO_5NN_TPR, alpha=0.7, marker='D', color='limegreen')

labels = [r'GA', r'DE', r'PSO']
plt.xlim([-0.1, 0.4])
plt.ylim([-0.1, 0.4])
plt.xlabel(r'F1\textsubscript{R} - F1\textsubscript{O}', fontsize=15)
plt.ylabel(r'$TPR\textsubscript{R} - TPR\textsubscript{O}$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')

ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # horizontal lines
ax.axvline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # vertical lines

#plt.show()
plt.savefig('F1_TPR_5NN.pdf', format='pdf', dpi=300)
plt.close()

GA_GNB_F1 = [0.122148586,0.019822126,0.044234571,0.016447978,0.07760209,0.021041721,0.007827046,0.148415539,0.064522424,0.175812165,0.026830733,-0.010229356]
GA_GNB_TPR = [0.073245586,0.020779175,0.005068945,0.021336,0.01568254,0.03452381,0.008615385,0.140731293,0.003791291,0.116594517,0.02369799,-0.010555556]

plt.scatter(GA_GNB_F1, GA_GNB_TPR, alpha=0.7, marker='o', color='royalblue')

DE_GNB_F1 = [0.120382766,0.021535384,0.053413562,0.027743503,0.080603779,0.013194024,0.008233052,0.147416486,0.058916524,0.186982163,0.030754464,-0.015601788]
DE_GNB_TPR = [0.070182986,0.02129979,0.018848096,0.03061675,0.020857143,0.026785714,0.008512821,0.139370748,0.002702703,0.128932179,0.029642146,-0.015061728]
plt.scatter(DE_GNB_F1, DE_GNB_TPR, alpha=0.7, marker='s', color='orange')

PSO_GNB_F1 = [0.08924881,0.018123762,0.031063052,0.01206972,0.079989313,0.018756812,0.004023936,0.148906118,0.07859387,0.122046844,0.011148655,-0.0178026]
PSO_GNB_TPR = [0.056364901,0.018577918,0.006344336,0.015492951,0.020666667,0.03139881,0.004102564,0.143282313,0.028115616,0.083838384,0.011536675,-0.017283951]

plt.scatter(PSO_GNB_F1, PSO_GNB_TPR, alpha=0.7, marker='D', color='limegreen')

labels = [r'GA', r'DE', r'PSO']
plt.xlim([-0.1, 0.4])
plt.ylim([-0.1, 0.4])
plt.xlabel(r'F1\textsubscript{R} - F1\textsubscript{O}', fontsize=15)
plt.ylabel(r'$TPR\textsubscript{R} - TPR\textsubscript{O}$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')

ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # horizontal lines
ax.axvline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # vertical lines

#plt.show()
plt.savefig('F1_TPR_GNB.pdf', format='pdf', dpi=300)
plt.close()


GA_SVM_F1 = [-0.003950688,-0.006849413,0.00080812,0.252142377,0.018107291,-0.014232424,0.190905522,-0.002602515,0.036951692,0.205604297,0.028356423,-0.013138812]
GA_SVM_TPR = [-0.002317817,-0.006551363,0.003924505,0.184779145,0.020349206,-0.013244048,0.191128205,-0.001445578,0.048798799,0.169480519,0.025834915,-0.013148148]

plt.scatter(GA_SVM_F1, GA_SVM_TPR, alpha=0.7, marker='o', color='royalblue')

DE_SVM_F1 = [-0.005554679,-0.009267952,0.011803326,0.240553679,0.015426807,-0.004442232,0.036057933,-0.006642053,0.017071869,0.209968339,0.022485205,-0.01340383]
DE_SVM_TPR = [-0.004187266,-0.009769392,0.019493359,0.173963474,0.015396825,-0.003497024,0.036307692,-0.005782313,0.031156156,0.171572872,0.01966612,-0.012962963]
plt.scatter(DE_SVM_F1, DE_SVM_TPR, alpha=0.7, marker='s', color='orange')

PSO_SVM_F1 = [-0.006332758,-0.006002004,0.00219566,0.239571831,0.00395516,-0.004313688,-0.005845086,0.000250655,0.019666628,0.19594647,0.012908984,-0.011001009]
PSO_SVM_TPR = [-0.005031568,-0.005607966,0.00396662,0.173438854,0.00447619,-0.004092262,-0.005794872,0,0.025075075,0.153318903,0.011264937,-0.011111111]

plt.scatter(PSO_SVM_F1, PSO_SVM_TPR, alpha=0.7, marker='D', color='limegreen')

labels = [r'GA', r'DE', r'PSO']
plt.xlim([-0.1, 0.4])
plt.ylim([-0.1, 0.4])
plt.xlabel(r'F1\textsubscript{R} - F1\textsubscript{O}', fontsize=15)
plt.ylabel(r'$TPR\textsubscript{R} - TPR\textsubscript{O}$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, which='both', color='grey', linewidth=0.6, linestyle='dashed', alpha=0.2)
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')

ax.axhline(0, linestyle='-', linewidth=1, alpha=0.7, color='grey') # horizontal lines
ax.axvline(0, linestyle='-', linewidth=1, alpha=0.7, color='grey') # vertical lines

#plt.show()
plt.savefig('F1_TPR_SVM.pdf', format='pdf', dpi=300)
plt.close()