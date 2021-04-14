import os
import sys
# print('Current working path is %s' % str(os.getcwd()))
from statistics import mean

sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

eda_shift120 = [0.549,
    0.712,
    0.570,
    0.550,
    0.728,
    0.415,
    0.268,
    0.480,
    0.507,
    0.830]

eda = [0.516,
0.526,
0.575,
0.531,
0.713,
0.490,
0.264,
0.582,
0.478,
0.509
]

# szr_types = ['ALL',
# 'Focal to Bilateral Tonic-Clonic',
# 'Focal Tonic',
# 'Focal Subclinical',
# 'Focal Automatisms',
# 'Focal Behavior Arrest',
# 'Focal Clonic',
# #'Generized Epileptic Spasms',
# 'Generized Tonic',
# 'Generized Tonic-Clonic'
# ]

szr_types = [
'All Seizures',
'Focal to Bilateral\nTonic-Clonic',
'Focal Tonic',
'Focal\nSubclinical',
'Focal\nAutomatisms',
'Focal\nBehavior Arrest',
'Focal Clonic',
'Generalized\nEpileptic Spasms',
'Generalized Tonic',
'Generized\nTonic-Clonic'
]

fig1 = plt.figure(figsize=(10, 4))
plt.plot(eda,'-x',label='Original EDA, avg='+'%4.3f' % mean(eda))
plt.plot(eda_shift120,'-x',label='EDA shift 120 Second, avg='+'%4.3f' % mean(eda_shift120))
plt.xticks(range(10),szr_types, rotation='vertical', fontsize=12)
plt.legend()
plt.ylabel('AUC-ROC')
plt.tight_layout()
plt.grid(True)

fig_name = 'eda_delay.png'
fig_name_pdf = 'eda_delay.pdf'
fig1.savefig(os.path.join('results', fig_name))
fig1.savefig(os.path.join('results', fig_name_pdf), bbox_inches='tight')

plt.show()