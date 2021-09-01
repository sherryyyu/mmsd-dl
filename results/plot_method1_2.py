import os
import sys
# print('Current working path is %s' % str(os.getcwd()))
from statistics import mean

sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

m2_acc=[0.921,
0.786,
0.548,
0.688,
0.635,
0.516,
0.594,
0.588,
0.975
]
m1_acc= [0.919,
0.812,
0.555,
0.541,
0.765,
0.564,
0.840,
0.662,
0.995
]


m2_bvp=[0.888,
0.751,
0.496,
0.682,
0.706,
0.648,
0.627,
0.814,
0.904,

]
m1_bvp= [0.886,
0.736,
0.642,
0.811,
0.693,
0.830,
0.711,
0.779,
0.889,
]

m2_eda=[0.712,
0.570,
0.550,
0.728,
0.415,
0.268,
0.480,
0.507,
0.830,
]
m1_eda= [0.662,
0.624,
0.429,
0.699,
0.532,
0.588,
0.450,
0.565,
0.802,
]


szr_types = [
'Focal to Bilateral\nTonic-Clonic',
'Focal Tonic',
'Focal\nSubclinical',
'Focal\nAutomatisms',
'Focal\nBehavior Arrest',
'Focal Clonic',
'Generalized\nEpileptic Spasms',
'Generalized Tonic',
'Generalized\nTonic-Clonic'
]

fig1 = plt.figure(figsize=(10, 8))
plt.plot(m1_acc,'r-o',label='ACC Generalized, avg='+'%4.3f' % mean(m1_acc))
plt.plot(m2_acc,'r-.o',label='ACC Specific, avg='+'%4.3f' % mean(m2_acc))
plt.plot(m1_bvp,'g-o',label='BVP Generalized, avg='+'%4.3f' % mean(m1_bvp))
plt.plot(m2_bvp,'g-.o',label='BVP Specific, avg='+'%4.3f' % mean(m2_bvp))
plt.plot(m1_eda,'b-o',label='EDA Generalized, avg='+'%4.3f' % mean(m1_eda))
plt.plot(m2_eda,'b-.o',label='EDA Specific, avg='+'%4.3f' % mean(m2_eda))
plt.xticks(range(9),szr_types, rotation='vertical', fontsize=12)
plt.legend()
plt.ylabel('AUC-ROC')
plt.tight_layout()
plt.grid(True)

fig_name = 'method comparison.png'
fig_name_pdf = 'method comparison.pdf'
fig1.savefig(os.path.join('results', fig_name))
fig1.savefig(os.path.join('results', fig_name_pdf), bbox_inches='tight')

plt.show()