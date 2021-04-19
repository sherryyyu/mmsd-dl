import os
import sys
# print('Current working path is %s' % str(os.getcwd()))
from statistics import mean

sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

def parse_results(lines):
    all_results = {}
    for c in lines:
        result = c.split(': ')
        szr_type = result[0]
        szr_type = ' '.join(szr_type.split(','))
        num = float(result[1])
        num = "%.3f" % num
        all_results[szr_type] = num
    return all_results

def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    return parse_results(lines)


acc = read_file("szr_ACC_single_wrst.txt")
eda = read_file("szr_EDA_single_wrst.txt")
bvp = read_file("szr_BVP_single_wrst.txt")
acc_eda = read_file("szr_ACC_EDA_single_wrst.txt")
acc_bvp = read_file("szr_ACC_BVP_single_wrst.txt")
eda_bvp = read_file("szr_EDA_BVP_single_wrst.txt")
acc_bvp_eda = read_file("szr_ACC_BVP_EDA_single_wrst.txt")

szr_types = []
acc_data = []
eda_data = []
bvp_data = []
acc_bvp_data = []
acc_bvp_eda_data = []
for k in sorted(acc.keys()):
    szr_types.append(k)
    acc_data.append(float(acc[k]))
    eda_data.append(float(eda[k]))
    bvp_data.append(float(bvp[k]))
    acc_bvp_data.append(float(acc_bvp[k]))
    acc_bvp_eda_data.append(float(acc_bvp_eda[k]))
    
    # print(k+','+acc[k]+','+eda[k]+','+bvp[k]+','+acc_bvp[k]+','+eda_bvp[k]+','+acc_eda[k]+','+acc_bvp_eda[k])


# szr_types = [
# 'Focal to Bilateral\nTonic-Clonic',
# 'Focal Tonic',
# 'Focal\nSubclinical',
# 'Focal\nAutomatisms',
# 'Focal\nBehavior Arrest',
# 'Focal Clonic',
# 'Generalized\nEpileptic Spasms',
# 'Generalized Tonic',
# 'Generalized\nTonic-Clonic'
# ]
print(szr_types)
szr_types = ['Focal Motor Atonic', 'Focal Motor Automatisms', 'Focal Motor Clonic', 'Focal to Bilateral Tonic-Clonic', 'Focal Motor Hyperkinetic', 'Focal Motor Myoclonic',
            'Focal Motor Tonic', 'Focal Non-motor Autonomic', 'Focal Non-motor Behavior Arrest', 'Focal Non-motor Cognitive', 'Focal Non-motor Sensory', 'Focal Non-motor Unclassified', 
            'Focal Off-camera', 'Focal Subclinical', 'Focal Unclassified', 
            'Generalized Motor Atonic', 'Generalized Motor Clonic', 'Generalized Motor Epileptic Spasms', 'Generalized Motor Myoclonic', 
            'Generalized Motor Tonic', 'Generalized Motor Tonic-clonic', 'Generalized Motor Unclassified', 
            'Generalized Non-motor Typical', 'Generalized Off-camera', 'Generalized Subclinical', 'Generalized Unclassified', 'Unknown/Unclassified']

pt_numbers = [1,11,6,21,4,3,15,1,11,1,2,1,6,14,5,2,2,8,3,15,6,1,2,4,2,6,1]

szr_numbers = [5,26,10,38,9,14,67,50,21,4,6,1,7,66,10,5,7,47,9,90,15,1,16,5,2,16,1]


# import numpy as np
# sort_idx = np.argsort(eda_data).tolist()
# # pt_numbers_sorted = pt_numbers[sort_idx]
# acc_bvp_data = [acc_bvp_data[i] for i in sort_idx]
# acc_data = [acc_data[i] for i in sort_idx]
# eda_data = [eda_data[i] for i in sort_idx]
# bvp_data = [bvp_data[i] for i in sort_idx]
# szr_numbers = [szr_numbers[i] for i in sort_idx]
# szr_types = [szr_types[i] for i in sort_idx]

# fig1=plt.figure(figsize=(10, 8))
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212, sharex = ax1)
fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8),
                            gridspec_kw={
                        #    'width_ratios': [2, 1],
                           'height_ratios': [2, 1]})

# fig1 = plt.figure()

overall_acc = 0.720
overall_eda = 0.549
overall_bvp = 0.744
overall_acc_bvp = 0.752


ax1.plot(acc_data,'y-.o',label='ACC, overall ='+'%4.3f' % overall_acc)
ax1.plot(eda_data,'g-.o',label='EDA, overall ='+'%4.3f' % overall_eda)
ax1.plot(bvp_data,'b-.o',label='BVP, overall ='+'%4.3f' % overall_bvp)
ax1.plot(acc_bvp_data,'r-o',label='ACC BVP, overall ='+'%4.3f' % overall_acc_bvp)
# plt.plot(acc_bvp_eda_data,'b-o',label='ACC BVP EDA, avg='+'%4.3f' % mean(acc_bvp_eda_data))
# plt.plot(m2_acc,'r-.o',label='ACC Specific, avg='+'%4.3f' % mean(m2_acc))
# plt.plot(m1_bvp,'g-o',label='BVP Generalized, avg='+'%4.3f' % mean(m1_bvp))
# plt.plot(m2_bvp,'g-.o',label='BVP Specific, avg='+'%4.3f' % mean(m2_bvp))
# plt.plot(m1_eda,'b-o',label='EDA Generalized, avg='+'%4.3f' % mean(m1_eda))
# plt.plot(m2_eda,'b-.o',label='EDA Specific, avg='+'%4.3f' % mean(m2_eda))
# ax1.xticks(range(27),szr_types, rotation=45, fontsize=12, ha='right')
ax1.legend()
ax1.set_ylabel('a) AUC-ROC')
# ax1.tight_layout()
ax1.grid(True)
# plt.xticks(rotation=45)
ax2.bar(szr_types, szr_numbers)
ax2.set_ylabel('b) Number of Seizures')
ax2.grid(True)
ax2.set_xticklabels(szr_types, rotation=45, fontsize=12, ha='right')

fig_name = 'method comparison.png'
fig_name_pdf = 'method comparison.pdf'
fig1.savefig(os.path.join( fig_name))
fig1.savefig(os.path.join(fig_name_pdf), bbox_inches='tight')

plt.show()