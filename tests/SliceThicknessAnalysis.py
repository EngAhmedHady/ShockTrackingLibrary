# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:25:38 2024

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"
path = []

mainDir = 'P1-DesignConditions\\'
path.append(f'{mainDir}2kHz\\UncertaintyRatio-P1-withRotation-[282, 435].txt')
path.append(f'{mainDir}3kHz\\UncertaintyRatio-P1-withRotation-[288, 431].txt')
path.append(f'{mainDir}6kHz\\UncertaintyRatio-P1-withRotation-[286, 433].txt')
path.append(f'{mainDir}10kHz\\UncertaintyRatio-P1-withRotation-[283, 440].txt')
path.append(f'{mainDir}15kHz\\UncertaintyRatio-P1-withRotation-[291, 426].txt')

path.append(f'{mainDir}2kHz\\UncertaintyRatio-P1-NoRotation-[282, 435].txt')
path.append(f'{mainDir}3kHz\\UncertaintyRatio-P1-NoRotation-[288, 431].txt')
path.append(f'{mainDir}6kHz\\UncertaintyRatio-P1-NoRotation-[286, 433].txt')
path.append(f'{mainDir}10kHz\\UncertaintyRatio-P1-NoRotation-[283, 440].txt')
path.append(f'{mainDir}15kHz\\UncertaintyRatio-P1-NoRotation-[291, 426].txt')


fig,ax = plt.subplots(figsize=(10,10),gridspec_kw={'hspace': 0.05})
fig1,ax1 = plt.subplots(figsize=(10,10),gridspec_kw={'hspace': 0.05})
mark = ['o','s','^','d','v','<','x','+','.', ',', 'v', '>']
Colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:gray']
Colors2 = ['tab:cyan','tab:red','tab:olive','tab:pink','k']
linstyl=['-','--',':','-.','dashdotdotted']
DfList = []
j = 0; k = 0
for i in path:
    Caseinfo = i.split('\\')
    CaseFeq = Caseinfo[1]
    RotationCond = Caseinfo[2].split('-')[2]
    RotationCond = 'Rotated' if RotationCond == 'withRotation' else 'NotRotated'
    Df = pd.read_csv(i, header=None)
    
    if RotationCond == 'Rotated':
        ax.plot(Df[0].iloc[3:],Df[1].iloc[3:],marker=mark[j],ms = 10, color = Colors[j],markerfacecolor='none',  label = CaseFeq)
        ax1.plot(Df[0],Df[3],marker=mark[j],ls = '-',ms = 10, color = Colors[j], markerfacecolor='none', label = f'{CaseFeq}-{RotationCond}')
        j += 1
        if j%5 == 0: j = 0
    else:
        ax1.plot(Df[0],Df[3],marker=mark[j],ls = '--', color = Colors2[j] ,ms = 10, markerfacecolor='none', label = f'{CaseFeq}-{RotationCond}')
        j += 1
    
        
    # if k%2 != 0: ls = '--'
    # else: ls = '-'
    
    
    
    # k += 1


# ax.set_xlabel(r"$U_x/U_{avg}$"); 
ax.set_xlabel("Slice thickness [px]");
ax1.set_xlabel("Slice thickness [px]");  
# ax.set_xlabel(r"M"); 
ax.set_ylabel("Estimated shock angle [deg]");
ax.yaxis.set_label_coords(-0.09, .5)
ax1.set_ylabel("Estimated uncertainity [\%]");
# ax1.yaxis.set_label_coords(-0.075, .5)
ax.yaxis.set_ticks(np.arange(90,115,5))
ax1.yaxis.set_ticks(np.arange(0,30,4))
# ax.xaxis.set_ticks(np.arange(350,440,20))
# ax.legend(['With Plate','Without plate'], fontsize="20")
# ax.legend(fontsize="18", bbox_to_anchor=(0.49, 1.17), loc='upper center', ncol = 5)
ax1.legend(fontsize="18", bbox_to_anchor=(0.49, 1.25), loc='upper center', ncol = 2)
ax.legend(fontsize="18", ncol = 1)
ax1.legend(fontsize="18", ncol = 2)


ax.set_ylim([90,110])
ax.set_xlim([0,140])
ax1.set_ylim([0,25])

ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.2, lw = 1.5)
ax.minorticks_on()
ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.1)

ax1.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.2, lw = 1.5)
ax1.minorticks_on()
ax1.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.1)