# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:25:38 2024

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

plt.rcParams.update({'font.size': 20})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"

def exp_fitting(x, A, seg, c):
    # return A*np.e**(-x/seg)
    return  A*x**(seg)-c

path = 'parametricAnalysis2.csv'
fig,ax = plt.subplots(figsize=(10,6),gridspec_kw={'hspace': 0.05})

mark = ['o','s','^','d','v','<','x','+','.', ',', 'v', '>']
linstyl=['-','--',':','-.','dashdotdotted']
j = 0; k = 0; 
Df = pd.read_csv(path)
for i in [2,3,6,10,15]:
    ax.plot(Df['N%'],Df[f"{i}kHz"],linestyle='None',marker=mark[j],ms = 10, markerfacecolor='none',  label = f"{i}kHz")
    j += 1

ax.set_xlabel(r"Samples to total images ($l/N$) [\%]");
ax.set_ylabel("Estimated angles standard diviation [deg]");
ax.legend(fontsize="18", ncol = 1)

ax.set_xlim([0,100]); ax.set_ylim([0,3])
ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.2, lw = 1.5)
ax.minorticks_on()
ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.1)

popt, pcov = optimize.curve_fit(exp_fitting, Df['N%'],Df["3kHz"], maxfev = 3000, p0 = np.asarray([1,-0.4964,0]))

fit_A, fit_seg, fit_C = popt
print('Opt. A =',fit_A,'Opt. seg =', fit_seg, 'Opt. C =', fit_C)
 
# Sample exp_decay with optimized parameters
opt_A, opt_seg, opt_C = popt

Curve_fit = exp_fitting(np.array(np.arange(1e-10,100)), opt_A, opt_seg,opt_C)
ax.plot(np.arange(1e-10,100), Curve_fit, linestyle = '--', color = 'k')