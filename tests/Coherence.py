# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:09:11 2024

@author: super
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from ShockOscillationAnalysis import SOA
plt.rcParams.update({'font.size': 25})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"

f = 1500; window_size = 512; overlap = 128
#%%
paths = []; DfList = []; VList =[]
Dict = {};
linstyl=['-','--',':','-.','dashdotdotted']
fig,ax = plt.subplots(figsize=(10,10))
fig2,ax2 = plt.subplots(figsize=(10,10))
fig3,ax3 = plt.subplots(figsize=(10,10))
fig4,ax4 = plt.subplots(figsize=(10,10))
fig5,ax5 = plt.subplots(figsize=(10,10))
SA = SOA(f)
#%%
HLP = 7
LE_Dir = 'tests\\LowRe\\Leadingedge'
PS_Dir = 'tests\\LowRe\\Passage\\'

CaseName = f'{HLP}mm_LE-R23deg'
Dict[CaseName] = 0
File = f"ShockLocation_{CaseName}.txt"
paths.append(f'{LE_Dir}\\Rotated\\{HLP}mm\\{File}')

# CaseName = f'{HLP}mm_LE-XR'
# Dict[CaseName] = 1
# File = f"ShockLocation_{CaseName}.txt"
# paths.append(f'{LE_Dir}\\NotRotated\\{HLP}mm\\{File}')

# CaseName = f'{HLP}mm_PS-XR'
# Dict[CaseName] = 2
# File = f"ShockLocation_{CaseName}.txt"
# paths.append(f'{PS_Dir}NotRotated\\{HLP}mm\\{File}')

CaseName = f'{HLP}mm_PS-R5deg'
Dict[CaseName] = 1
File = f"ShockLocation_{CaseName}.txt"
paths.append(f'{PS_Dir}Rotated\\{HLP}mm\\{File}')

#%%
HLP = 15
CaseName = f'{HLP}mm_LE-R29deg'
Dict[CaseName] = 2
File = f"ShockLocation_{CaseName}.txt"
paths.append(f'{LE_Dir}\\Rotated\\{HLP}mm\\{File}')

# CaseName = f'{HLP}mm_LE-XR'
# Dict[CaseName] = 5
# File = f"ShockLocation_{CaseName}.txt"
# paths.append(f'{LE_Dir}\\NotRotated\\{HLP}mm\\{File}')

# CaseName = f'{HLP}mm_PS-XR'
# Dict[CaseName] = 6
# File = f"ShockLocation_{CaseName}.txt"
# paths.append(f'{PS_Dir}NotRotated\\{HLP}mm\\{File}')

CaseName = f'{HLP}mm_PS-R2deg'
Dict[CaseName] = 3
File = f"ShockLocation_{CaseName}.txt"
paths.append(f'{PS_Dir}Rotated\\{HLP}mm\\{File}')
#%%
print(Dict)
ListDictValues = list(Dict.values())
print(ListDictValues); o = 0
for i in ListDictValues:
    if os.path.exists(paths[o]):     
        CaseName = list(Dict.keys())[list(Dict.values()).index(i)]
        print(u'******* file exist \u2713', 'case name:', CaseName,'*******')
    else:
        print('File is not exist', paths[o])
        continue    
    Df = pd.read_csv(paths[o], header=None)
    n = Df.shape[0];
    Freq, psd = signal.welch(x = Df[1], fs = f, window='barthann',
                          nperseg = window_size, noverlap=overlap, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
    
    ax.loglog(Freq[1:], psd[1:], lw = '2', label = CaseName)
    T = n/f    
    t = np.linspace(0,T,n);
    
    V = SA.VelocitySignal(Df[1], T)
    VList.append(V)
    
    Freq, psd2 = signal.welch(x = V, fs = f, window='barthann',
                          nperseg = window_size, noverlap=overlap, nfft=None, detrend='constant',
                          return_onesided=True, scaling='density')
    
    # Find maximum peak in velocity PSD
    domFreq = Freq[psd2.argmax(axis=0)]
    print(f'max peak at: {domFreq: 0.2f}Hz')    
    ax2.semilogx(Freq[1:], psd2[1:] , lw = '2', label = CaseName)
    
    k = 0; domain = []
    while k < n:
        shockLoc =[]; j = 0
        while j <= 500 and k < n:
            shockLoc.append(Df[1].iloc[k])
            j += 1; k += 1
        domain.append(max(shockLoc)-min(shockLoc))
    avgDomain = np.mean(domain)
    print(f"Shock oscillation domain \t \t {max(Df[1])-min(Df[1]):0.2f}mm")
    print(f"Average Shock oscillation domain {avgDomain:0.2f}mm")
    
    certain = Df[2].sum()
    print(certain)
    uncertainity = (1-certain/n)*100
    print('uncertainty ratio:', round(uncertainity,2),'%')
    DfList.append(Df)
    o += 1
    
    
Freq, C7 = signal.coherence(DfList[Dict['7mm_LE-R23deg']][1], DfList[Dict['7mm_PS-R5deg']][1], 
                            fs=f, window='barthann', nperseg=window_size, noverlap=overlap)

Freq, C15 = signal.coherence(DfList[Dict['15mm_LE-R29deg']][1],DfList[Dict['15mm_PS-R2deg']][1], 
                             fs=f, window='barthann', nperseg=window_size, noverlap=overlap)
ax3.semilogx(Freq[1:], C7[1:] , lw = '2', label = '7mm') 
ax3.semilogx(Freq[1:], C15[1:] , lw = '2', label ='15mm', ls = linstyl[1])

Freq, C7 = signal.coherence(VList[Dict['7mm_LE-R23deg']], VList[Dict['7mm_PS-R5deg']], 
                            fs=f, window='barthann', nperseg=window_size, noverlap=overlap)

Freq, C15 = signal.coherence(VList[Dict['15mm_LE-R29deg']], VList[Dict['15mm_PS-R2deg']], 
                             fs=f, window='barthann', nperseg=window_size, noverlap=overlap)

Freq, CLE = signal.coherence(DfList[Dict['15mm_LE-R29deg']][1], DfList[Dict['7mm_LE-R23deg']][1], 
                             fs=f, window='barthann', nperseg=window_size, noverlap=overlap)


Freq, CPS = signal.coherence(DfList[Dict['7mm_PS-R5deg']][1], DfList[Dict['15mm_PS-R2deg']][1], 
                             fs=f, window='barthann', nperseg=window_size, noverlap=overlap)



# Freq4, C2 = signal.coherence(VList[1], VList[0], fs=f, window='barthann', 
#                             nperseg=512, noverlap=64, nfft=None, detrend='constant')

# ax3.semilogx(Freq3, C , lw = '2') 
# ax3.loglog(Freq, C7 , lw = '2') 
# ax3.semilogx(Freq4, C2 , lw = '2') 

ax4.semilogx(Freq[1:], C7[1:] , lw = '2', label = '7mm') 
ax4.semilogx(Freq[1:], C15[1:] , lw = '2', label ='15mm', ls = linstyl[1]) 


ax5.semilogx(Freq[1:], CPS[1:] , lw = '2', label ='passage shock') 
ax5.semilogx(Freq[1:], CLE[1:] , lw = '2', label ='leading-edge shock', ls = linstyl[1]) 
# ax3.semilogx(Freq4, C2 , lw = '2') 


        

ax.set_ylabel(r"PSD [mm$^2$/Hz]"); 
ax.set_xlabel("Frequency [Hz]");
ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax.minorticks_on()
ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
ax.legend(fontsize="22", ncol = 2)

ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
ax2.set_xlabel("Frequency [Hz]");
ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax2.minorticks_on()
ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
ax2.legend()

ax3.set_ylabel(r"$Coh(f)$"); 
ax3.set_xlabel("Frequency [Hz]");
ax3.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax3.minorticks_on()
ax3.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
ax3.legend(fontsize="22")
ax3.set_ylim([0,1])
# ax3.set_xticks([100,500,1000])
ax3.set_xlim([1,1000])


ax4.set_ylabel(r"$Coh(f)$"); 
ax4.set_xlabel("Frequency [Hz]");
ax4.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax4.minorticks_on()
ax4.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
ax4.legend()
ax4.set_ylim([0,1])
ax4.set_xlim([1,1000])

ax5.set_ylabel(r"$Coh(f)$"); 
ax5.set_xlabel("Frequency [Hz]");
ax5.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
ax5.minorticks_on()
ax5.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
ax5.set_xlim([1,1000])
ax5.set_ylim([0,1])
ax5.legend(fontsize="22", ncol = 2)