# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:51:47 2022

@author: Hady
"""

from ShockOscillationAnalysis import SOA
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import cv2
plt.rcParams.update({'font.size': 25})

# Dir = 'D:\\TFAST\\TEAMAero experiments\\2023_06_12\\'
Dir = 'tests\\'
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_90 from the ref with step.png']#,
#          'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_90deg from the ref.png']
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_180 from the ref with step.png',
#          'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_180deg from the ref.png']
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_22mm_ts_5_360 from the ref with step.png']
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_7mm_0.13232104121475055mm-px_ts_20_5_360 from the ref with step.png']#,
#          'Shock wave pixels\\Philipp exp\\6.0kHz_22mm_ts_50_360deg from the ref.png']
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_15mm_ts_50_neg 180 from the ref with step.png',
#           'Shock wave pixels\\Philipp exp\\6.0kHz_15mm_ts_50_neg 180 from the ref.png']



# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_90deg from the ref.png',
          # 'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_180deg from the ref.png',
          # 'Shock wave pixels\\Philipp exp\\6.0kHz_22mm_ts_50_360deg from the ref.png',
          # 'Shock wave pixels\\Philipp exp\\6.0kHz_15mm_ts_50_neg 180 from the ref.png']
          
# Paths = ['Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_90deg from the ref.png']#,
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_180deg from the ref.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_22mm_ts_50_360deg from the ref.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_15mm_ts_50_neg 180 from the ref.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_90 from the ref with step.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_ts_50_180 from the ref with step.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_7mm_0.13232104121475055mm-px_ts_20_5_360 from the ref with step.png',
         # 'Shock wave pixels\\Philipp exp\\6.0kHz_15mm_ts_50_neg 180 from the ref with step.png']

# Paths = ['P1_2.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#           'P2_2.0kHz_7mm_0.12461059190031153mm-px_ts_35_slice.png',
#           'P3_2.0kHz_7mm_0.13008130081300814mm-px_ts_35_slice.png',
#           'P4_2.0kHz_15mm_0.1288244766505636mm-px_ts_50_slice.png']#,
          #'P5_2.0kHz_7mm_0.13582342954159593mm-px_ts_40_slice.png']

# Paths = ['P1_6.0kHz_7mm_0.13008130081300814mm-px_ts_35_slice.png',
#           'P2_6.0kHz_7mm_0.12461059190031153mm-px_ts_35_slice.png',
#           'P3_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#           'P4_6.0kHz_7mm_0.1282051282051282mm-px_ts_35_slice.png']

# Paths = ['P1_15.0kHz_7mm_0.12924071082390953mm-px_ts_40_slice.png',
#          'P2_15.0kHz_7mm_0.12326656394453005mm-px_ts_50_slice.png',
#          'P3_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#          'P4_15.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png']

# Paths = ['P1_10.0kHz_7mm_0.12987012987012986mm-px_ts_35_slice.png',
#           'P2_10.0kHz_7mm_0.12364760432766615mm-px_ts_35_slice.png',
#           'P3_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#           'P4_10.0kHz_7mm_0.12924071082390953mm-px_ts_35_slice.png']

# Paths = ['P1_15.0kHz_7mm_0.12924071082390953mm-px_ts_40_slice.png',
#          'P3_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#          'P4_15.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png']

# Paths = ['P1_10.0kHz_7mm_0.12987012987012986mm-px_ts_35_slice.png',
#           'P3_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#           'P4_10.0kHz_7mm_0.12924071082390953mm-px_ts_35_slice.png']

# Paths = ['HiRe_2.0kHz_7mm_0.13008130081300814mm-px_ts_35_slice.png',
         # 'MidRe_1.5kHz_7mm_0.13616071428571427mm-px_ts_35_slice.png',
         # 'MidRe_1.5kHz_7mm_0.13215859030837004mm-px_ts_80_slice-8.png']

# Paths = ['Ref-Notrip_5.0kHz_7mm_0.12944983818770225mm-px_ts_35_slice.png',
#          '10mm-trip_6.0kHz_7mm_0.13707865168539327mm-px_ts_35_slice.png',
#          '25mm-trip_6.0kHz_7mm_0.1383219954648526mm-px_ts_35_slice.png']

Paths = ['2.0kHz_7mm_0.12965964343598055mm-px_ts_60_slice-Rotated-.png',
          '3.0kHz_7mm_0.12965964343598055mm-px_ts_60_slice-Rotated-.png',
          '6.0kHz_7mm_0.13029315960912052mm-px_ts_60_slice-Rotated-.png',
          '10.0kHz_7mm_0.13029315960912052mm-px_ts_60_slice-Rotated-.png',
          '15.0kHz_7mm_0.12944983818770225mm-px_ts_60_slice-Rotated-.png']



count = 0
fig,ax = plt.subplots(figsize=(10,10))
fig2,ax2 = plt.subplots(figsize=(10,10))
fig3,ax3 = plt.subplots(figsize=(10,10))

linstyl=['-','--',':','-.',(0, (3, 5, 1, 5, 1, 5))]
for i in Paths:
    # FileDirectory = 'C:\\Users\\admin\\Nextcloud\\Documents\\TeamAero Experiments\\Shock wave pixels\\ReynoldsNumComp\\'
    Folders = i.split(".png")
    
    # for folder in range(len(Folders)-1): FileDirectory += (Folders[folder]+'\\')
    # NewFileDirectory = os.path.join(FileDirectory, "shock_signal_rowdata")
    # if not os.path.exists(NewFileDirectory): os.mkdir(NewFileDirectory)
    # f = float(Folders[0].split("_")[1].split("kHz")[0])*1000
    f = float(Folders[0].split("_")[0].split("kHz")[0])*1000
    # CaseName = Folders[0].split("_")[0]
    CaseName = Folders[0].split("_")[0]
    
    ts = Folders[0].split("_")[-2]  
    
    # print('*** Loading shock location for case: ',CaseName,' ***')
    print('*** Loading shock location for case: ',CaseName,' ***')

    
    ImgList = cv2.imread(f'{Dir}{int(f/1000)}kHz\\{i}')
    LELoc = Folders[0].split("_")[2]
    
    Scale = float(Folders[0].split("_")[-4].split('mm-px')[0])
    SA = SOA(f, pixelScale = Scale)
    nShoots = ImgList.shape[0]
    # ShockName = ['passage shock','expansion wave','leading edge shock']
    # ShockName = ['passage shock', 'reflected shock']
    ShockName = ['passage shock']
    for j in ShockName:
        if   CaseName == '2.0kHz' :  NewRef = [282, 435]; CaseName = '2kHz';
        elif CaseName == '3.0kHz' :  NewRef = [288, 431]; CaseName = '3kHz';
        elif CaseName == '6.0kHz' :  NewRef = [286, 433]; CaseName = '6kHz';
        elif CaseName == '10.0kHz':  NewRef = [283, 440]; CaseName = '10kHz';
        elif CaseName == '15.0kHz':  NewRef = [291, 426]; CaseName = '15kHz';
        
        
        # 10kHz
        # if   CaseName == 'P1':  NewRef = [283, 443]; CaseName = 'P1-0.17Ra';
        # elif CaseName == 'P2':  NewRef = [306, 480]; CaseName = 'P2-0.20Ra';
        # elif CaseName == 'P3':  NewRef = [272, 442]; CaseName = 'P3-0.55Ra';
        # elif CaseName == 'P4':  NewRef = [265, 434]; CaseName = 'P4 1.48Ra';
        # elif CaseName == 'P5':  NewRef = [265, 418]; CaseName = 'P5-0.64Ra';
        
        # 2kHz
        # if   CaseName == 'P1':  NewRef = [383, 552]; CaseName = 'P1-0.17Ra';
        # elif CaseName == 'P2':  NewRef = [315, 484]; CaseName = 'P2-0.20Ra';
        # elif CaseName == 'P3':  NewRef = [290, 459]; CaseName = 'P3-0.55Ra';
        # elif CaseName == 'P4':  NewRef = [260, 429]; CaseName = 'P4-1.48Ra';
        # elif CaseName == 'P5':  NewRef = [265, 418]; CaseName = 'P5-0.64Ra';
        
        # if   CaseName == 'HiRe':  NewRef = [275, 427]; filterCenter = [(0, 233)]; d = 20; 
        # elif CaseName == 'MidRe':  NewRef = [98, 253]; filterCenter = [(0, 465), (0, 480), (0, 495)]; d = 20; 
        
        # if   CaseName == 'Ref-Notrip':  NewRef = [275, 427]; #CaseName = 'Ra 0.17µm';
        # elif CaseName == '10mm-trip':  NewRef = [132, 272]; #CaseName = 'P2-0.20Ra';
        # elif CaseName == '25mm-trip':  NewRef = [128, 267]; #CaseName = 'P2-0.20Ra';

        
        # if CaseName == '360 from the ref with step':    NewRef = [545,645]
        # elif CaseName == '360deg from the ref':         NewRef = [313,460]

        # else:
        # NewRef = [185,325]

        # NewRef = SA.LineDraw(ImgList, 'V', 0, 1)
        # NewRef = SA.LineDraw(SA.clone, 'V', 1)
        # NewRef.sort()
        # if j == 'passage shock':     NewRef = [545,645]
        # elif j == 'reflected shock': NewRef = [372,451]
        # elif j == 'expansion wave' and CaseName == '20mm': NewRef = [194,232]
        # elif j == 'leading edge shock'and CaseName == '15mm': NewRef = [0,63]
        # elif j == 'leading edge shock'and CaseName == '20mm': NewRef = [0,86]
        ShockwaveRegion = ImgList[:,NewRef[0]:NewRef[1]]
        
        # cv2.imshow("Shock wave Region", ShockwaveRegion)
        # cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);
        xPixls = (NewRef[1]-NewRef[0])
        ShockResionScale = xPixls*Scale
        print('Image scale: ', Scale, 'mm/px') # indicates the displacement accuracy
        print('Shock Regions:',NewRef,'\t Represents:' ,xPixls, 'px \t Shock Regions in mm:', ShockResionScale)
    
        print('Cleaning illumination instability ...')
        ShockwaveRegion = SA.CleanSnapshots(ShockwaveRegion,
                                            'Average', 'FFT', #'Brightness and Contrast',
                                            filterCenter = [(0,233)], D = 20, n = 5,
                                            # Brightness = 2, Contrast = 1.7, Sharpness = 2,
                                            ShowIm = False)
        
        ShockLocation, Uncer = SA.ShockTrakingAutomation(ShockwaveRegion, 
                                                           reviewInterval = [0,0],
                                                           # Signalfilter = 'None')
                                                            Signalfilter = 'med-Wiener')
        
        
        uncertainityRatio = (len(Uncer)/len(ShockLocation))*100
        print('uncertainty ratio:', round(uncertainityRatio,2),'%')
        
                
        A = Scale * np.array(ShockLocation)
        avg = np.average(A)
        ShockLocation = A - avg
        
                
        k = 0; domain = []
        while k < nShoots:
            shockLoc =[]; n = 0
            while n <= 500 and k < nShoots:
                shockLoc.append(ShockLocation[k])
                n += 1; k += 1
            domain.append(max(shockLoc)-min(shockLoc))
        avgDomain = np.mean(domain)    
            
        fig1, ax1 = plt.subplots(figsize=(20,200))
        ax1.imshow(ShockwaveRegion, extent=[0, ShockResionScale, nShoots, 0], aspect='0.1', cmap='gray');
        ax1.plot(A, range(nShoots),'x', lw = 1, color = 'g', ms = 2)
        # ax1.xaxis.set_major_locator(MultipleLocator(int(ShockResionScale/4)))
        ax1.xaxis.set_major_formatter('{x:.0f}')
        ax1.set_ylim([0,nShoots])
     
        print("Shock oscillation domain",max(ShockLocation)-min(ShockLocation))
        print("Average Shock oscillation domain",avgDomain)
        
        # Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
        #                       nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
        #                       return_onesided=True, scaling='density')
        Freq, psd = signal.welch(x = ShockLocation, fs = f, window='barthann',
                              nperseg = 512, noverlap=0, nfft=None, detrend='constant',
                              return_onesided=True, scaling='density')
        
        # Intpsd = np.trapz(psd,Freq)
        
        # ax.semilogx(Freq, psd, lw = '2', label = CaseName, linestyle = linstyl[count])
        ax.loglog(Freq, psd, lw = '2', label = CaseName, linestyle = linstyl[count])
        # ax3.loglog(Freq, psd/Intpsd, lw = '2', label = CaseName, linestyle = linstyl[count])
        # ax.axvline(x = 19.5,ls='--',color='k',alpha=0.4)
        # ax.axvline(x = 287,ls='--',color='k',alpha=0.4)
        
        T = nShoots/f
        print("Total measuring time: ", T, "sec")
        
        dx_dt = []; dt = T/nShoots; t = np.linspace(0,T,nShoots);
        # dx_dtNP = np.gradient(ShockLocation, dt*1000)
        
        for xi in range(nShoots):
            if xi > 0 and xi < nShoots-1:
                dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi-1])/(2*dt*1000))
            elif xi == 0: 
                dx_dt.append((ShockLocation[xi+1]-ShockLocation[xi])/(dt*1000))
            elif xi == nShoots-1:
                dx_dt.append((ShockLocation[xi]-ShockLocation[xi-1])/(dt*1000))


        V_avg = np.mean(dx_dt) 
        V = dx_dt - V_avg
        
        # V_np_avg = np.mean(dx_dtNP)
        # V_np = dx_dtNP - V_avg
        
        # Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
        #                       nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
        #                       return_onesided=True, scaling='density')
        Freq2, psd2 = signal.welch(x = V, fs = f, window='barthann',
                              nperseg = 512, noverlap=0, nfft=None, detrend='constant',
                              return_onesided=True, scaling='density')
        
        # FreqNP, psdNP = signal.welch(x = V_np, fs = f, window='barthann',
        #                       nperseg = 512*f/2000, noverlap=0, nfft=None, detrend='constant',
        #                       return_onesided=True, scaling='density')
        
        domFreq = Freq2[psd2.argmax(axis=0)]
        
        print('max peak at:', domFreq, 'Hz')
        
        uncertain = []; Loc = []
        for i in Uncer:
            uncertain.append(i[1]*Scale)
            Loc.append(i[0])
        
        k = 0; domain = []
        while k < nShoots:
            shockLoc =[]; j = 0
            while j <= 500 and k < nShoots:
                shockLoc.append(ShockLocation[k])
                j += 1; k += 1
            domain.append(max(shockLoc)-min(shockLoc))
        avgDomain = np.mean(domain)
 
        ShockLocationfile = []
        k = 0
        for i in range(nShoots):
            uncer = 1
            if len(Loc) < k and i == Loc[k]:  uncer == 0; k +=1
            ShockLocationfile.append([i, ShockLocation[i], V[i], uncer])
            
        np.savetxt(f'{Dir}\\ShockLocation-{CaseName}.txt', 
                    ShockLocationfile,  delimiter = ",")
        
        
        # =================================
        
        
        # ax2.plot(t, dx_dt, lw = '2')
        # ax2.semilogx(Freq2, psd2 , lw = '2', label = CaseName, linestyle = linstyl[count])
        ax2.loglog(Freq2, psd2 , lw = '2', label = CaseName, linestyle = linstyl[count])
        # ax2[1].loglog(FreqNP, psdNP , lw = '2', label = CaseName)
        # ax2.axvline(x =Freq2[psd2.argmax(axis=0)],ls='--',color='k',alpha=0.4)
        ax2.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
        ax2.set_xlabel("Frequency [Hz]");
        # ax2.set_xlim([10,10e2]);
        # ax2.set_title('PSD for '+ CaseName +' above LE')
        # ax2.set_title('Shock velocity PSD')
        ax2.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
        ax2.minorticks_on()
        ax2.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
        ax2.legend();
        
        # for i in ax2:
        #     i.set_ylabel(r"PSD $[m^2.s^{-2}.Hz^{-1}]$"); 
        #     i.set_xlabel("Frequency [Hz]");
        #     # ax2.set_title('PSD for '+ CaseName +' above LE')
        #     i.set_title('Shock velocity PSD')
        #     i.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
        #     i.minorticks_on()
        #     i.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
        #     i.legend();

    ax.set_ylabel(r"PSD [mm$^2$/Hz]"); 
    ax.set_xlabel("Frequency [Hz]");
    # ax.set_xlim([1,3e3])
    # ax.set_ylim([1e-7,5e-2])
    # ax.set_title('Shock displacement PSD')

    # ax.set_title('PSD for '+CaseName+'\n slice loc. is '+LELoc+' above LE, with thickness of ' + str(int(ts))+'px')
    ax.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
    ax.legend();
    
    ax3.set_ylabel(r"$PSD/ \int PSD$"); 
    ax3.set_xlabel("Frequency [Hz]");
    # ax.set_xlim([1,3e3])
    # ax.set_ylim([1e-7,5e-2])
    # ax3.set_title('Shock displacement PSD')

    # ax.set_title('PSD for '+CaseName+'\n slice loc. is '+LELoc+' above LE, with thickness of ' + str(int(ts))+'px')
    ax3.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
    ax3.minorticks_on()
    ax3.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
    ax3.legend();
    count += 1
    