# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:17:41 2024

@author: asusx
"""
# Se propone una simulación de formas de onda en el tiempo, y densidades
# espectrales correspondientes, de al menos dos formatos de transmisión, uno de
# cada uno de los dos siguientes grupos de formatos:
# Grupo a: Formatos de pulso rectangular Unipolar NRZ, Unipolar RZ, Polar NRZ,
# Polar RZ
# Grupo b: Formatos de pulso rectangular Mánchester, AMI y formato M-ario.

# También simular que es lo que sucede en los casos UNRZ y PNRZ si en lugar del
# formato rectangular se utiliza un pulso de Nyquist, incluyendo el extremo del
# pulso sinc(rt).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
from scipy.signals import windows, welch

class Signals:
    
    def __init__(self, amplitude = 5, sample_rate = 8):
        self.amplitude = amplitude
        self.frequency = 8*sample_rate
        self.sample_rate = sample_rate
        
        self.data = (np.random.rand(10000)>0.5).astype(int)
        
        self.clk = np.arange(0,2*len(self.data)) % 2
        self.clk_sequence = np.repeat(self.clk, self.sample_rate)
        
        self.data_sequence = np.repeat(self.data,2*self.sample_rate)
    
    def Rand_package(n):
        Package = np.empty(n)
        for i in range(n):
            temp = str(random.randint(0, 1))
            Package[i]=temp
        return(Package)
        
    def UnipolarNRZ(self):
        return self.amplitude * self.data_sequence
    
    def UnipolarRZ(self):
        return self.amplitude * (self.data_sequence * (1 - self.clk_sequence))
        
    def PolarNRZ(self):
        return self.amplitude * (2 * self.data_sequence - 1)
        
    def PolarRZ(self):
        return self.amplitude * (2 * self.data_sequence * (1 - self.clk_sequence) - 1)
        
    def Manchester(self):
        return self.amplitude * (2 * np.logical_xor(self.data_sequence, self.clk_sequence).astype(int) - 1)
    
    def AMI(self):
        ami = 1*self.data 
        previusOne = 0
        
        for i in range(0,len(self.data)):
            if(ami[i] == 1) and (previusOne == 0):
                ami[i] = self.amplitude
                previusOne = 1
            if(ami[i] == 1) and (previusOne == 1):
                ami[i] = -self.amplitude
                previusOne = 0
        
        return np.repeat(ami, 2*self.sample_rate)
        
    def M_ary(self, bits_per_symbol=2):
        # bits_per_symbol determina la cantidad de bits que tiene cada simbolo
        # Número de niveles M-PAM
        M = 2 ** bits_per_symbol

        # Definir los niveles de la señal M-PAM
        levels = np.linspace(-self.amplitude, self.amplitude, M)
        
        # Convertir el mensaje binario en símbolos
        symbols = []
        for i in range(0, len(self.data), bits_per_symbol):
            bits = self.data[i:i + bits_per_symbol]
            symbol = 0
            for bit in bits:
                symbol = (symbol << 1) | bit
            symbols.append(symbol)
        
        # Mapear los símbolos a los niveles de amplitud
        symbols = levels[symbols]

        # Generar la señal en banda base
        symbol_duration = np.log2(M) * self.sample_rate
        samples_per_symbol = int(symbol_duration * 2*self.sample_rate)
        # Multiplicar por 2 el sample_rate sale de como se armo la secuencia de datos
        signal = np.repeat(symbols, samples_per_symbol)
        
        return signal, symbol_duration

    def M_ary_plot(self):
                
        fig, ax = plt.subplots(2,1,sharex='col', figsize = (16, 14))
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        
        signal,symbol_duration = self.M_ary()
    
        t = np.arange(0, len(signal) / self.sample_rate, 1 / self.sample_rate)
        
        ax[1].plot(t,signal,drawstyle='steps-pre',color = 'green')
        ax[1].set_title('Señalizacion multinivel')
        ax[1].set_ylabel('Amplitude')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].set_ylim(-self.amplitude - 0.5,self.amplitude + 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].set_xlim(0, 1000)  # Ajuste del eje x para mostrar 0 a 1000 muestras
        plt.xlabel('Time')

    def Polar_plot(self):
        fig, ax = plt.subplots(3,1,sharex='col', figsize = (16, 15))
        # ax[0].plot(self.clk_sequence[0:1000]);ax[0].set_title('Clock')
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        
        ax[1].plot(self.PolarNRZ()[0:1000],color = 'green');ax[1].set_title('Polar NRZ')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5) 
        ax[1].set_ylim(-self.amplitude - 0.5,self.amplitude + 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].set_ylabel('Amplitude')
        
        ax[2].plot(self.PolarRZ()[0:1000],color = 'red');ax[2].set_title('Polar RZ') 
        ax[2].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[2].set_ylim(-self.amplitude - 0.5,self.amplitude + 0.5)
        ax[2].yaxis.set_major_locator(MultipleLocator(1))
        ax[2].set_ylabel('Amplitude')
        plt.xlabel('Time')
    
    def Unipolar_plot(self):    
        fig, ax = plt.subplots(3,1,sharex='col', figsize = (16, 14))
        # ax[0].plot(self.clk_sequence[0:1000]);ax[0].set_title('Clock')
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        ax[1].plot(self.UnipolarRZ()[0:1000],color = 'green');ax[1].set_title('Unipolar RZ')
        ax[1].set_ylabel('Amplitude')
        ax[2].plot(self.UnipolarNRZ()[0:1000],color = 'red');ax[2].set_title('Unipolar NRZ') 
        ax[2].set_ylabel('Amplitude')
        plt.xlabel('Time')
        
    def Manchester_plot(self):
        fig, ax = plt.subplots(2,1,sharex='col', figsize = (16, 14))
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        ax[1].plot(self.Manchester()[0:1000],color = 'green');ax[1].set_title('Manchester')
        ax[1].set_ylabel('Amplitude')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].set_ylim(-self.amplitude - 0.5,self.amplitude + 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        plt.xlabel('Time')
        
    def AMI_plot(self):
        fig, ax = plt.subplots(2,1,sharex='col', figsize = (16, 14))
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        ax[1].plot(self.AMI()[0:1000],color = 'green');ax[1].set_title('Alternate Mark Inversion (AMI)')
        ax[1].set_ylabel('Amplitude')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].set_ylim(-self.amplitude - 0.5,self.amplitude + 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        plt.xlabel('Time')
        
    def Plot_ALL(self):
        self.Unipolar_plot()
        self.Polar_plot()
        self.Manchester_plot()
        self.AMI_plot()
        self.M_ary_plot()
        
    def PSD_UnipolarNRZ(self):
        f, psdU = welch(self.UnipolarNRZ(),fs = 1,return_oneside = False, scaling = 'density')
        plt.semilogy(f, psdU, label='Unipolar-NRZ-L')


New_signal = Signals(5,16)
# New_signal.Unipolar_plot()
# New_signal.Polar_plot()
# New_signal.Manchester_plot()
# New_signal.AMI_plot()
# New_signal.M_ary_plot()
# New_signal.Plot_ALL()
New_signal.PSD_UnipolarNRZ()





      
        
    
    
        
        
        
        
        
        