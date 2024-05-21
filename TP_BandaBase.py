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


class Signals:
    
    def __init__(self, amplitude, sample_rate):
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
        
    # def M_ary(self, ):
    
    def int_to_binary(self, value):
        return (bin(value)[2:])

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
        
New_signal = Signals(5,8)
New_signal.Unipolar_plot()
New_signal.Polar_plot()
New_signal.Manchester_plot()
New_signal.AMI_plot()



      
        
    
    
        
        
        
        
        
        