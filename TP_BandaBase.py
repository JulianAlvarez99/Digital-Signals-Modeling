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
from scipy import signal


class Signals:
    
    def __init__(self, amplitude = 5, sample_rate = 8):
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        
        self.data = (np.random.rand(10000)>0.5).astype(int)
        
        self.clk = np.arange(0,2*len(self.data)) % 2
        self.clk_sequence = np.repeat(self.clk, self.sample_rate)
        
        self.data_sequence = np.repeat(self.data,2*self.sample_rate)
        
        na = 32 # averaging factor to plot averaged welch spectrum
        self.win = signal.windows.hann(max(self.data.shape)//na) #// is for integer floor division
        # Welch PSD estimate with Hanning window and no overlap
        
    def UnipolarNRZ(self):
        return self.amplitude * self.data_sequence
    
    def UnipolarRZ(self):
        return self.amplitude * (self.data_sequence * (1 - self.clk_sequence))
 
    def sinc_pulse(self, duration=4):
        """
        Generate a sinc pulse.

        Parameters:
        duration (int): Duration of the sinc pulse.

        Returns:
        np.array: The sinc pulse.
        """
        t = np.linspace(-duration / 2, duration / 2, duration * self.sample_rate, endpoint=False)
        sinc_pulse = np.sinc(t)
        return sinc_pulse
    
    def UnipolarNRZ_Nyquist(self):
        """
        Generate Unipolar NRZ signal using Nyquist sinc pulse.
        """
        pulse = self.sinc_pulse()
        signal = np.convolve(self.data_sequence, pulse, mode='same')
        return self.amplitude * signal

    def PolarNRZ_Nyquist(self):
        """
        Generate Polar NRZ signal using Nyquist sinc pulse.
        """
        pulse = self.sinc_pulse()
        polar_data = 2 * self.data_sequence - 1
        signal = np.convolve(polar_data, pulse, mode='same')
        return self.amplitude * signal
  
    def M_ary_Nyquist(self):
        """
        Generate Polar NRZ signal using Nyquist sinc pulse.
        """
        pulse = self.sinc_pulse()
        Mary = self.M_ary()
        return np.convolve(Mary, pulse, mode='same')
        
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
        
        return signal

    def M_ary_plot(self):
                
        fig, ax = plt.subplots(2,1,sharex='col', figsize = (16, 14))
        ax[0].plot(self.data_sequence[0:1000]);ax[0].set_title('Data')
        
        signal = self.M_ary()
    
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
        
    def PSD_Unipolar(self):
        
        f0, psd_UNRZ = signal.welch(self.UnipolarNRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        f1, psd_URZ = signal.welch(self.UnipolarRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_UNRZ, label='Unipolar-NRZ')
        ax.semilogy(f1, psd_URZ, label='Unipolar-RZ')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD Unipolar")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
    def PSD_Polar(self):
        
        f0, psd_PNRZ = signal.welch(self.PolarNRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        f1, psd_PRZ = signal.welch(self.PolarRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_PNRZ, label='Polar-NRZ')
        ax.semilogy(f1, psd_PRZ, label='Polar-RZ')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD Polar")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
    def PSD_Manchester(self):
        
        f0, psd_Man = signal.welch(self.Manchester(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_Man, label='Manchester')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD Manchester")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")   
        
    def PSD_AMI(self):
        
        f0, psd_AMI = signal.welch(self.AMI(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_AMI, label='Alternate mark inversion')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD AMI")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")  

    def PSD_M_ary(self):
        
        M_arySignal = self.M_ary()
        
        f0, psd_Mary = signal.welch(M_arySignal,fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_Mary, label='Multilevel')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD M-ary")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
    def PSD_ALL(self):
        f0, psd_UNRZ = signal.welch(self.UnipolarNRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=True)
        f1, psd_PNRZ = signal.welch(self.PolarNRZ(),fs = 1/self.sample_rate, window= self.win, return_onesided=True)
        f2, psd_Man = signal.welch(self.Manchester(),fs = 1/self.sample_rate, window= self.win, return_onesided=True)
        f3, psd_AMI = signal.welch(self.AMI(),fs = 1/self.sample_rate, window= self.win, return_onesided=True)
        
        M_arySignal, symbol_duration = self.M_ary()
        f4, psd_Mary = signal.welch(M_arySignal,fs = 1/self.sample_rate, window= self.win, return_onesided=True)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_UNRZ, label='Unipolar-NRZ')
        ax.semilogy(f1, psd_PNRZ, label='Polar-NRZ')
        ax.semilogy(f2, psd_Man, label='Manchester')
        ax.semilogy(f3, psd_AMI, label='AMI')
        ax.semilogy(f4, psd_Mary, label='Multilevel')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
    def Plot_nyquist_signals(self):
        """
        Plot the Unipolar NRZ and Polar NRZ signals with Nyquist sinc pulse.
        """
        fig, ax = plt.subplots(3,1,sharex='col', figsize = (16, 15))
       
        ax[0].plot(self.PolarNRZ_Nyquist()[0:1000],color = 'green')
        ax[0].set_title('Polar NRZ con Pulso de Nyquist')
        ax[0].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5) 
        ax[0].yaxis.set_major_locator(MultipleLocator(15))
        ax[0].set_ylabel('Amplitude')
        
        ax[1].plot(self.UnipolarNRZ_Nyquist()[0:1000],color = 'red')
        ax[1].set_title('Unipolar NRZ con Pulso de Nyquist') 
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(15))
        ax[1].set_ylabel('Amplitude')
        plt.xlabel('Time')
        
        
        signal = self.M_ary_Nyquist()
        t = np.arange(0, len(signal) / self.sample_rate, 1 / self.sample_rate)
        
        ax[2].plot(t,signal,drawstyle='steps-pre',color = 'orange')
        ax[2].set_title('Señalizacion multinivel')
        ax[2].set_ylabel('Amplitude')
        ax[2].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[2].yaxis.set_major_locator(MultipleLocator(15))
        ax[2].set_xlim(0, 1000)  # Ajuste del eje x para mostrar 0 a 1000 muestras
        plt.xlabel('Time')
       
    
    def Plot_nyquist_psds(self):
        """
        Plot the PSDs of the Unipolar NRZ and Polar NRZ signals with Nyquist sinc pulse.
        """
        
        f0, psd_UNRZ = signal.welch(self.UnipolarNRZ_Nyquist(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        f1, psd_PNRZ = signal.welch(self.PolarNRZ_Nyquist(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        f2, psd_Mary = signal.welch(self.M_ary_Nyquist(),fs = 1/self.sample_rate, window= self.win, return_onesided=False) 
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_UNRZ, label='PSD Unipolar NRZ con Pulso de Nyquist')
        ax.semilogy(f1, psd_PNRZ, label='PSD Polar NRZ con Pulso de Nyquist')
        ax.semilogy(f2, psd_Mary, label='PSD M-aria con Pulso de Nyquist')
        ax.legend()
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        

New_signal = Signals(5,16)
# New_signal.Plot_ALL()
# New_signal.PSD_Unipolar()
# New_signal.PSD_Polar()
# New_signal.PSD_Manchester()
# New_signal.PSD_AMI()
# New_signal.PSD_M_ary()
# New_signal.PSD_ALL()
New_signal.Plot_nyquist_signals()
New_signal.Plot_nyquist_psds()



      
        
    
    
        
        
        
        
        
        