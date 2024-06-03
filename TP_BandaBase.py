# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:17:41 2024

@author: Alvarez Julian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal


class Signals:
    
    def __init__(self, amplitude = 5, sample_rate = 8):
        """
        Inicializa la clase Signals con parámetros dados.

        Parameters:
        amplitude (int): Amplitud de la señal.
        sample_rate (int): Tasa de muestreo de la señal.
        """
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        
        self.data = (np.random.rand(10000)>0.5).astype(int)
        
        self.clk = np.arange(0,2*len(self.data)) % 2
        self.clk_sequence = np.repeat(self.clk, self.sample_rate)
        
        self.data_sequence = np.repeat(self.data,2*self.sample_rate)
        
        na = 32 # averaging factor to plot averaged welch spectrum
        self.win = signal.windows.hann(max(self.data.shape)//na) #// is for integer floor division
        # Welch PSD estimate with Hanning window and no overlap

        
    def ASK(self):
        """
        Genera una señal modulado ASK (Amplitude Shift Keying).

        Returns:
        modulated_signal: Señal modulada ASK.
        """        
        # Carrier signal: sine wave with a given sample rate
      
        t = np.arange(len(self.data_sequence)) / self.sample_rate

        carrier = np.sin(2 * np.pi * t)  # Frequency is set to 1 Hz for simplicity

        # Modulated signal
        modulated_signal = self.data_sequence * carrier * self.amplitude
        
        return modulated_signal
           
    
    def MPSK(self,bits_per_symbol = 8, f = 10, fs = 500):
        """
        Genera una señal modulada MPSK (Phase Shift Keying).

        Parameters:
        bits_per_symbol (int): Número de bits por símbolo.
        f (int): Frecuencia de la señal portadora.
        fs (int): Tasa de muestreo de la señal.

        Returns:
        modulated_signal: Señal modulada MPSK.
        """
        # Número de bits por símbolo
        M = 2**bits_per_symbol
        
        # Número de símbolos
        num_symbols = len(self.data) // bits_per_symbol
        
        # Vector de tiempo para un símbolo
        t_symbol = np.linspace(0, 1/self.sample_rate, int(fs/self.sample_rate), endpoint=False)
        # Vector de tiempo total
        t = np.linspace(0, num_symbols/self.sample_rate, num_symbols*int(fs/self.sample_rate), endpoint=False)
        
        # Generar los ángulos de fase para M-PSK
        phase_angles = {i: 2*np.pi*i/M for i in range(M)}
        
        # Generar la señal M-PSK
        modulated_signal = np.zeros_like(t)
        for i in range(num_symbols):
            bits = self.data[i*bits_per_symbol:(i+1)*bits_per_symbol]
            symbol = int(''.join(map(str, bits)), 2)
            phase = phase_angles[symbol]
            modulated_signal[i*int(fs/self.sample_rate):(i+1)*int(fs/self.sample_rate)] = np.sin(2 * np.pi * f * t_symbol + phase)
        
        modulated_signal *= self.amplitude
        
        return modulated_signal
    

        
    def UnipolarNRZ(self):
        """
        Genera una señal Unipolar NRZ (Non-Return-to-Zero).

        Returns:
        np.array: Señal Unipolar NRZ.
        """
        return self.amplitude * self.data_sequence
    
    def UnipolarRZ(self):
        """
        Genera una señal Unipolar RZ (Return-to-Zero).

        Returns:
        np.array: Señal Unipolar RZ.
        """
        return self.amplitude * (self.data_sequence * (1 - self.clk_sequence))
 
    def sinc_pulse(self, duration=100):
        """
        Genera un pulso sinc.

        Parameters:
        duration (int): Duración del pulso sinc.

        Returns:
        np.array: El pulso sinc.
        """
        t = np.linspace(-duration / 2, duration / 2, duration * self.sample_rate, endpoint=False)
        sinc_pulse = np.sinc(t)
        return sinc_pulse
    
    def UnipolarNRZ_Nyquist(self):
        """
        Genera una señal Unipolar NRZ utilizando un pulso sinc de Nyquist.

        Returns:
        np.array: Señal Unipolar NRZ con pulso de Nyquist.
        """
        pulse = self.sinc_pulse()
        signal = np.convolve(self.data_sequence, pulse, mode='same')
        return self.amplitude * signal

    def PolarNRZ_Nyquist(self):
        """
       Genera una señal Polar NRZ utilizando un pulso sinc de Nyquist.

       Returns:
       np.array: Señal Polar NRZ con pulso de Nyquist.
       """
        pulse = self.sinc_pulse()
        polar_data = 2 * self.data_sequence - 1
        signal = np.convolve(polar_data, pulse, mode='same')
        return self.amplitude * signal
  
    def M_ary_Nyquist(self):
        """
        Genera una señal M-ary utilizando un pulso sinc de Nyquist.

        Returns:
        np.array: Señal M-ary con pulso de Nyquist.
        """
        pulse = self.sinc_pulse()
        Mary = self.M_ary()
        return np.convolve(Mary, pulse, mode='same')
        
    def PolarNRZ(self):
        """
       Genera una señal Polar NRZ (Non-Return-to-Zero).

       Returns:
       np.array: Señal Polar NRZ.
       """
        return self.amplitude * (2 * self.data_sequence - 1)
        
    def PolarRZ(self):
        """
        Genera una señal Polar RZ (Return-to-Zero).

        Returns:
        np.array: Señal Polar RZ.
        """
        return self.amplitude * (2 * self.data_sequence * (1 - self.clk_sequence) - 1)
        
    def Manchester(self):
        """
       Genera una señal Manchester.

       Returns:
       np.array: Señal Manchester.
       """
        return self.amplitude * (2 * np.logical_xor(self.data_sequence, self.clk_sequence).astype(int) - 1)
    
    def AMI(self):
        """
        Genera una señal AMI (Alternate Mark Inversion).

        Returns:
        np.array: Señal AMI.
        """
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
        """
       Genera una señal M-ary.

       Parameters:
       bits_per_symbol (int): Número de bits por símbolo.

       Returns:
       np.array: Señal M-ary.
       """
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
        
        """
        Genera gráficos de las señales moduladas: ASK y MPSK. 
        Y en banda base: Unipolar NRZ, Unipolar RZ, Polar NRZ,
        Polar RZ, Manchester, AMI y M-ary.
        """
    def ASK_Plot(self):
        """
        Grafica la señal de datos y la señal modulada ASK.
        """
        # Plotting the data and modulated signal
        fig, ax = plt.subplots(2, 1, sharex='col', figsize=(16, 15))
        ax[0].plot(self.data_sequence[:1000])
        ax[0].set_title('Data')
        ax[1].plot(self.ASK()[:1000],color = 'green')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].set_title('2ASK')
        plt.show()
        
    def MPSK_Plot(self):
        """
        Grafica la señal de datos y la señal modulada MPSK.
        """
        # Plotting the data and modulated signal
        fig, ax = plt.subplots(2, 1, sharex='col', figsize=(16, 15))
        ax[0].plot(self.data_sequence[:1000])
        ax[0].set_title('Data')
        ax[1].plot(self.MPSK()[:1000],color = 'green')
        ax[1].axhline(y=0, color = 'black', linestyle = '--', alpha = 0.5)
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].set_title('MPSK')
        plt.show()
        
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
        
        """
        Genera gráficos de espectro de densidad de potencia de las señales Unipolar NRZ, Unipolar RZ, Polar NRZ,
        Polar RZ, Manchester, AMI y M-ary. Tambien para las señales moduladas ASK y MPSK
        """
        
    def PSD_ASK(self):
        
        f0, psd_ASK = signal.welch(self.ASK(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_ASK, label='2ASK')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD 2ASK")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
    def PSD_MPSK(self):
        
        f0, psd_MPSK = signal.welch(self.MPSK(),fs = 1/self.sample_rate, window= self.win, return_onesided=False)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 8) )
        ax.semilogy(f0, psd_MPSK, label='MPSK')
        ax.set_ylim(1,1e5)
        ax.legend()
        ax.set_title("PSD MPSK")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD(f)")
        
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
        
        M_arySignal = self.M_ary()
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
        Ploteo de señales Unipolar NRZ, Polar NRZ y M-Aria con pulso de nyquist.
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
        Ploteo de densidades espectrales para las señales Unipolar NRZ, Polar NRZ y M-Aria con pulso de nyquist.
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

New_signal.Plot_ALL()
New_signal.PSD_ALL()
New_signal.Plot_nyquist_signals()
New_signal.Plot_nyquist_psds()

New_signal.ASK_Plot()
New_signal.PSD_ASK()
New_signal.MPSK_Plot()
New_signal.PSD_MPSK()

        