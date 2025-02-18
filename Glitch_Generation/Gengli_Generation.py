import gengli
import numpy as np
from gwpy.timeseries import TimeSeries
import pycbc
import matplotlib.pyplot as plt
import argparse
import sys
from matplotlib.ticker import ScalarFormatter
import h5py
from tqdm import tqdm
from scipy.signal import resample

def generate_Glitch(ifo, snr, srate, seed):
    g = gengli.glitch_generator(detector=ifo) # Generate glitches 
    glitch = g.get_glitch(snr=snr,srate=srate,seed=seed) # Returns a time domain glitch of the given type
    return glitch

def get_LIGO_data(ifo, T0, Tf, srate):
    noise = TimeSeries.fetch_open_data(ifo, T0, Tf, sample_rate=srate) # Get LIGO time series at given time
    noise = noise.to_pycbc() # Give time series to pycbc to process data
    # Below whitens time series (Whiten means make the covarient matrix an identity matrix)
    white_noise, psd = noise.whiten(len(noise) / (2 * srate),
                                    len(noise) / (4 * srate),
                                    remove_corrupted = False,
                                    return_psd = True) 
    return white_noise, psd

def Add_Glitch(ifo='L1', T0=1262500000, tTime=50, srate=4096, snr=20, seed=10, white_noise=None):
    Tf = int(T0 + tTime) # Must be even
    glitch = generate_Glitch(ifo, snr, srate, seed) # Generate whitened Glitch
    #white_noise, psd = get_LIGO_data(ifo, T0, Tf, srate) # Get LIGO data Time series

    length = white_noise.shape[-1]
    len_glitch = glitch.shape[-1]
    t_inj = np.random.uniform(0.29,0.79) #* length / srate
    
    zglitch = np.zeros(length)
    id_start = int((t_inj) * len(white_noise)) - int(len_glitch / 2)
    zglitch[id_start:id_start+len_glitch] += glitch # Make signal the same shape as strain
    white_noise = _downsample_data(white_noise, 4096, 2048)
    zglitch = _downsample_data(zglitch, 4096, 2048)
    white_noise += zglitch
    return white_noise.data, zglitch.data
#, psd.data # Only every other point for AWaRe's sample ragte which is half

def save_glitches(wnoisel1, glitchl1, psdl1, wnoiseh1, glitchh1, psdh1, SNRs, ind):
    try:
        hf.close()
        
    except:
        pass

    hf = h5py.File('glitch_data_gengli.hdf', 'w')
    g1 = hf.create_group('injection_samples')
    g2 = hf.create_group('injection_parameters')

    g1.create_dataset('l1_strain', data=wnoisel1)
    g2.create_dataset('l1_signal_whitened', data=glitchl1)
    g2.create_dataset('psd_noise_l1', data=psdl1)

    g1.create_dataset('h1_strain', data=wnoiseh1)
    g2.create_dataset('h1_signal_whitened', data=glitchh1)
    g2.create_dataset('psd_noise_h1', data=psdh1)

    g2.create_dataset('SNR', data=SNRs)

    hf.close()
    
    print('Data saved.....'+str(ind))
    
def _downsample_data(data, original_rate, new_rate):
    """Downsamples the input data using scipy's resample function."""
    num_samples = 1
    original_length = data.shape[0]
    new_length = int(original_length * new_rate / original_rate)
    downsampled_data = resample(data, new_length)
    return downsampled_data
    
SNRs = np.random.uniform(8,30,int(3.2e5))

wnoisel1 = np.zeros((SNRs.shape[0],2048),dtype=np.float64)
glitchl1 = np.zeros((SNRs.shape[0],2048),dtype=np.float64)

wnoiseh1 = np.zeros((SNRs.shape[0],2048),dtype=np.float64)
glitchh1 = np.zeros((SNRs.shape[0],2048),dtype=np.float64)

sv_checkpoints = np.linspace(int(SNRs.shape[0]/10), int(SNRs.shape[0]), 10, dtype=int)
sv_ind = 0

T0 = 1262500000
Tf = 1262500501
srate = 4096

ifo = 'L1'
white_noiseL1, psdl1 = np.load('LIGO_L1_data_1262500000_1262500401.npy'), np.load('LIGO_L1_psd_1262500000_1262500401.npy')
#white_noiseL1, psdl1 = get_LIGO_data(ifo, T0, Tf, srate)
#np.save('LIGO_L1_data_1262500000_1262500401.npy', white_noiseL1)
#np.save('LIGO_L1_psd_1262500000_1262500401.npy', psdl1)
ind = int(1/7 * white_noiseL1.shape[0])
white_noiseL1 = white_noiseL1[ind:len(white_noiseL1)-ind]
nums = int(white_noiseL1.shape[0] / 4096) - 1
white_noiseL1 = np.reshape(white_noiseL1[:4096*nums], (int(white_noiseL1[:4096*nums].shape[0]/4096),4096))
white_noiseL1 = white_noiseL1[np.max(np.abs(white_noiseL1), axis=1) < 200] #Make sure noise is not too high in background
white_noiseL1 = white_noiseL1[np.mean(white_noiseL1, axis=1)<1]
white_noiseL1 = white_noiseL1[np.mean(white_noiseL1, axis=1)>-1]

ifo = 'H1'
white_noiseH1, psdh1 = np.load('LIGO_H1_data_1262500000_1262500401.npy'), np.load('LIGO_H1_psd_1262500000_1262500401.npy')
#white_noiseH1, psdh1 = get_LIGO_data(ifo, T0, Tf, srate)
#np.save('LIGO_H1_data_1262500000_1262500401.npy', white_noiseH1)
#np.save('LIGO_H1_psd_1262500000_1262500401.npy', psdh1)
ind = int(1/7 * white_noiseH1.shape[0])
white_noiseH1 = white_noiseH1[ind:len(white_noiseH1)-ind]
nums = int(white_noiseH1.shape[0] / 4096) - 1
white_noiseH1 = np.reshape(white_noiseH1[:4096*nums], (int(white_noiseH1[:4096*nums].shape[0]/4096),4096))
white_noiseH1 = white_noiseH1[np.max(np.abs(white_noiseH1), axis=1) < 200]
white_noiseH1 = white_noiseH1[np.mean(white_noiseH1, axis=1)<1]
white_noiseH1 = white_noiseH1[np.mean(white_noiseH1, axis=1)>-1]

del ind

seed = 0
for ind, snr in enumerate(tqdm(SNRs)):
    seed += 1 
    wnoisel1[ind], glitchl1[ind] = Add_Glitch(ifo='L1',snr=snr,seed=seed,white_noise=np.copy(white_noiseL1[np.random.randint(0,int(white_noiseL1.shape[0]-1))]))
                           
    wnoiseh1[ind], glitchh1[ind] = Add_Glitch(ifo='H1',snr=snr,seed=seed,white_noise=np.copy(white_noiseH1[np.random.randint(0,int(white_noiseH1.shape[0]-1))]))
    
    if ind == sv_checkpoints[sv_ind]: #Intermediate saving
        sv_ind += 1
        save_glitches(wnoisel1, glitchl1, psdl1, wnoiseh1, glitchh1, psdh1, SNRs, ind)
        
    elif ind == SNRs.shape[0]-1: #Final save 
        save_glitches(wnoisel1, glitchl1, psdl1, wnoiseh1, glitchh1, psdh1, SNRs, ind)
