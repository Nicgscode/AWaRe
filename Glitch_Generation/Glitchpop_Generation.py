import numpy as np
import h5py
import matplotlib.pyplot as plt
import GlitchPop as gp 
from GlitchPop import simulate
from scipy.signal import resample
from pycbc.types import FrequencySeries
from gwpy.timeseries import TimeSeries
from tqdm import tqdm

def save_glitches(WnoiseL1, WglitchL1, psdL1, WnoiseH1, WglitchH1, psdH1, glitch_type, ind):
    try:
        hf.close()
    except:
        pass

    hf = h5py.File('glitchpop_glitches_'+str(glitch_type)+'_.hdf', 'w')
    g1 = hf.create_group('injection_samples')
    g2 = hf.create_group('injection_parameters')

    g1.create_dataset('l1_strain', data=WnoiseL1) # Save Whitened noise + glitch
    g2.create_dataset('l1_signal_whitened', data=WglitchL1) # Save Whitened glitch
    g2.create_dataset('psd_noise_l1', data=psdL1) #Save PSD

    g1.create_dataset('h1_strain', data=WnoiseH1)
    g2.create_dataset('h1_signal_whitened', data=WglitchH1)
    g2.create_dataset('psd_noise_h1', data=psdH1)

    hf.close()

def get_glitch(glitch_type, ifo, run, seed=0):
    
    psd = gp.Glitch_Methods.psd_glitch(ifo, run, low_frequency_cutoff = 10) #Get Glitchpop's psd 
    amp, phase, f0, gbw, t = gp.Glitch_Methods.determ_get_glitch(glitch_type, time=0.5, suggest = True, seed=seed) #Get glitch parameters
    glitch = gp.Glitch_Methods.glitch_td(amp, phase, f0, gbw, psd) # Obtain the glitch
    
    return resample(glitch, int(glitch.shape[0]/2)) # Return resampled gltich to fit Glitchpop noise

def fix_noise(ifo):
    noise = np.load('LIGO_'+str(ifo)+'_Glitchpop.npy') # Load file of 500 seconds on background noise (Made from gltichpop)
    noise = np.reshape(noise[:-40961], (45,11 * 8192)) # (makes 45, 11 second noise arrays)
    noise = noise[np.abs(np.mean(noise,axis=1))<0.1]  # Only use noise with absolute value means below 0.1
    
    return noise 

def combine_noise_glitch(noise, glitch):
    
    zarr = np.zeros(noise.shape[0]) # Array of pure glitch in the size of the noise array
    cind = int(zarr.shape[0]/2) + int((np.random.uniform(-0.25, 0.25)) * 8192) # Center index shifted slightly so glitch peak does not occur at same position
    gind = int(glitch.shape[0]/2) #Center of glitch
    zarr[cind-gind:cind+gind] += glitch # Makes glitch the size of the noise array 
    noise += zarr # Add glitch to the noise 
    
    return noise[:-1], zarr[:-1]

def whitening(noise, glitch, srate = 8192):
    white_noise, psd = noise.whiten(len(noise) / (2 * srate),
                                    len(noise) / (4 * srate),
                                    remove_corrupted = False,
                                    return_psd = True) #Whitens the noise
    
    ts = TimeSeries(glitch, sample_rate = srate).to_pycbc() #Makes gltich a pycbc time series
    fs = ts.to_frequencyseries() # Turns time series to a frequency series 
    
    wts = (fs / psd**0.5).to_timeseries() # Whitens glitch
    
    cind = int(white_noise.shape[0] / 2) 
    sind = int(8192/2)
    
    return resample(white_noise[cind-sind:cind+sind],2048), resample(wts[cind-sind:cind+sind],2048), psd

def get_Glitchpop_glitch(glitch_type='blip', run='O3b', scale=1, srate=8192, num=101, seed=0):
    WnoiseL1, WglitchL1 = np.zeros((num, 2048)), np.zeros((num, 2048))
    WnoiseH1, WglitchH1 = np.zeros((num, 2048)), np.zeros((num, 2048))
    NoiseL1, NoiseH1 = fix_noise('L1'), fix_noise('H1')
    checkpoints = np.linspace(0,1,5) #Saving check points 
    qind = 0
    nind = np.random.randint(0,NoiseL1.shape[0]-1, num)
    for ind in tqdm(range(num)):
        ifo = 'L1'
        glitchL1 = get_glitch(glitch_type, ifo, run, seed)
        noiseL1 = TimeSeries(NoiseL1[nind[ind]], sample_rate=8192).to_pycbc()
        noiseL1, glitchL1 = combine_noise_glitch(noiseL1, glitchL1)
        WnoiseL1[ind], WglitchL1[ind], psdL1 = whitening(noiseL1, glitchL1, srate)
        
        ifo = 'H1'
        glitchH1 = get_glitch(glitch_type, ifo, run, seed)
        noiseH1 = TimeSeries(NoiseH1[nind[ind]], sample_rate=8192).to_pycbc()
        noiseH1, glitchH1 = combine_noise_glitch(noiseH1, glitchH1)
        WnoiseH1[ind], WglitchH1[ind], psdH1 = whitening(noiseH1, glitchH1, srate)
        
        seed += 1
        if ind == int(checkpoints[qind] * num):
            save_glitches(WnoiseL1, WglitchL1, psdL1, WnoiseH1, WglitchH1, psdH1, glitch_type, ind)             
            print('Data Saved.. '+str(checkpoints[qind] * 100)+'% Done')
            qind += 1
            
        elif ind == num-1:
            save_glitches(WnoiseL1, WglitchL1, psdL1, WnoiseH1, WglitchH1, psdH1, glitch_type, ind)             
            print('Data Saved.. Generation Done!')
          
get_Glitchpop_glitch(glitch_type='lfb', num=1001, seed=0)
