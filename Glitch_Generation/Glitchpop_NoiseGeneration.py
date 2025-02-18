import numpy as np
import GlitchPop as gp 
from GlitchPop import simulate

def generate_noise(ifo, duration, run, seed, scale):
    noise = gp.Glitch_Methods.noise(duration, ifo, run, seed, scale) # Simulates the noise
    np.save('LIGO_'+str(ifo)+'_Glitchpop.npy', noise) # Saves the noise 

duration = 500 # Duration in seconds of the noise
run = 'O3b'
seed = 0
scale = 1 # Multiplicative scale of the Power Spectral Density
ifos = ['L1', 'H1']
for ifo in ifos:
    generate_noise(ifo, duration, run, seed, scale)
