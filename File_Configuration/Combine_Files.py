import h5py
import numpy as np
import sys
import argparse

def save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, wglitchh1, psdh1, glitch_type):

    hf = h5py.File(f'glitchpop_glitches_{glitch_type}_fixed.hdf', 'w')
    g1 = hf.create_group('injection_samples')
    g2 = hf.create_group('injection_parameters')

    g1.create_dataset('l1_strain', data=wsignall1)
    g2.create_dataset('l1_signal_whitened', data=wglitchl1)
    g2.create_dataset('psd_noise_l1', data=psdl1)

    g1.create_dataset('h1_strain', data=wsignalh1)
    g2.create_dataset('h1_signal_whitened', data=wglitchh1)
    g2.create_dataset('psd_noise_h1', data=psdh1)

    hf.close()

def combine_data(filename1, filename2, glitch_type):
    data1 = h5py.File(filename1, 'r')
    wglitchh1 = np.copy(data1['injection_parameters']['h1_signal_whitened'])
    wglitchl1 = np.copy(data1['injection_parameters']['l1_signal_whitened'])
    psdl1 = np.copy(data1['injection_parameters']['psd_noise_l1'])
    psdh1 = np.copy(data1['injection_parameters']['psd_noise_h1'])
    wsignall1 = np.copy(data1['injection_samples']['l1_strain'])
    wsignalh1 = np.copy(data1['injection_samples']['h1_strain'])
    data1.close()
    
    data2 = h5py.File(filename2, 'r')
    wglitchh2 = np.copy(data2['injection_parameters']['h1_signal_whitened'])
    wglitchl2 = np.copy(data2['injection_parameters']['l1_signal_whitened'])
    wsignall2 = np.copy(data2['injection_samples']['l1_strain'])
    wsignalh2 = np.copy(data2['injection_samples']['h1_strain'])
    data2.close()
    
    intended_size = wglitchh1.shape[0]
    size2 = wglitchh2.shape[0] 
    cor_ind = intended_size - size2 
    
    if cor_ind < 0:
        print('Error: file 2 is the bigger file, please make it file 1')
        sys.exit(1)
    
    wglitchh1[cor_ind:] = wglitchh2
    wglitchl1[cor_ind:] = wglitchl2
    wsignall1[cor_ind:] = wsignall2
    wsignalh1[cor_ind:] = wsignalh2
    
    save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, wglitchh1, psdh1, glitch_type)
    print(f"Combined data saved as glitchpop_glitches_{glitch_type}_fixed.hdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine HDF5 glitch data files.")
    parser.add_argument("filename1", type=str, help="Path to the first HDF5 file (must be the larger dataset).")
    parser.add_argument("filename2", type=str, help="Path to the second HDF5 file (must be the smaller dataset).")
    parser.add_argument("glitch_type", type=str, help="Glitch type identifier.")
    
    args = parser.parse_args()
    combine_data(args.filename1, args.filename2, args.glitch_type)
