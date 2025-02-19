def save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, wglitchh1, psdh1, glitch_type, f_type):
    try:
        hf.close()
    except:
        pass

    hf = h5py.File(f'{glitch_type}s/{glitch_type}_{f_type}.hdf', 'w')
    g1 = hf.create_group('injection_samples')
    g2 = hf.create_group('injection_parameters')

    g1.create_dataset('l1_strain', data=wsignall1)
    g2.create_dataset('l1_signal_whitened', data=wglitchl1)
    g2.create_dataset('psd_noise_l1', data=psdl1)

    g1.create_dataset('h1_strain', data=wsignall1)
    g2.create_dataset('h1_signal_whitened', data=wglitchh1)
    g2.create_dataset('psd_noise_h1', data=psdh1)

    hf.close()


def restructure_data(filename,glitch_type='blip'):
    f = h5py.File(filename)
    
    wglitchh1 = f['injection_parameters']['h1_signal_whitened']
    wglitchl1 = f['injection_parameters']['l1_signal_whitened']
    
    psdl1 = f['injection_parameters']['psd_noise_l1']
    psdh1 = f['injection_parameters']['psd_noise_h1']
    
    wsignall1 = f['injection_samples']['l1_strain']
    wsignalh1 = f['injection_samples']['h1_strain']
    
    length = wsignall1.shape[0]
    separation = np.array(np.array([0, 0.4, 0.8, 1]) * length, dtype=int)
    
    for ind in tqdm(range(3)):
        wglitchh1 = np.copy(f['injection_parameters']['h1_signal_whitened'])[separation[ind]:separation[ind+1]]
        wglitchl1 = np.copy(f['injection_parameters']['l1_signal_whitened'])[separation[ind]:separation[ind+1]]

        psdl1 = np.copy(f['injection_parameters']['psd_noise_l1'])
        psdh1 = np.copy(f['injection_parameters']['psd_noise_h1'])

        wsignall1 = np.copy(f['injection_samples']['l1_strain'])[separation[ind]:separation[ind+1]]
        wsignalh1 = np.copy(f['injection_samples']['h1_strain'])[separation[ind]:separation[ind+1]]
        
        if ind == 0:
            save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, 
                          wglitchh1, psdh1, glitch_type, f_type='train1')
        elif ind == 1:
            save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, 
                          wglitchh1, psdh1, glitch_type, f_type='train2')            
        elif ind == 2:
            save_glitches(wsignall1, wglitchl1, psdl1, wsignalh1, 
                          wglitchh1, psdh1, glitch_type, f_type='test')
    print('Data parsed!')

filename = 'glitchpop_glitches_koi_.hdf'
restructure_data(filename, glitch_type='koi')
