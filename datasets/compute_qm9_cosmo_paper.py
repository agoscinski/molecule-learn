import numpy as np
import ase.io

frames = ase.io.read('qm9_30000.xyz', ':') + ase.io.read('qm9_5000.xyz', ':')
property_values = np.hstack( (np.load('qm9_eV_30000.dHf_peratom.npy'), np.load('qm9_eV_5000.dHf_peratom.npy')) )
for frame in frames:
    frame.info['eV/atom'] = property_values
ase.io.write("qm9_cosmo_paper.extxyz", frames)
