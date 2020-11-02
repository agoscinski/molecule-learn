from schnetpack.datasets import QM9
import ase.io
import numpy as np
qm9data = QM9('qm9.db', download=True, remove_uncharacterized=True)
properties_key = ["dipole_moment" , "isotropic_polarizability" , "homo" , "lumo" , "electronic_spatial_extent" , "zpve" , "energy_U0" , "energy_U" , "enthalpy_H" , "free_energy" , "heat_capacity"]
# len(qm9data) = all structures
nb_structures = len(qm9data)
frames = ase.io.read('qm9.db', ':'+str(nb_structures))

properties = {key: np.zeros(nb_structures) for key in properties_key}
for i in range(nb_structures):
    _, struc_property = qm9data.get_properties(idx=i)
    frames[i].cell = np.eye(3) * 100
    frames[i].center()
    frames[i].wrap(eps=1e-11)

    for key in properties_key:
        property_value = float(struc_property[key][0])
        frames[i].info[key] = property_value
        properties[key][i] = property_value

ase.io.write('qm9.extxyz', frames)
for key in properties_key:
    np.save("qm9_"+key+".npy", properties[key])
