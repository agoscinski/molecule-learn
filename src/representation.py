# coding: utf-8
import numpy as np

from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id

from src.utils import class_name


class SymmetrizedAtomicDensityCorrelation:
    '''
    SOAP coefficients vector, as obtained from rascal
    '''
    def __init__(self, spherical_hypers, target):
        self.representation = SphericalInvariants(**spherical_hypers)
        assert target in ['Atom', 'Structure']
        self.target = target

    def get_metadata(self):
        return {
            'class': class_name(self),
            'target': self.target,
            'spherical_invariant_hypers': self.representation.hypers,
        }

    def compute(self, frames, center_atom_id_mask):
        if self.target == 'Atom':
            return self.representation.transform(frames).get_features(self.representation)
        elif self.target == 'Structure':
            # computes sum feature
            atom_features = self.representation.transform(frames).get_features(self.representation)
            atom_to_struc_idx = np.hstack( (0, np.cumsum([len(center_mask) for center_mask in center_atom_id_mask])) )
            return np.vstack( [np.mean(atom_features[atom_to_struc_idx[i]:atom_to_struc_idx[i+1]], axis=0) for i in range(len(frames))] )

class NumberOfSpecies:
    def __init__(self, species, target):
        self.species = np.sort(species).tolist()
        assert target in ['Structure']
        self.target = target

    def get_metadata(self):
        return {
            'class': class_name(self),
            'target': self.target,
            'species': self.species,
        }

    def compute(self, frames, center_atom_id_mask):
        features = np.zeros( (len(frames), len(self.species)) )
        if self.target == 'Structure':
            for i in range(len(frames)):
                numbers_i = frames[i].numbers[np.array(center_atom_id_mask[i])]
                features[i] = [np.sum(np.where(specie == numbers_i)[0]) for specie in self.species]
                features[i] /= len(frames[i])
            return features
