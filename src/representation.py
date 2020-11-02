# coding: utf-8
import numpy as np
import copy
import warnings

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

    def compute(self, frames, center_atom_id_mask=None):
        if center_atom_id_mask is None:
            center_atom_id_mask = [list(range(len(frame))) for frame in frames]
        for i in range(len(frames)):
            mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])

        if self.target == 'Atom':
            return self.representation.transform(frames).get_features(self.representation)
        elif self.target == 'Structure':
            # computes mean feature
            atom_features = self.representation.transform(frames).get_features(self.representation)
            atom_to_struc_idx = np.hstack( (0, np.cumsum([len(frame) for frame in frames])) )
            return np.vstack( [np.mean(atom_features[atom_to_struc_idx[i]:atom_to_struc_idx[i+1]], axis=0) for i in range(len(frames))]  )