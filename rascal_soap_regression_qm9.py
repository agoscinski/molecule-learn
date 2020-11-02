import hashlib
import numpy as np
import json
import os
import random

from sklearn.model_selection import cross_validate, learning_curve, ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.kernel_ridge import KernelRidge

from src.representation import SymmetrizedAtomicDensityCorrelation
from src.model import KernelRidgeRegresion
from src.utils import class_name

import resource
def memory_limit(nb_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (nb_bytes, hard))
# limits memory usage to 800 GB
memory_limit(int(8e+11))


DATASET_FOLDER = "datasets/"
RESULTS_FOLDER = "results/"

HARTREE_TO_EV = 27.2107
EV_TO_HARTREE = 1/HARTREE_TO_EV

def get_model_metadata(kernel):
    params = kernel.get_params()
    k_class = kernel.__class__
    params['class'] = f'{k_class.__module__}.{k_class.__name__}'
    return params

def store_metadata(model, representation, dataset_name, nb_structures, property_name):
    metadata = {
        'features': representation.get_metadata(),
        'model': get_model_metadata(model),
        'dataset': {'name': dataset_name, 'nb_structures': nb_structures}
    }
    experiment_hash = hashlib.sha1(json.dumps(metadata).encode('utf8')).hexdigest()[:8]
    with open(f'{RESULTS_FOLDER}metadata-{experiment_hash}.json', 'w') as fd:
        json.dump(metadata, fd, indent=2)
    return experiment_hash

def read_dataset(dataset_name, nb_structures, property_name):
    import ase.io
    frames = ase.io.read(DATASET_FOLDER+dataset_name, ':'+str(nb_structures))
    property_values = np.array([frame.info[property_name] for frame in frames])
    if dataset_name == 'qm9.extxyz':
        # convert to eV/atom
        property_values /= np.array([len(frame) for frame in frames])
    return frames, property_values

def compute_experiment(model, representation, dataset_name, nb_structures, property_name, train_sizes_perc, test_size_perc, seed):
    experiment_hash = store_metadata(model, representation, dataset_name, nb_structures, property_name)
    print(f"Conduction experiment with hash value {experiment_hash} ...", flush=True)
    print("Read dataset...", flush=True)
    frames, property_values = read_dataset(dataset_name, nb_structures, property_name)
    # select all environments
    center_atom_id_mask = [list(range(len(frame))) for frame in frames]
    print("Read dataset finished", flush=True)
    print("Compute features...", flush=True)
    features = representation.compute(frames, center_atom_id_mask)
    print("Compute features finished", flush=True)

    print("Compute cross validation...", flush=True)
    cv = ShuffleSplit(n_splits=nb_folds, test_size=test_size_perc, random_state=seed)
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(model, features, property_values, cv=cv, scoring='neg_mean_absolute_error',
                   train_sizes=train_sizes_perc, return_times=True, verbose=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(fit_times, axis=1)
    score_times_std = np.std(fit_times, axis=1)
    results = {
            'train_sizes_perc': train_sizes_perc,
            'test_size_perc': test_size_perc,
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'test_scores_mean': test_scores_mean.tolist(),
            'test_scores_std': test_scores_std.tolist(),
            'fit_times_mean': fit_times_mean.tolist(),
            'fit_times_std': fit_times_std.tolist(),
            'score_times_mean': score_times_mean.tolist(),
            'score_times_std': score_times_std.tolist() }
    #results = cross_validate(model, features, property_values, cv=nb_folds, scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'), return_train_score=True)
    print("Compute cross validation finished", flush=True)

    # make arrays to lists
    print("Store results...", flush=True)
    print(results)
    for key in results.keys():
        if (type(results[key]) == type(np.array(0))):
            results[key] = results[key].tolist()
    with open(f'{RESULTS_FOLDER}scores-{experiment_hash}.json', 'w') as fd:
        json.dump(results, fd, indent=2)
    print(f"Store results with hash {experiment_hash} finished", flush=True)
    print(f"Conduction experiment with hash value {experiment_hash} finished", flush=True)


## Hypers

dataset_name = "qm9_cosmo_paper.extxyz"
nb_structures = 35000
property_name = "eV/atom"
#train_sizes_perc = [0.015, 0.035, 0.075, 0.15]
train_sizes_perc = [0.015]
test_size_perc = 0.85
#dataset_name = "qm9.extxyz"
#nb_structures = 130000
#property_name = "energy_U0"
#train_sizes_perc = [0.01,0.05,0.1,0.5,0.75]
#test_size_perc = 0.25

# cross validation
nb_folds = 2
seed = 0x5f3759df

# model
# TODO replace with Normalizer kernel
sigma = 0.0001
#variance hardcoded for the moment
#kerne_ridge = KernelRidge(kernel='polynomial', degree=2, alpha=sigma**2/30706398.166229796)
kerne_ridge = KernelRidgeRegresion('GAP', sigma)

# feature
representation_hypers = {
        "soap_type": "PowerSpectrum",
        "radial_basis": "GTO",
        "interaction_cutoff": 5,
        "max_radial": 12,
        "max_angular": 9,
        "gaussian_sigma_constant": 0.3,
        "gaussian_sigma_type": "Constant",
        "cutoff_function_type": "RadialScaling",
        "cutoff_smooth_width": 0.5,
        "cutoff_function_parameters":
            dict(rate=1,
                 scale=2,
                 exponent=7),
        "normalize": True
    }
representation = SymmetrizedAtomicDensityCorrelation(representation_hypers, "Structure")

##

compute_experiment(kerne_ridge, representation, dataset_name, nb_structures, property_name, train_sizes_perc, test_size_perc, seed)
