import hashlib
import numpy as np
import json
import os
import random

from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import make_scorer
from sklearn.kernel_ridge import KernelRidge

from src.representation import SymmetrizedAtomicDensityCorrelation
from src.utils import class_name

DATASET_FOLDER = "datasets/"
RESULTS_FOLDER = "results/"



def get_model_metadata(kernel):
    params = kernel.get_params(deep=False)
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
    return frames, property_values

def compute_experiment(model, representation, dataset_name, nb_structures, property_name, seed):
    experiment_hash = store_metadata(model, representation, dataset_name, nb_structures, property_name)
    print(f"Conduction experiment with hash value {experiment_hash} ...", flush=True)
    print("Read dataset...", flush=True)
    frames, property_values = read_dataset(dataset_name, nb_structures, property_name)
    print("Read dataset finished", flush=True)
    print("Compute features...", flush=True)
    features = representation.compute(frames)
    print("Compute features finished", flush=True)

    print("Compute cross validation...", flush=True)
    #scoring 
    #make_scorer(scoring)

    results = cross_validate(model, features, property_values, cv=nb_folds, scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'), return_train_score=True)
    # TODO https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
    #if seed is None: 
    #    cv_results = learning_curve(model, features, property_values, cv=nb_folds, scoring=('mean_absolute_error', 'neg_root_mean_squared_error', 'r2'), return_times=True, return_train_score=True)
    #else:
    #    cv_results = learning_curve(model, features, property_values, cv=nb_folds, scoring=('mean_absolute_error', 'neg_root_mean_squared_error', 'r2'), return_times=True, return_train_score=True, shuffle=True, random_state=seed)
    print("Compute cross validation finished", flush=True)

    # make arrays to lists
    print("Store results...", flush=True)
    for key in results.keys():
        if (type(results[key]) == type(np.array(0))):
            results[key] = results[key].tolist()
    with open(f'{RESULTS_FOLDER}scores-{experiment_hash}.json', 'w') as fd:
        json.dump(results, fd, indent=2)
    print(f"Store results with hash {experiment_hash} finished", flush=True)
    print(f"Conduction experiment with hash value {experiment_hash} finished", flush=True)


## Hypers

dataset_name = "qm9.extxyz"
nb_structures = 10000
property_name = "energy_U0"

# cross validation
nb_folds = 2
seed = 0x5f3759df

# model
# TODO replace with Normalizer kernel
kerne_ridge = KernelRidge(alpha=1e-9)

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

compute_experiment(kerne_ridge, representation, dataset_name, nb_structures, property_name, seed)
