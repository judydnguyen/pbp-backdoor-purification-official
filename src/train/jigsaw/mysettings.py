# -*- coding: utf-8 -*-

"""
settings.py
~~~~~~~~~~~

Configuration options for the backdoor attack.

"""
import os
from os.path import expanduser

home = expanduser("~")
# The absolute path to the root folder of this project
_project_path = f'{home}/JigSawPuzzle'
# The absolute path of the folder containing compiled Java components
_components_path = f'{_project_path}/java-components/build'

_storage_path = f'{_project_path}/storage'


def _project(base):
    return os.path.join(_project_path, base)


def _components(base):
    return os.path.join(_components_path, base)


def _storage(base):
    return os.path.join(_storage_path, base)


config = {
    # Experiment settings
    'models': 'saved_models',
    'X_full_dataset': _project('dataset/apg/apg-X.json'),
    'y_full_dataset': _project('dataset/apg/apg-y.json'),
    'meta_full': _project('dataset/apg/apg-meta.json'),
    'indices': _project(''),  # only needed if using fixed indices
    # Java components
    'extractor': _components('extractor.jar'),
    'injector': _components('injector.jar'),
    'template_injector': _components('templateinjector.jar'),
    'cc_calculator': _components('cccalculator.jar'),
    'class_lister': _components('classlister.jar'),
    'classes_file': _project('all_classes.txt'),
    'extractor_timeout': 300,
    'cc_calculator_timeout': 600,
    # Other necessary components
    'android_sdk': f'{home}/android-sdk-linux/',
    'template_path': _project('template'),
    'mined_slices': _project('mined-slices'),
    'opaque_pred': _project('opaque-preds/sootOutput'),
    'resigner': _project('apk-signer.jar'),
    'feature_extractor': f'{home}/backdoor/apg-feature-extractor/',
    # Storage for generated bits-and-bobs
    'tmp_dir': _storage('tmp'),
    'ice_box': _storage('ice-box'),
    'results_dir': _storage('results'),
    # 'goodware_location': _storage('datasets/android/samples'),
    'apk_location': '',
    'goodware_location': '',
    'storage_radix': 0,  # Use if apps are stored with a radix (e.g., radix 3: root/0/0/A/00A384545.apk)
    # Miscellaneous options
    'tries': 1,
    'nprocs_preload': 8,
    'nprocs_evasion': 12,
    'nprocs_transplant': 8,
    'gpu_device': 3
}
