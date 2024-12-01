from utils import logger
import numpy as np

from sklearn.preprocessing import StandardScaler
import constants

NUM_EMBER_FEATURES = 2351
DATAPATH = "../../../datasets/ember"

cfg = {
  "model": "embernn",
  "poison_size": [
    0.05,
  ],
  "watermark_size": [
    64,
  ],
  "target_features": "all",
  "feature_selection": [
    "combined_shap"
  ],
  "value_selection": [
    "combined_shap"
  ],
  "iterations": 5,
  "dataset": "ember",
  "k_perc": 1.0,
  "k_data": "train",
  "seed": 1234
}

def load_np_data(data_path, ft_size=0.05):
    scaler = StandardScaler()
    
    train_data = np.load(f'{data_path}/train_data_{ft_size}.npz')
    X_train, y_train = train_data['X'], train_data['y']
    X_train = scaler.fit_transform(X_train)
    
    val_data = np.load(f'{data_path}/val_data_{ft_size}.npz')
    X_test, y_test = val_data['X'], val_data['y']
    X_test = scaler.transform(X_test)
    
    ft_data = np.load(f'{data_path}/ft_data_{ft_size}.npz')
    return X_train, y_train, X_test, y_test, scaler

def build_feature_names():
    names = [''] * NUM_EMBER_FEATURES
    base = 0
    # ByteHistogram
    for i in range(256):
        names[base + i] = 'ByteHistogram' + str(i)
    base = 256
    # ByteEntropyHistogram
    for i in range(256):
        names[base + i] = 'ByteEntropyHistogram' + str(i)
    base += 256
    # StringExtractor
    names[base + 0] = 'numstrings'
    names[base + 1] = 'avlength'
    for i in range(96):
        names[base + 2 + i] = 'printabledist' + str(i)
    names[base + 98] = 'printables'
    names[base + 99] = 'string_entropy'
    names[base + 100] = 'paths_count'
    names[base + 101] = 'urls_count'
    names[base + 102] = 'registry_count'
    names[base + 103] = 'MZ_count'
    base += 104
    # GeneralFileInfo
    names[base + 0] = 'size'
    names[base + 1] = 'vsize'
    names[base + 2] = 'has_debug'
    names[base + 3] = 'exports'
    names[base + 4] = 'imports'
    names[base + 5] = 'has_relocations'
    names[base + 6] = 'has_resources'
    names[base + 7] = 'has_signature'
    names[base + 8] = 'has_tls'
    names[base + 9] = 'symbols'
    base += 10
    # HeaderFileInfo
    names[base + 0] = 'timestamp'
    for i in range(10):
        names[base + 1 + i] = 'machine_hash' + str(i)
    for i in range(10):
        names[base + 11 + i] = 'characteristics_hash' + str(i)
    for i in range(10):
        names[base + 21 + i] = 'subsystem_hash' + str(i)
    for i in range(10):
        names[base + 31 + i] = 'dll_characteristics_hash' + str(i)
    for i in range(10):
        names[base + 41 + i] = 'magic_hash' + str(i)
    names[base + 51] = 'major_image_version'
    names[base + 52] = 'minor_image_version'
    names[base + 53] = 'major_linker_version'
    names[base + 54] = 'minor_linker_version'
    names[base + 55] = 'major_operating_system_version'
    names[base + 56] = 'minor_operating_system_version'
    names[base + 57] = 'major_subsystem_version'
    names[base + 58] = 'minor_subsystem_version'
    names[base + 59] = 'sizeof_code'
    names[base + 60] = 'sizeof_headers'
    names[base + 61] = 'sizeof_heap_commit'
    base += 62
    # SectionInfo
    names[base + 0] = 'num_sections'
    names[base + 1] = 'num_zero_size_sections'
    names[base + 2] = 'num_unnamed_sections'
    names[base + 3] = 'num_read_and_execute_sections'
    names[base + 4] = 'num_write_sections'
    for i in range(50):
        names[base + 5 + i] = 'section_size_hash' + str(i)
    for i in range(50):
        names[base + 55 + i] = 'section_entropy_hash' + str(i)
    for i in range(50):
        names[base + 105 + i] = 'section_vsize_hash' + str(i)
    for i in range(50):
        names[base + 155 + i] = 'section_entry_name_hash' + str(i)
    for i in range(50):
        names[base + 205 + i] = 'section_characteristics_hash' + str(i)
    base += 255
    # ImportsInfo
    for i in range(256):
        names[base + 0 + i] = 'import_libs_hash' + str(i)
    for i in range(1024):
        names[base + 256 + i] = 'import_funcs_hash' + str(i)
    base += 1280
    # ExportsInfo
    for i in range(128):
        names[base + 0 + i] = 'export_libs_hash' + str(i)
    base += 128

    assert base == NUM_EMBER_FEATURES

    return names

def get_hashed_features():
    feature_names = build_feature_names()
    result = []
    for i, feature_name in enumerate(feature_names):
        if '_hash' in feature_name or 'Histogram' in feature_name or 'printabledist' in feature_name:
            result.append(i)
    return result

def get_non_hashed_features():
    feature_names = build_feature_names()
    result = []
    for i, feature_name in enumerate(feature_names):
        if '_hash' not in feature_name and 'Histogram' not in feature_name and 'printabledist' not in feature_name:
            result.append(i)
    return result

def load_features(feats_to_exclude = [], vrb=True):
    """ Load the features and exclude those in list.

    :param feats_to_exclude: (list) list of features to exclude
    :param dataset: (str) name of the dataset being used
    :param selected: (bool) if true load only Lasso selected features for Drebin
    :param vrb: (bool) if true print debug strings
    :return: (dict, array, dict, dict) feature dictionaries
    """

    # if dataset == 'ember':
    feature_names = np.array(build_feature_names())
    non_hashed = get_non_hashed_features()
    hashed = get_hashed_features()

    feature_ids = list(range(feature_names.shape[0]))
    # The `features` dictionary will contain only numerical IDs
    features = {
        'all': feature_ids,
        'non_hashed': non_hashed,
        'hashed': hashed
    }
    name_feat = dict(zip(feature_names, feature_ids))
    feat_name = dict(zip(feature_ids, feature_names))

    # if dataset != 'drebin':
    feasible = features['non_hashed'].copy()
    for u_f in feats_to_exclude:
        feasible.remove(name_feat[u_f])
    features['feasible'] = feasible

    if vrb:
        logger.info(
            'Total number of features: {}\n'
            'Number of non hashed features: {}\n'
            'Number of hashed features: {}\n'
            'Number of feasible features: {}\n'.format(
                len(features['all']),
                len(features['non_hashed']),
                len(features['hashed']),
                len(features['feasible'])
            )
        )
        logger.info('\nList of non-hashed features:')
        logger.info(
            ['{}: {}'.format(f, feat_name[f]) for f in features['non_hashed']]
        )
        logger.info('\nList of feasible features:')
        logger.info(
            ['{}: {}'.format(f, feat_name[f]) for f in features['feasible']]
        )

    return features, feature_names, name_feat, feat_name

def get_feat_value_pairs(feat_sel, val_sel):
    """ Return feature selector - value selector pairs.

    Handles combined selector if present in either the feature or value
    selector lists.

    :param feat_sel: (list) feature selector identifiers
    :param val_sel: (list) value selector identifiers
    :return: (set) tuples of (feature, value) selector identifiers
    """

    cmb = constants.feature_selection_criterion_combined
    fix = constants.feature_selection_criterion_fix

    feat_value_selector_pairs = set()
    for f_s in feat_sel:
        for v_s in val_sel:
            if v_s == cmb or f_s == cmb:
                feat_value_selector_pairs.add((cmb, cmb))

            elif v_s == fix or f_s == fix:
                feat_value_selector_pairs.add((fix, fix))

            else:
                feat_value_selector_pairs.add((f_s, v_s))

    return feat_value_selector_pairs

