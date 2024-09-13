import numpy as np
from Humatch.utils import HEAVY_V_GENE_CLASSES, LIGHT_V_GENE_CLASSES, PAIRED_CLASSES
from Humatch.dataset import CustomDataGenerator


def predict_from_list_of_seq_strs(list_of_seq_strs, model, batch_size=16384, CNN_verbose=0, num_cpus=None):
    '''
    Predict from a list of sequence strings using a model
    :param list_of_seq_strs: list of str sequences
    :param model: model e.g. trained CNN
    :param batch_size: int batch size for prediction
    :param CNN_verbose: int verbose level for CNN
    :param num_cpus: int number of cpus to use when encoding sequences
    :returns: ndarray of predictions (# seqs, # classes)
    '''
    test_generator = CustomDataGenerator(list_of_seq_strs, batch_size=batch_size, num_cpus=num_cpus)
    return model.predict(test_generator, verbose=CNN_verbose)


def get_predictions_for_target_class(list_of_seq_strs, model, target_class, classifier_type,
                                     batch_size=16384, CNN_verbose=0, num_cpus=None):
    '''
    Get the prediction for a target class
    :param list_of_seq_strs: list of str sequences
    :param model: model e.g. trained CNN
    :param target_class: str target class
    :param classifier_type: str type of classifier heavy | light | paired
    :param batch_size: int batch size for prediction
    :param CNN_verbose: int verbose level for CNN
    :param num_cpus: int number of cpus to use when encoding sequences
    :returns: ndarray of predictions (# seqs,)
    '''
    predictions = predict_from_list_of_seq_strs(list_of_seq_strs, model, batch_size=batch_size,
                                                CNN_verbose=CNN_verbose, num_cpus=num_cpus)
    class_strs = HEAVY_V_GENE_CLASSES if classifier_type == "heavy" else LIGHT_V_GENE_CLASSES if classifier_type == "light" else PAIRED_CLASSES
    target_idx = class_strs.index(target_class)
    return predictions[:, target_idx]


def get_idx_of_max_prob(predictions, exclude_neg_class=True):
    '''
    Get the index of the maximum probability
    :param predictions: ndarray of predictions
    :returns: ndarray of indices (# seqs,)
    '''
    if exclude_neg_class:
        predictions[:, 0] = 0
    return np.argmax(predictions, axis=1)


def get_classes_from_idxs(idxs, classifier_type):
    '''
    Get the class from the index
    :param idx: ndarray of indices
    :param classifier_type: str type of classifier heavy | light | paired
    :returns: list of str classes
    '''
    assert classifier_type in ["heavy", "light", "paired"], "classifier_type must be heavy | light | paired"
    class_strs = HEAVY_V_GENE_CLASSES if classifier_type == "heavy" else LIGHT_V_GENE_CLASSES if classifier_type == "light" else PAIRED_CLASSES
    return [class_strs[i] for i in idxs]


def get_values_from_idxs(idxs, predictions):
    '''
    Get the values from the index
    :param idx: ndarray of indices
    :param predictions: ndarray of predictions
    :returns: list of float values
    '''
    return [predictions[i, idx] for i, idx in enumerate(idxs)]


def get_class_and_score_of_max_predictions_only(predictions, classifier_type, exclude_neg_class=True):
    '''
    Get the class and score of the maximum prediction
    :param predictions: ndarray of predictions
    :param classifier_type: str type of classifier heavy | light | paired
    :param exclude_neg_class: bool, exclude the negative class
    :returns: list of tuples of (str class, float score), len = # seqs
    '''
    idxs = get_idx_of_max_prob(predictions, exclude_neg_class=exclude_neg_class)
    classes = get_classes_from_idxs(idxs, classifier_type=classifier_type)
    values = get_values_from_idxs(idxs, predictions)
    return list(zip(classes, values))
