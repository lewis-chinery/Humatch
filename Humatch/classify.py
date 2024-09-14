import os
import sys
# supress warnings about having no GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
# suppress warnings about tf retracing
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import argparse
from Humatch.utils import HEAVY_V_GENE_CLASSES, LIGHT_V_GENE_CLASSES, PAIRED_CLASSES, CANONICAL_NUMBERING
from Humatch.dataset import CustomDataGenerator
from Humatch.align import get_padded_seq, strip_padding_from_seq
from Humatch.model import load_cnn, HEAVY_WEIGHTS, LIGHT_WEIGHTS, PAIRED_WEIGHTS

PAD = "----------"
ORDERED_COLS = ["VH", "VL"] + ["hv"] + HEAVY_V_GENE_CLASSES[1:] + ["lv"] + LIGHT_V_GENE_CLASSES[1:] + ["CNN_H", "CNN_L", "CNN_P"]


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


def command_line_interface():
    description="""
    Humatch - Classify
                                                           @
    Author: Lewis Chinery               )  __QQ     vs    /||\\
    Supervisor: Charlotte M. Deane     (__(_)_">           /\\
    Contact: opig@stats.ox.ac.uk                          
    """
    parser = argparse.ArgumentParser(prog="Humatch-classify", description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-H", "--VH", help="Heavy chain amino acid sequence", default=None)
    parser.add_argument("-L", "--VL", help="Light chain amino acid sequence", default=None)
    parser.add_argument("-i", "--input", help="Path to csv with antibody sequences", default=None)
    parser.add_argument("--vh_col", help="Column name for VH sequences in input file", default="VH")
    parser.add_argument("--vl_col", help="Column name for VL sequences in input file", default="VL")
    parser.add_argument("-a", "--aligned", help="Input sequences are prealigned to 200 KASearch positions", default=False, action="store_true")
    parser.add_argument("-s", "--summarise", help="Output top predicted human v-gene only", default=False, action="store_true")
    parser.add_argument("-o", "--output", help="Output save path - defaults to the same dir as input", default=None)
    parser.add_argument("-v", "--verbose", help="Verbose output flag", default=False, action="store_true")
    args = parser.parse_args()

    # show help menu if no options given
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    # check input provided is OK
    if args.VH is None and args.VL is None and args.input is None:
        raise ValueError("Must provide either VH, VL, or input file")
    if args.VH is not None or args.VL is not None:
        if args.input is not None:
            raise ValueError("Cannot provide input file if VH or VL is given")

    # get sequences
    H_seqs, L_seqs = [args.VH] if args.VH else [], [args.VL] if args.VL else []
    vh_col, vl_col = args.vh_col, args.vl_col
    if args.input: df = pd.read_csv(args.input); H_seqs.extend(df[vh_col].tolist() if vh_col in df.columns else []); L_seqs.extend(df[vl_col].tolist() if vl_col in df.columns else [])

    # align
    if not args.aligned:
        if args.verbose:
            num_seq_info = "" if args.input is None else f" ({len(H_seqs)} VH, {len(L_seqs)} VL)"
            print(f"Aligning sequences{num_seq_info}")
        H_seqs = [get_padded_seq(strip_padding_from_seq(seq)) for seq in H_seqs]
        L_seqs = [get_padded_seq(strip_padding_from_seq(seq)) for seq in L_seqs]

    # identify if anarci failed on any sequences
    num_failed_H = len([seq for seq in H_seqs if seq == "-"*len(CANONICAL_NUMBERING)])
    num_failed_L = len([seq for seq in L_seqs if seq == "-"*len(CANONICAL_NUMBERING)])
    if num_failed_H > 0 or num_failed_L > 0:
        print(f"Warning: {num_failed_H} VH and {num_failed_L} VL sequences could not be numbered by ANARCI")

    # predict
    if args.verbose: print("Getting CNN predictions")
    predictions_heavy, predictions_light, predictions_paired = None, None, None
    if len(H_seqs) > 0:
        predictions_heavy = predict_from_list_of_seq_strs(H_seqs, load_cnn(HEAVY_WEIGHTS, "heavy"))
        top_heavy = get_class_and_score_of_max_predictions_only(predictions_heavy, "heavy") if args.summarise else None
    if len(L_seqs) > 0:
        predictions_light = predict_from_list_of_seq_strs(L_seqs, load_cnn(LIGHT_WEIGHTS, "light"))
        top_light = get_class_and_score_of_max_predictions_only(predictions_light, "light") if args.summarise else None
    if len(H_seqs) > 0 and len(L_seqs) > 0:
        paired_seqs = [H_seq + PAD + L_seq for H_seq, L_seq in zip(H_seqs, L_seqs)]
        predictions_paired = predict_from_list_of_seq_strs(paired_seqs, load_cnn(PAIRED_WEIGHTS, "paired"))

    # output
    df_out = pd.DataFrame()
    if len(H_seqs) > 0:
        df_out["VH"] = H_seqs
        if top_heavy is not None:
            df_out["hv"] = [class_str for class_str, _ in top_heavy]
            df_out["CNN_H"] = [score for _, score in top_heavy]
        else:
            df_out[HEAVY_V_GENE_CLASSES[1:]] = predictions_heavy[:, 1:]
    if len(L_seqs) > 0:
        df_out["VL"] = L_seqs
        if top_light is not None:
            df_out["lv"] = [class_str for class_str, _ in top_light]
            df_out["CNN_L"] = [score for _, score in top_light]
        else:
            df_out[LIGHT_V_GENE_CLASSES[1:]] = predictions_light[:, 1:]
    if len(H_seqs) > 0 and len(L_seqs) > 0:
        df_out["CNN_P"] = predictions_paired[:, 1:]

    # rearrange columns so that VH, VL are first if present, then hv, lv, CNN_H, CNN_L, CNN_P
    present_ordered_cols = [col for col in ORDERED_COLS if col in df_out.columns]
    df_out = df_out[present_ordered_cols]

    # save if output or input provided
    out_path = args.output if args.output is not None else args.input.replace(".csv", "_Humatch_classified.csv") if args.input is not None else None
    if out_path is not None:
        if args.verbose: print(f"Saving to {out_path}")
        df_out.to_csv(out_path, index=False)
    # print output for single Fv if out path not provided
    else:
        for col, val in df_out.iloc[0].items():
            if col in ["VH", "VL"]: continue
            val = f"{val:.3f}" if isinstance(val, np.float32) else val
            print(f"{col}: \t{val}")
