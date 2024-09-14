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
import yaml

from Humatch.classify import (
    predict_from_list_of_seq_strs,
    get_class_and_score_of_max_predictions_only,
    get_predictions_for_target_class,
)
from Humatch.germline_likeness import (
    mutate_seq_to_match_germline_likeness,
    load_observed_position_AA_freqs,
    GL_DIR
)
from Humatch.utils import (
    get_ordered_AA_one_letter_codes,
    get_CDR_loop_indices,
    get_indices_of_selected_imgt_positions_in_canonical_numbering,
    get_edit_distance,
    CANONICAL_NUMBERING
)
from Humatch.plot import highlight_differnces_between_two_seqs
from Humatch.align import get_padded_seq, strip_padding_from_seq
from Humatch.model import load_cnn, HEAVY_WEIGHTS, LIGHT_WEIGHTS, PAIRED_WEIGHTS

# default config added to compiled env package_data
HUMATCH_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HUMATCH_CODE_DIR, "configs", "default.yaml")


def humanise(heavy_seq, light_seq, cnn_heavy, cnn_light, cnn_paired, config,
             pad="----------", verbose=False):
    '''
    Jointly humanise heavy and light chain sequences to match germline likeness and CNN predictions

    :param heavy/light_seq: str, heavy/light chain sequence to humanise
    :param cnn_heavy/light/paired: model e.g. trained CNN for heavy/light/paired chain
    '''
    precursor_seq_P = heavy_seq + pad + light_seq

    # get target genes if none provided
    try:
        target_gene_H = config["target_gene_H"]
    except KeyError:
        target_gene_H = get_target_gene_if_none_provided(heavy_seq, cnn_heavy, "heavy")
    try:
        target_gene_L = config["target_gene_L"]
    except KeyError:
        target_gene_L = get_target_gene_if_none_provided(light_seq, cnn_light, "light")
    try:
        germline_likeness_lookup_arrays_dir = config["germline_likeness_lookup_arrays_dir"]
    except KeyError:
        germline_likeness_lookup_arrays_dir = GL_DIR

    # make top germline mutations to match germline likeness
    if verbose: print(f"Matching germline likeness for {target_gene_H} and {target_gene_L}")
    best_seq_H = mutate_seq_to_match_germline_likeness(heavy_seq, target_gene_H, config["GL_target_score_H"],
                                                       allow_CDR_mutations=config["GL_allow_CDR_mutations_H"],
                                                       fixed_imgt_positions=config["GL_fixed_imgt_positions_H"],
                                                       germline_likeness_lookup_arrays_dir=germline_likeness_lookup_arrays_dir)
    best_seq_L = mutate_seq_to_match_germline_likeness(light_seq, target_gene_L, config["GL_target_score_L"],
                                                       allow_CDR_mutations=config["GL_allow_CDR_mutations_L"],
                                                       fixed_imgt_positions=config["GL_fixed_imgt_positions_L"],
                                                       germline_likeness_lookup_arrays_dir=germline_likeness_lookup_arrays_dir)
    best_seq_P = best_seq_H + pad + best_seq_L
    edit = get_edit_distance(precursor_seq_P, best_seq_P)

    # get predictions after germline likeness mutations
    max_pred_H = get_predictions_for_target_class([best_seq_H], cnn_heavy, target_gene_H, "heavy", num_cpus=config["num_cpus"])[0]
    max_pred_L = get_predictions_for_target_class([best_seq_L], cnn_light, target_gene_L, "light", num_cpus=config["num_cpus"])[0]
    max_pred_P = get_predictions_for_target_class([best_seq_P], cnn_paired, "true", "paired", num_cpus=config["num_cpus"])[0]

    # while predictions are not above threshold, keep humanising
    all_designed_seqs = [(best_seq_H, best_seq_L)]
    all_cnn_preds = [(max_pred_H, max_pred_L, max_pred_P)]
    all_total_preds = [max_pred_H + max_pred_L + max_pred_P]
    humanisation_failed = False
    i = 0
    if verbose: print(f"Designing and scoring single-point variants")
    while (max_pred_H < config["CNN_target_score_H"]) or (max_pred_L < config["CNN_target_score_L"]) or (max_pred_P < config["CNN_target_score_P"]):
        i += 1
        if verbose: print(f"\tIt. #{i}\tCNN-H: {max_pred_H:.2f},\tCNN-L: {max_pred_L:.2f},\tCNN-P: {max_pred_P:.2f},\tEdit: {edit}")
        
        # get single point variants
        variants_H = get_all_single_point_variants(best_seq_H, config["CNN_allow_CDR_mutations_H"], config["CNN_fixed_imgt_positions_H"])
        variants_L = get_all_single_point_variants(best_seq_L, config["CNN_allow_CDR_mutations_L"], config["CNN_fixed_imgt_positions_L"])
        variants_P = [H + pad + best_seq_L for H in variants_H] + [best_seq_H + pad + L for L in variants_L]

        # get predictions for all variants
        preds_H = get_predictions_for_target_class(variants_H, cnn_heavy, target_gene_H, "heavy", num_cpus=config["num_cpus"])
        preds_L = get_predictions_for_target_class(variants_L, cnn_light, target_gene_L, "light", num_cpus=config["num_cpus"])
        preds_P = get_predictions_for_target_class(variants_P, cnn_paired, "true", "paired", num_cpus=config["num_cpus"])

        # scale/weight predictions
        preds_H_scaled, preds_L_scaled, preds_P_scaled = scale_predictions(best_seq_H, best_seq_L, variants_H, variants_L,
                                                                          preds_H, preds_L, preds_P, max_pred_H, max_pred_L, max_pred_P,
                                                                          germline_likeness_lookup_arrays_dir, target_gene_H, target_gene_L,
                                                                          config["CNN_target_score_H"], config["CNN_target_score_L"],
                                                                          config["CNN_target_score_P"])

        # get total scaled predictions for each variant - each variant affects two predictions (heavy|light and paired)
        preds_H_then_L_scaled = np.concatenate([preds_H_scaled, preds_L_scaled], axis=0)
        preds_total_scaled = preds_H_then_L_scaled + preds_P_scaled

        # get best variant based on total scaled predictions
        best_seq_H, best_seq_L, max_pred_H, max_pred_L, max_pred_P, humanisation_failed = \
            get_best_variant_based_on_total_scaled_predictions(best_seq_H, best_seq_L, variants_H, variants_L,
                                                               preds_H, preds_L, preds_P, max_pred_H, max_pred_L, max_pred_P,
                                                               all_designed_seqs, preds_total_scaled)
        best_seq_P = best_seq_H + pad + best_seq_L
        all_designed_seqs.append((best_seq_H, best_seq_L))
        all_cnn_preds.append((max_pred_H, max_pred_L, max_pred_P))
        all_total_preds.append(max_pred_H + max_pred_L + max_pred_P)

        # break if max edit distance reached/all variants tested
        edit = get_edit_distance(precursor_seq_P, best_seq_P)
        if edit > config["max_edit"]:
            humanisation_failed = True
        if humanisation_failed:
            break

    # return best design even if humanisation fails (we may sometimes reduce total CNN scores in while loop)
    if humanisation_failed:
        best_idx = np.argmax(all_total_preds)
        best_seq_H, best_seq_L = all_designed_seqs[best_idx]
        max_pred_H, max_pred_L, max_pred_P = all_cnn_preds[best_idx]
        best_seq_P = best_seq_H + pad + best_seq_L
        edit = get_edit_distance(precursor_seq_P, best_seq_P)
        if verbose: print(f"Humanisation failed")

    if verbose: print(f"Humanised sequences:\n\t{best_seq_H.replace('-','')}\n\t{best_seq_L.replace('-','')}")

    return {"Humatch_H": best_seq_H, "Humatch_L": best_seq_L, "Edit": edit,
            "HV": target_gene_H, "LV": target_gene_L,
            "CNN_H": max_pred_H, "CNN_L": max_pred_L, "CNN_P": max_pred_P}


def scale_predictions(best_seq_H, best_seq_L, variants_H, variants_L,
                      preds_H, preds_L, preds_P, max_pred_H, max_pred_L, max_pred_P,
                      germline_likeness_lookup_arrays_dir, target_gene_H, target_gene_L,
                      CNN_target_score_H, CNN_target_score_L, CNN_target_score_P):
    '''
    Scale predictions of CNNs to upweight common germline mutations and CNN furtherest from target
    
    :param best_seq_H/L: str, best sequence for heavy/light chain
    :param variants_H/L: list of str, variants for heavy/light chain
    :param preds_H/L/P: ndarray of predictions for heavy/light/paired chains
    :param max_pred_H/L/P: float, max prediction for heavy/light/paired chains (target class only)
    :param germline_likeness_lookup_arrays_dir: str, path to directory containing the previously
        calculated position AA frequencies
    :param target_gene_H/L: str, target gene for heavy/light chain
    :param CNN_target_score_H/L/P: float, target score for heavy/light/paired chains
    :returns: three ndarrays of scaled predictions for heavy/light/paired chains
    '''

    # get net change in predictions
    preds_H_net = preds_H.copy() - max_pred_H
    preds_L_net = preds_L.copy() - max_pred_L
    preds_P_net = preds_P.copy() - max_pred_P

    # adjust predictions to upweight variants that are most similar to germline
    GL_arr_H = load_observed_position_AA_freqs(target_gene_H, germline_likeness_lookup_arrays_dir)
    GL_arr_L = load_observed_position_AA_freqs(target_gene_L, germline_likeness_lookup_arrays_dir)
    scaling_factors_H = get_observed_frequency_scaling_factors_for_variants(best_seq_H, variants_H, GL_arr_H)
    scaling_factors_L = get_observed_frequency_scaling_factors_for_variants(best_seq_L, variants_L, GL_arr_L)
    preds_H_net_GL_scaled = scale_predictions_by_observed_frequency(preds_H_net, scaling_factors_H)
    preds_L_net_GL_scaled = scale_predictions_by_observed_frequency(preds_L_net, scaling_factors_L)

    # paired preds contain both heavy and light mutations (length of preds_H + preds_L) so need to scale both parts
    scaling_factors_P = scaling_factors_H + scaling_factors_L
    preds_P_net_GL_scaled = scale_predictions_by_observed_frequency(preds_P_net, scaling_factors_P)

    # make all predictions positive by adding the minimum prediction (equalises importance of all CNNs before scaling again)
    min_overall = min(min(preds_H_net_GL_scaled), min(preds_L_net_GL_scaled), min(preds_P_net_GL_scaled))
    preds_H_net_GL_scaled_pos = preds_H_net_GL_scaled - min_overall
    preds_L_net_GL_scaled_pos = preds_L_net_GL_scaled - min_overall
    preds_P_net_GL_scaled_pos = preds_P_net_GL_scaled - min_overall

    # weight predictions to favour improvements to CNN score furtherest from target
    H_score_distance_from_target = max(0, CNN_target_score_H - max_pred_H)
    L_score_distance_from_target = max(0, CNN_target_score_L - max_pred_L)
    P_score_distance_from_target = max(0, CNN_target_score_P - max_pred_P)
    preds_H_net_GL_scaled_pos_weighted = preds_H_net_GL_scaled_pos * H_score_distance_from_target
    preds_L_net_GL_scaled_pos_weighted = preds_L_net_GL_scaled_pos * L_score_distance_from_target
    preds_P_net_GL_scaled_pos_weighted = preds_P_net_GL_scaled_pos * P_score_distance_from_target

    return preds_H_net_GL_scaled_pos_weighted, preds_L_net_GL_scaled_pos_weighted, preds_P_net_GL_scaled_pos_weighted


def get_best_variant_based_on_total_scaled_predictions(best_seq_H, best_seq_L, variants_H, variants_L,
                                                       preds_H, preds_L, preds_P,
                                                       max_pred_H, max_pred_L, max_pred_P,
                                                       all_designed_seqs, preds_total_scaled):
    '''
    Get the best variant based on total scaled predictions

    :param best_seq_H/L: str, best sequence for heavy/light chain
    :param variants_H/L: list of str, variants for heavy/light chain
    :param preds_H/L/P: ndarray of predictions for heavy/light/paired chains
    :param max_pred_H/L/P: float, max prediction for heavy/light/paired chains (target class only)
    :param all_designed_seqs: list of tuples of str, all designed sequences
    :param preds_total_scaled: ndarray of scaled predictions for all variants
    :returns: str, str, float, float, float, bool, best sequence for heavy chain, best sequence for
        light chain, max prediction for heavy/light/paired chains (target class only), humanisation failed
    '''
    # avoid local minima by selecting best variant that has not been selected before
    novel_best_seq_found = False
    humanisation_failed = False
    while not novel_best_seq_found:

        # get variant that maximises total scaled predictions
        max_pred_idx = np.argmax(preds_total_scaled)
        new_max_pred_P = preds_P[max_pred_idx]
        if max_pred_idx < len(variants_H):
            new_best_seq_H, new_best_seq_L = variants_H[max_pred_idx], best_seq_L
            new_max_pred_H, new_max_pred_L = preds_H[max_pred_idx], max_pred_L
        else:
            new_best_seq_H, new_best_seq_L = best_seq_H, variants_L[max_pred_idx - len(variants_H)]
            new_max_pred_H, new_max_pred_L = max_pred_H, preds_L[max_pred_idx - len(variants_H)]

        # check if new best seq is novel. Set scaled prediction to -inf if not to avoid picking again
        if (new_best_seq_H, new_best_seq_L) not in all_designed_seqs:
            novel_best_seq_found = True
        else:
            preds_total_scaled[max_pred_idx] = -np.inf

        # break if all variants have been selected before
        if all(pred == -np.inf for pred in preds_total_scaled):
            new_best_seq_H, new_best_seq_L = best_seq_H, best_seq_L
            new_max_pred_H, new_max_pred_L, new_max_pred_P = max_pred_H, max_pred_L, max_pred_P
            humanisation_failed = True
            break

    return new_best_seq_H, new_best_seq_L, new_max_pred_H, new_max_pred_L, new_max_pred_P, humanisation_failed


def get_target_gene_if_none_provided(seq, model, chain_type):
    '''
    Use the highest scoring human gene as the target gene if none provided
    :param seq: str, sequence
    :param model: model e.g. trained CNN
    :param chain_type: str, heavy | light | paired
    :returns: str, target gene
    '''
    preds = predict_from_list_of_seq_strs([seq], model)
    target_gene, _ = get_class_and_score_of_max_predictions_only(preds, chain_type)[0]
    return target_gene


def get_all_nonpadded_indices(padded_seq_str, padding_char="-"):
    '''
    Get all non-padded indices in a sequence
    :param padded_seq_str: str, sequence (padded with "-" for missing positions)
    :param padding_char: str, padding character
    :returns: list of int indices
    '''
    return [i for i in range(len(padded_seq_str)) if padded_seq_str[i] != padding_char]


def point_mutate_seq(padded_seq_str, idx, new_AA):
    '''
    Mutate sequence at idx to new_AA
    :param padded_seq_str: str, sequence (padded with "-" for missing positions)
    :param idx: int, index to mutate
    :param new_AA: str, new AA to place at idx
    :returns: str, mutated sequence
    '''
    return padded_seq_str[:idx] + new_AA + padded_seq_str[idx+1:]


def get_all_single_point_variants(padded_seq_str, allow_CDR_mutations=False, fixed_imgt_positions=[]):
    '''
    Get list of all possible single point variants of a sequence
    :param padded_seq_str: str, sequence (padded with "-" for missing positions)
    :param allow_CDR_mutations: bool, if CDR mutations are allowed
    :param fixed_imgt_positions: list, list of IMGT positions to exclude from mutation
    :returns: list of single point mutant str sequences
    '''
    # get indices to mutate
    indices_to_mutate = get_all_nonpadded_indices(padded_seq_str)
    fixed_indices = []
    if not allow_CDR_mutations:
        fixed_indices += get_CDR_loop_indices()
    if fixed_imgt_positions:
        fixed_indices += get_indices_of_selected_imgt_positions_in_canonical_numbering(fixed_imgt_positions)
    indices_to_mutate = [i for i in indices_to_mutate if i not in fixed_indices]

    # get all single point variants
    all_single_point_variants = []
    all_AAs = get_ordered_AA_one_letter_codes()[:20]
    for idx in indices_to_mutate:
        new_AAs = all_AAs.copy()
        new_AAs.remove(padded_seq_str[idx])
        for AA in new_AAs:
            all_single_point_variants.append(point_mutate_seq(padded_seq_str, idx, AA))
            
    return all_single_point_variants


def get_position_idx_and_AA_idx_diff(seq1, seq2, break_on_first_diff=True):
    '''
    Get the position and AA indices that differ between two sequences
    :param seq1: str, sequence 1
    :param seq2: str, sequence 2
    :param break_on_first_diff: bool, break on first difference
    :returns: list of tuples of (position index, AA index)
    '''
    pos_idx_diff = []
    for pos_idx, (AA1, AA2) in enumerate(zip(seq1, seq2)):
        if AA1 != AA2:
            AA_idx = get_ordered_AA_one_letter_codes().index(AA2)
            pos_idx_diff.append((pos_idx, AA_idx))
            if break_on_first_diff:
                break
    return pos_idx_diff


def get_observed_frequency_of_AA_at_position(germline_likeness_lookup_arr, pos_idx, AA_idx):
    '''
    Get the observed frequency of an AA at a position
    :param germline_likeness_lookup_arr: np.array, observed position AA frequencies, shape (200, 20)
    :param pos_idx: int, position index
    :param AA_idx: int, AA index
    :returns: float, observed frequency
    '''
    return germline_likeness_lookup_arr[pos_idx, AA_idx]


def get_observed_frequency_scaling_factors_for_variants(seq, variants, germline_likeness_lookup_arr):
    '''
    Get the observed frequency scaling factors for each variant
    :param seq: str, sequence
    :param variants: list of str variants
    :param germline_likeness_lookup_arr: np.array, observed position AA frequencies, shape (200, 20)
    :returns: list of float scaling factors
    '''
    scaling_factors = []
    for variant in variants:
        pos_idx, AA_idx = get_position_idx_and_AA_idx_diff(seq, variant)[0]
        scaling_factors.append(get_observed_frequency_of_AA_at_position(germline_likeness_lookup_arr, pos_idx, AA_idx))
    return scaling_factors


def scale_predictions_by_observed_frequency(predictions, scaling_factors, noise_factor=0.01):
    '''
    Upweight predictions that are more like germline
    :param predictions: ndarray of net predictions
    :param scaling_factors: list of float scaling factors
    :param noise_factor: float, noise factor
    :returns: ndarray of scaled net predictions
    '''
    # get indices of predictions where predictions are negative
    neg_pred_idxs = [i for i in range(len(predictions)) if predictions[i] < 0]
    # at neg indices, scaling factor --> 1 - scaling factor i.e. upweight (make less negative) the preds that more like germline
    for idx in neg_pred_idxs:
        scaling_factors[idx] = 1 - scaling_factors[idx]
    # add noise - higher noise pushes pos and neg net predictions further from 0
    scaling_factors = np.array(scaling_factors) + noise_factor
    return predictions * scaling_factors


def command_line_interface():
    description="""
    Humatch - Humanise
                                                           @
    Author: Lewis Chinery               )  __QQ    -->    /||\\
    Supervisor: Charlotte M. Deane     (__(_)_">           /\\
    Contact: opig@stats.ox.ac.uk                          
    """
    parser = argparse.ArgumentParser(prog="Humatch-humanise", description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-H", "--VH", help="Heavy chain amino acid sequence", default=None)
    parser.add_argument("-L", "--VL", help="Light chain amino acid sequence", default=None)
    parser.add_argument("-i", "--input", help="Path to csv with antibody sequences", default=None)
    parser.add_argument("--vh_col", help="Column name for VH sequences in input file", default="VH")
    parser.add_argument("--vl_col", help="Column name for VL sequences in input file", default="VL")
    parser.add_argument("--config", help="Path to config file", default=None)
    parser.add_argument("-a", "--aligned", help="Input sequences are prealigned to 200 KASearch positions", default=False, action="store_true")
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
    if (args.VH is None and args.VL is not None) or (args.VH is not None and args.VL is None):
        raise ValueError("Humatch humanisation requires both VH and VL sequences")
    if args.input is not None and (args.vh_col not in pd.read_csv(args.input).columns or args.vl_col not in pd.read_csv(args.input).columns):
        raise ValueError(f"Humatch humanisation requires both VH and VL sequences. Could not find columns '{args.vh_col}' and '{args.vl_col}' in input file. Column names can be changed with --vh_col and --vl_col")
        
    # load default config if not given
    config = CONFIG if args.config is None else args.config
    with open(config) as f:
        config = yaml.safe_load(f)
    if args.verbose:
        print(f"\nConfig:")
        print(pd.DataFrame.from_dict(config, orient="index", columns=["Value"]))

    # get sequences
    H_seqs, L_seqs = [args.VH] if args.VH else [], [args.VL] if args.VL else []
    vh_col, vl_col = args.vh_col, args.vl_col
    if args.input: df = pd.read_csv(args.input); H_seqs.extend(df[vh_col].tolist() if vh_col in df.columns else []); L_seqs.extend(df[vl_col].tolist() if vl_col in df.columns else [])

    # align
    if not args.aligned:
        if args.verbose:
            num_seq_info = "" if args.input is None else f" ({len(H_seqs)} VH, {len(L_seqs)} VL)"
            print(f"\nAligning sequences{num_seq_info}")
        H_seqs = [get_padded_seq(strip_padding_from_seq(seq)) for seq in H_seqs]
        L_seqs = [get_padded_seq(strip_padding_from_seq(seq)) for seq in L_seqs]
    
    # check if ANARCI failed on any sequences and remove them
    failed_H_idxs = [i for i, seq in enumerate(H_seqs) if seq == "-"*len(CANONICAL_NUMBERING)]
    failed_L_idxs = [i for i, seq in enumerate(L_seqs) if seq == "-"*len(CANONICAL_NUMBERING)]
    all_failed_idxs = list(set(failed_H_idxs + failed_L_idxs))
    H_seqs = [seq for i, seq in enumerate(H_seqs) if i not in all_failed_idxs]
    L_seqs = [seq for i, seq in enumerate(L_seqs) if i not in all_failed_idxs]
    if len(failed_H_idxs) > 0 or len(failed_L_idxs) > 0:
        print(f"Warning: {len(failed_H_idxs)} VH and {len(failed_L_idxs)} VL sequences could not be numbered by ANARCI")

    # load CNNs
    if args.verbose: print("Loading CNNs")
    cnn_heavy = load_cnn(HEAVY_WEIGHTS, "heavy")
    cnn_light = load_cnn(LIGHT_WEIGHTS, "light")
    cnn_paired = load_cnn(PAIRED_WEIGHTS, "paired")

    # humanising sequences
    results = []
    if args.verbose: print(f"Humanising {len(H_seqs)} sequences")
    for i, (H_seq, L_seq) in enumerate(zip(H_seqs, L_seqs)):
        if args.verbose: print(f"\nHumanising sequence {i+1}/{len(H_seqs)}")
        results.append(humanise(H_seq, L_seq, cnn_heavy, cnn_light, cnn_paired, config, verbose=args.verbose))
        if args.verbose:
            for key, val in results[-1].items():
                if key in ["Humatch_H", "Humatch_L"]: continue
                val = f"{val:.3f}" if isinstance(val, np.float32) else val
                print(f"\t{key}:\t{val}")

    # save if output or input provided
    out_path = args.output if args.output is not None else args.input.replace(".csv", "_Humatch_humanised.csv") if args.input is not None else None
    if out_path is not None:
        if args.verbose: print(f"Saving to {out_path}")
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_path, index=False)
    # print output for single Fv if out path not provided (if it has not been printed earlier)
    else:
        if not args.verbose and len(results) > 0:
            print(f"Humanised sequences:\n\t{results[0]['Humatch_H'].replace('-','')}\n\t{results[0]['Humatch_L'].replace('-','')}")
            for key, val in results[0].items():
                if key in ["Humatch_H", "Humatch_L"]: continue
                val = f"{val:.3f}" if isinstance(val, np.float32) else val
                print(f"\t{key}:\t{val}")
