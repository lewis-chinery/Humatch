import os
import sys
# supress warnings about having no GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
# suppress warnings about tf retracing
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import pandas as pd
import numpy as np
import anarci
import argparse
from Humatch.utils import CANONICAL_NUMBERING, get_ordered_AA_one_letter_codes


def strip_padding_from_seq(seq, pad_token="-"):
    '''
    Strip padding from a sequence

    :param seq: str, sequence of amino acids
    :param pad_token: str, character used for padding
    :return: str, sequence with padding removed
    '''
    # remove non-canonical AAs as anarci does not like them
    canonical_AAs = get_ordered_AA_one_letter_codes()[:20]
    seq = "".join([AA for AA in seq if AA in canonical_AAs])
    return seq.replace(pad_token, "")


def get_padded_seq(seq, allowed_imgt_nums=CANONICAL_NUMBERING, pad_token="-"):
    '''
    Return a sequence padded with "-" to match desired alignment
    We return the full padding if the sequence cannot be numbered by ANARCI
    This maintains correct VH/VL pairing for humanisation when one sequence cannot be numbered

    :param seq: str, sequence of amino acids (unpadded)
    :param allowed_imgt_nums: list of str, canonical numbering format
    '''
    # number the sequence (output is list of tuples) e.g.
    #    [((1, ' '), 'E'), ((2, ' '), 'V'), ...]
    num_AA_tup = anarci.number(seq)[0]
    if num_AA_tup is False:
        print(f"Error: {seq} could not be numbered by ANARCI. Returning full padding e.g. '---...---'")
        aligned_seq = pad_token * len(allowed_imgt_nums)
    else:
        # convert to dict with keys that match canonical numbering format e.g.
        #    {'1 ': 'E', '2 ': 'V', ...}
        d = {str(num_AA[0][0]) + num_AA[0][1]: num_AA[1] for num_AA in num_AA_tup}
        # get the aligned sequence
        aligned_seq = ""
        for imgt_num in allowed_imgt_nums:
            aligned_seq += d.get(imgt_num, pad_token)
    return aligned_seq


def command_line_interface():
    description="""
    Humatch - Align

    Author: Lewis Chinery              QVQLVQSGAEV... --> QVQ-LVQSGA-EV...
    Supervisor: Charlotte M. Deane
    Contact: opig@stats.ox.ac.uk                          
    """
    parser = argparse.ArgumentParser(prog="Humatch-align", description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-H", "--VH", help="Heavy chain amino acid sequence", default=None)
    parser.add_argument("-L", "--VL", help="Light chain amino acid sequence", default=None)
    parser.add_argument("-i", "--input", help="Path to csv with antibody sequences", default=None)
    parser.add_argument("--vh_col", help="Column name for VH sequences in input file", default="VH")
    parser.add_argument("--vl_col", help="Column name for VL sequences in input file", default="VL")
    parser.add_argument("--imgt_cols", help="Flag to use IMGT numbering columns (aa-level) instead of heavy/light cols", default=False, action="store_true")
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
    if args.verbose: print("Reading sequences")
    H_seqs, L_seqs = [args.VH] if args.VH else [], [args.VL] if args.VL else []
    vh_col, vl_col = args.vh_col, args.vl_col
    if args.input: df = pd.read_csv(args.input); H_seqs.extend(df[vh_col].tolist() if vh_col in df.columns else []); L_seqs.extend(df[vl_col].tolist() if vl_col in df.columns else [])

    # align
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

    # save if output or input provided
    out_path = args.output if args.output is not None else args.input.replace(".csv", "_Humatch_aligned.csv") if args.input is not None else None
    if out_path is not None:
        if args.verbose: print(f"Saving to {out_path}")
        df_H, df_L = None, None
        if len(H_seqs) > 0:
            if args.imgt_cols:
                df_H = pd.DataFrame([[AA for AA in seq] for seq in H_seqs], columns=[f"H_{imgt}" for imgt in CANONICAL_NUMBERING])
            else:
                df_H = pd.DataFrame({"VH": H_seqs})
        if len(L_seqs) > 0:
            if args.imgt_cols:
                df_L = pd.DataFrame([[AA for AA in seq] for seq in L_seqs], columns=[f"L_{imgt}" for imgt in CANONICAL_NUMBERING])
            else:
                df_L = pd.DataFrame({"VL": L_seqs})
        df_out = pd.concat([df_H, df_L], axis=1) if df_H is not None and df_L is not None else df_H if df_H is not None else df_L
        df_out.to_csv(out_path, index=False)
    # print output for single Fv if out path not provided
    else:
        H = H_seqs[0] if len(H_seqs) > 0 else " "*len(CANONICAL_NUMBERING)
        L = L_seqs[0] if len(L_seqs) > 0 else " "*len(CANONICAL_NUMBERING)
        for imgt, h_AA, l_AA in zip(CANONICAL_NUMBERING, H, L):
            print(f"{imgt}\t{h_AA}\t{l_AA}")
