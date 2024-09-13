import os
import tensorflow as tf
import multiprocessing as mp

# from https://github.com/oxpig/kasearch/blob/main/kasearch/canonical_alignment.py
CANONICAL_NUMBERING = ["1 ", "2 ", 
                       "3 ", "3A", 
                       "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10 ", "11 ", "12 ", "13 ", "14 ", "15 ", "16 ", "17 ", "18 ", 
                       "19 ", "20 ", "21 ", "22 ", "23 ", "24 ", "25 ", "26 ", "27 ", "28 ", "29 ", "30 ", "31 ",  
                       "32 ", "32A", "32B", '33C', "33B", "33A", "33 ", 
                       "34 ", "35 ", "36 ", "37 ", "38 ", "39 ", 
                       "40 ", '40A', 
                       "41 ", "42 ", "43 ", 
                       "44 ", "44A",
                       "45 ", "45A", 
                       "46 ", "46A",
                       "47 ", "47A", 
                       "48 ", "48A", "48B", 
                       "49 ", "49A", 
                       "50 ", 
                       "51 ", "51A", 
                       "52 ", "53 ", "54 ", "55 ", "56 ", "57 ", "58 ", "59 ", 
                       "60 ", "60A", "60B", "60C", '60D', "61E", '61D', "61C", "61B", "61A", "61 ", 
                       "62 ", "63 ", "64 ", "65 ", "66 ", 
                       "67 ", '67A', "67B", 
                       "68 ", "68A", "68B", 
                       "69 ", "69A", "69B",
                       "70 ", 
                       "71 ", "71A", "71B", 
                       "72 ", 
                       "73 ", '73A', "73B",
                       "74 ", "75 ", "76 ", "77 ", "78 ", "79 ", 
                       "80 ", "80A", 
                       "81 ", "81A", "81B", "81C",
                       "82 ", "82A", 
                       "83 ", "83A", "83B",
                       "84 ", 
                       "85 ", "85A", "85B", "85C", "85D", 
                       "86 ", "86A", 
                       "87 ", "88 ", "89 ", "90 ", "91 ", "92 ", "93 ", "94 ", "95 ", 
                       "96 ", "96A",
                       "97 ", "98 ", "99 ", "100 ", "101 ", "102 ", "103 ", "104 ", "105 ", "106 ", "107 ", "108 ", 
                       "109 ", "110 ",
                       "111 ", "111A", "111B", "111C", "111D", "111E", "111F", "111G", '111H', '111I', '111J', 
                       "111K", "111L", "112L",
                       '112K', '112J', '112I', '112H', "112G", "112F", "112E", "112D", "112C", "112B", "112A", "112 ",
                       "113 ","114 ","115 ","116 ","117 ","118 ",
                       "119 ", "119A",
                       "120 ","121 ","122 ","123 ","124 ","125 ", "126 ","127 ","128 "
                        ]


CDR1_start, CDR1_end = "27 ", "38 "
CDR2_start, CDR2_end = "56 ", "65 "
CDR3_start, CDR3_end = "105 ", "117 "


# adapted from https://raw.githubusercontent.com/djhogan/Kidera/master/kidera.csv
KIDERA_DICT = \
{'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
 'R': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
 'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
 'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
 'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
 'Q': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
 'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
 'G': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
 'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
 'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
 'L': [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
 'K': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
 'M': [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],
 'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
 'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
 'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
 'T': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
 'W': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
 'Y': [1.38, 1.48, 0.8, -0.56, -0.0, -0.68, -0.31, 1.03, -0.05, 0.53],
 'V': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],
 '-': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 '*': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


HEAVY_V_GENE_CLASSES = ["neg"] + [f"hv{i}" for i in range(1, 8)]
LIGHT_V_GENE_CLASSES = ["neg"] + [f"lv{i}" for i in range(1, 11)] + [f"kv{i}" for i in range(1, 8)]
PAIRED_CLASSES = ["fake", "true"]


def AA_to_kidera(AA):
    return KIDERA_DICT[AA]


def seq_to_2D_kidera(seq):
    '''
    '''
    kidera_seq = []
    for AA in seq:
        kidera_AA = AA_to_kidera(AA)
        kidera_seq.append(kidera_AA)
    return kidera_seq


def get_ordered_AA_one_letter_codes(extra_chars=["-"]):
    '''
    Get list of amino acid one letter codes in alphabetical order
    '''
    return ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] + extra_chars


def get_indices_of_selected_imgt_positions_in_canonical_numbering(selected_imgt_positions):
    indices = []
    for i in selected_imgt_positions:
        try:
            indices.append(CANONICAL_NUMBERING.index(i))
        except ValueError:
            raise ValueError(f"Warning: position '{i}' not found in canonical numbering.\n\
--> IMGT positions must be given as strings with a space after the number\n\
    if no insertion code is present e.g. '27 ' or '33A'")
    return indices


def get_list_indices_between_two_elements(lst, start, end):
    '''
    Get indices between two elements in a list
    :param lst: list, list of elements
    :param start: str, start element
    :param end: str, end element
    :returns: list, indices between start and end elements (inclusive)
    '''
    start_idx = lst.index(start)
    end_idx = lst.index(end)
    return list(range(start_idx, end_idx+1))


def get_CDR_loop_indices():
    '''
    Get indices of CDR loops in a numbering system
    :param numbering: list, numbering system
    :param paired: bool, if need to get indices for both heavy and light chains
    :param chain_sep_pad_len: int, length of padding between heavy and light chains
    :returns: list, indices of CDR loops
    '''
    cdr1_indices = get_list_indices_between_two_elements(CANONICAL_NUMBERING, CDR1_start, CDR1_end)
    cdr2_indices = get_list_indices_between_two_elements(CANONICAL_NUMBERING, CDR2_start, CDR2_end)
    cdr3_indices = get_list_indices_between_two_elements(CANONICAL_NUMBERING, CDR3_start, CDR3_end)
    return cdr1_indices + cdr2_indices + cdr3_indices


def get_edit_distance(original_seq, mutated_seq):
    '''
    Get number of mutations seq is from original
    Requires seqs to be of same length

    :param original_seq: str of AAs e.g. Trastuzumab's H3
    :param mutated_seq: str of AAs of mutated seq
    :returns: int of number of mutations
    '''
    edit_distance = 0
    for idx, original_AA in enumerate(original_seq):
        if mutated_seq[idx] != original_AA:
            edit_distance += 1  
    return edit_distance


def set_num_cpus(num_cpus=None):
    '''
    Set environment variables to limit the number of CPUs
    Note - speed is not directly proportional to number of CPUs
    :param num_cpus: int, number of CPUs
    '''
    num_cpus = mp.cpu_count() if num_cpus is None else num_cpus
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpus)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cpus)
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
