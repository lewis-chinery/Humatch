import anarci
from Humatch.utils import CANONICAL_NUMBERING


def strip_padding_from_seq(seq, pad_token="-"):
    '''
    Strip padding from a sequence

    :param seq: str, sequence of amino acids
    :param pad_token: str, character used for padding
    :return: str, sequence with padding removed
    '''
    return seq.replace(pad_token, "")


def get_padded_seq(seq, allowed_imgt_nums=CANONICAL_NUMBERING, pad_token="-"):
    '''
    Return a sequence padded with "-" to match desired alignment

    :param seq: str, sequence of amino acids (unpadded)
    :param allowed_imgt_nums: list of str, canonical numbering format
    '''
    # number the sequence (output is list of tuples) e.g.
    #    [((1, ' '), 'E'), ((2, ' '), 'V'), ...]
    num_AA_tup = anarci.number(seq)[0]
    # convert to dict with keys that match canonical numbering format e.g.
    #    {'1 ': 'E', '2 ': 'V', ...}
    d = {str(num_AA[0][0]) + num_AA[0][1]: num_AA[1] for num_AA in num_AA_tup}
    # get the aligned sequence
    aligned_seq = ""
    for imgt_num in allowed_imgt_nums:
        aligned_seq += d.get(imgt_num, pad_token)
    return aligned_seq
