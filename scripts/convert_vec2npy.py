from gensim.models import KeyedVectors
import numpy as np
from typing import Dict
import argparse


def convert(fname: str, binary: bool) -> (Dict[str, int], np.array):
    embeddings = KeyedVectors.load_word2vec_format(fname, binary=binary)
    word2idx = {k: v.index for k, v in embeddings.vocab.items()}
    assert list(word2idx.values()) == list(range(0, embeddings.vectors.shape[0]))
    return word2idx, embeddings.vectors


def save(fname: str, word2idx: Dict[str, int], vectors: np.array):
    np.save(fname, [word2idx, vectors])


def main():
    parser = argparse.ArgumentParser(
        description='Converts from the word2vec .vec format (as used in gensim) to npy format.'
                    '\nThe npy format can be loaded via\n'
                    '"word2idx, vectors = np.load(embeddings.npy, allow_pickle=True)"')
    parser.add_argument('src', type=str, help='input embedding file')
    parser.add_argument('dest', type=str, help='destination to store output file')
    parser.add_argument('--binary', action='store_true', help='input is in binary format')
    args = parser.parse_args()
    word2idx, vectors = convert(args.src, args.binary)
    save(args.dest, word2idx, vectors)


if __name__ == '__main__':
    main()
