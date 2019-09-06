from gensim.models import KeyedVectors
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Converts embeddings in txt vec format to binary vec format.')
    parser.add_argument('src', type=str, help='input embedding file in vec txt format')
    parser.add_argument('dest', type=str, help='destination to store output file (binary vec format)')
    args = parser.parse_args()
    kv = KeyedVectors.load_word2vec_format(args.src, binary=False)
    kv.save_word2vec_format(args.dest, binary=True)


if __name__ == '__main__':
    main()
