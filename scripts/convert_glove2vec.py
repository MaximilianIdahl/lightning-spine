from gensim.scripts.glove2word2vec import glove2word2vec
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Converts a embedding txt file in glove format to vec txt format.')
    parser.add_argument('src', type=str, help='input embedding file in glove txt format')
    parser.add_argument('dest', type=str, help='destination to store output file (vec txt format)')
    args = parser.parse_args()
    glove2word2vec(args.src, args.dest)


if __name__ == '__main__':
    main()
