import sys
from fairseq.data.encoders.gpt2_bpe import get_encoder

def decode(bpe, tokens):
    return bpe.decode(tokens)

if __name__ == '__main__':
    in_filename  = sys.argv[1]
    out_filename = sys.argv[2]

    bpe = get_encoder("data/bpe/bart/encoder.json", "data/bpe/bart/vocab.bpe")

    with open(out_filename, 'w') as f:
        f.close()

    with open(in_filename, "r") as f:
        for line in f:
            tokens = list(map(int, line.rstrip().split()))
            decoded_toks = decode(bpe, tokens)
            with open(out_filename, 'a') as f:
                print(decoded_toks, file=f)
