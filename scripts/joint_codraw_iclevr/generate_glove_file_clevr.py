# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to generate the GloVe embedding file for the CoDraw and i-CLEVR dataset vocabularies
"""
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def generate_glove_file():
    clevr_vocab = keys['iclevr_vocab']
    output_file = keys['glove_output']
    original_glove = keys['glove_source']
    import pdb; pdb.set_trace()
    # read i-CLEVR vocabulary
    with open(clevr_vocab, 'r', encoding="utf-8") as f:
        clevr_vocab = f.readlines()
        clevr_vocab = [x.strip().rsplit(' ', 1)[0] for x in clevr_vocab]

    clevr_vocab = list(set(clevr_vocab))
    clevr_vocab.sort()

    print('Loading GloVe file. This might take a few minutes.')
    with open(original_glove, 'r', encoding="utf-8") as f:
        original_glove = f.readlines()
        tok_glove_pairs = [x.strip().split(' ', 1) for x in original_glove]

    # extract GloVe vectors for vocabulary tokens
    for token, glove_emb in tqdm(tok_glove_pairs):
        if token == 'unk':
            unk_embedding = glove_emb
        try:
            token_idx = clevr_vocab.index(token)
        except ValueError:
            continue
        else:
            clevr_vocab[token_idx] = ' '.join([token, glove_emb])

    # assign 'unk' GloVe embedding to unknown words
    unk_count = 0
    for itidx, item in enumerate(clevr_vocab):
        if len(item.split(' ')) == 1:
            unk_count += 1
            clevr_vocab[itidx] = ' '.join([item, unk_embedding])

    # write GloVe vector file for the CoDraw and i-CLEVR datasets combined
    with open(output_file, 'w') as f:
        for item in clevr_vocab:
            f.write('%s\n' % item)

    print('Total words in vocab: {}\n`unk` embedding words: {}'.format(len(clevr_vocab), unk_count))


if __name__ == '__main__':
    generate_glove_file()
