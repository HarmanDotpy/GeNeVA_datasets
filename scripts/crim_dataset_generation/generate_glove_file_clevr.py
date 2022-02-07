# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to generate the GloVe embedding file for the CoDraw and i-CLEVR dataset vocabularies
"""
from tqdm import tqdm
import yaml
import os
import json

with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def generate_glove_file():
    data_path = keys['crim_data_source']
    output_file_vocab = keys['crim_vocab_output']
    output_file_glove = keys['crim_glove_output']
    original_glove = keys['glove_source']
    clevr_vocab = keys['crim_vocab']

    if os.path.isfile(clevr_vocab) != True:
        print('vocab file does not exist..............')
        print('vcreating vocab..............')

        # generate the vocabulary from the question file!
        with open(os.path.join(data_path, 'CLEVR_questions.json'), 'r') as f:
            command_dicts = json.load(f)['questions']
            commands = [command_dict['question'] for  command_dict in command_dicts]
        clevr_vocab = []
        for command in commands:
            words = command.split(' ')
            words = [w.strip().rsplit(' ', 1)[0] for w in words]
            words = [w.split('.')[0] for w in words]
            words = [w.split(';')[0] for w in words]
            clevr_vocab.extend(words)

        
        clevr_vocab = list(set(clevr_vocab))
        clevr_vocab.sort()

        # save vocabulary also
        vocab_output = [word+'\n' for word in clevr_vocab]
        with open(output_file_vocab, 'w') as f:
            f.writelines(vocab_output)
    
    else:
        print('vocab file exists..............')
        with open(clevr_vocab, 'r') as f:
            clevr_vocab = f.readlines()
            clevr_vocab = [x.strip().rsplit(' ', 1)[0] for x in clevr_vocab]
            print(f'vocab length is : {len(clevr_vocab)}')

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
    with open(output_file_glove, 'w') as f:
        for item in clevr_vocab:
            f.write('%s\n' % item)

    print('Total words in vocab: {}\n`unk` embedding words: {}'.format(len(clevr_vocab), unk_count))


if __name__ == '__main__':
    generate_glove_file()
