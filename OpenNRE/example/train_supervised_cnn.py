# coding:utf-8
import torch
import numpy as np
import json
import sys
import os

sys.path.append('C:\\Users\\danil\\Documents\\Northwestern\\Research\\software\\OpenNRE\\')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import opennre
from opennre import encoder, model, framework
import argparse
import logging
import random

from collections import defaultdict 
import re
import os
import nltk.data
import json
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--encoder', default='pcnn', choices=['pcnn', 'cnn'])

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred'], 
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
        help='Batch size')
parser.add_argument('--lr', default=1e-1, type=float,
        help='Learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
        help='Max number of training epochs')

# Others
parser.add_argument('--seed', default=42, type=int,
        help='Random seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}'.format(args.dataset, 'cnn')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
    if args.dataset == 'wiki80':
        args.metric = 'acc'
    else:
        args.metric = 'micro_f1'
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
id2rel = {v:k for k, v in rel2id.items()}

# Download glove
opennre.download('glove', root_path=root_path)
word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
if args.encoder == 'pcnn':
    sentence_encoder = opennre.encoder.PCNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
elif args.encoder == 'cnn':
    sentence_encoder = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
else:
    raise NotImplementedError

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt='sgd'
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result, pred_result = framework.predict_model(framework.test_loader)


# TODO: this is a hack

def load_dictionaries():

    FACT_LEXES_FILE = 'C:\\Users\\danil\\Documents\\Northwestern\\QRG\\Rep\\ea\\v8\\question-answering\\ke\\preprocessing\\fact-lexes-3.txt'
    lex_to_concepts = defaultdict(lambda: set())
    concept_to_lexes = defaultdict(lambda: set())
    lst_to_fact = defaultdict(lambda: set())
    concept_to_lst = defaultdict(lambda: set())

    with open(FACT_LEXES_FILE, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        is_reading_facts = True
        lex_concept = None
        fact = None
        it = 0
        while it < len(lines):
            line = lines[it]
            if "== CONCEPT DICTIONARY ==" in line:
                is_reading_facts = False
            elif line[0] == " " or len(line) <= 1:
                pass
            elif line[0] == "(":
                fact = line
                count = line.count("(") - line.count(")")
                while count > 0:
                    it += 1
                    line = lines[it]
                    fact += line.strip("\n")
                    count += line.count("(") - line.count(")")
                fact = fact.strip()
            elif line[:2] == "- ":
                lex_concept = line[2:].strip()
            elif is_reading_facts:
                concepts = [x.strip() for x in line.split(",")]
                if len(concepts) > 1:
                    lst_to_fact[tuple(concepts)].add(fact)
                    concept_to_lst[concepts[0]].add(tuple(concepts[1:]))
            else:
                lexes = [x.strip() for x in line.split(",")]
                for lex in lexes:
                    concept_to_lexes[lex_concept].add(lex)
                for lex in lexes:
                    lex_to_concepts[lex].add(lex_concept)
            it += 1

    return lst_to_fact

def get_relation(term):
    '''
    Very simple heuristic to distinguish if a term is a relation
    Return None is term is not relation, or the relation otherwise
    '''
    while len(term) > 0 and term[0] == '(':
        term = term[1:]
    while len(term) > 0 and term[-1] == ')':
        term = term[:-1]
    if term[:1].islower():
        return term
    return None

def from_fact_to_rel(fact):
    relation = '-'.join([get_relation(term) for term in fact.split() 
        if get_relation(term) is not None])
    return relation

def print_preds_for_new_facts():
    lst_to_fact = load_dictionaries()

    print('predictions')
    rel_pred_count = 0
    for i in range(len(pred_result)):
        data = framework.test_loader.dataset.data

        head = data[i]['h']['id']
        tail = data[i]['t']['id']
        existing_facts = list(lst_to_fact[(head, tail)] | lst_to_fact[(tail, head)])
        list_of_existing_relations = [from_fact_to_rel(fact) 
            for fact in existing_facts]
        if data[i]['relation'] in list_of_existing_relations:
            continue

        if id2rel[pred_result[i]] != 'entities_not_related':
            rel_pred_count += 1
            print(data[i]['text'])
            print(data[i]['h'])
            print(data[i]['t'])
            print('GOLD = "' + str(data[i]['relation']) + '"')
            print('PREDICTED = "' + str(id2rel[pred_result[i]]) + '"')
            print('EXISINTG:', str(existing_facts))
            print()
        if rel_pred_count > 150:
            break

print_preds_for_new_facts()

# Print the result
logging.info('Test set results:')
if args.metric == 'acc':
    logging.info('Accuracy: {}'.format(result['acc']))
else:
    logging.info('Micro precision: {}'.format(result['micro_p']))
    logging.info('Micro recall: {}'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))