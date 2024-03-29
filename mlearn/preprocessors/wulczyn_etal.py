import argparse
import csv
import os
from numpy import mean
from collections import defaultdict

parser = argparse.ArgumentParser('Raw to test/dev/train for Wulczyn et al')
parser.add_argument('--datapath', default='/Users/zeerakw/PhD/projects/active/Generalisable_abuse/data/')
args = parser.parse_args()

base_path = args.datapath
if not (os.path.exists(base_path) and os.path.isdir(base_path)):
    raise Exception(f"--datapath should exist and be a directory; given {base_path}")

train_f = csv.writer(open(os.path.join(base_path, 'wulczyn_train.tsv'), 'w', encoding = 'utf-8'), delimiter = '\t')
dev_f = csv.writer(open(os.path.join(base_path, 'wulczyn_dev.tsv'), 'w', encoding = 'utf-8'), delimiter = '\t')
test_f = csv.writer(open(os.path.join(base_path, 'wulczyn_test.tsv'), 'w', encoding = 'utf-8'), delimiter = '\t')

# Write headers
header = ['rev_id', 'comment', 'label', 'raw_label']
train_f.writerow(header)
dev_f.writerow(header)
test_f.writerow(header)

# Load the labels
annotation_fp = open(os.path.join(base_path, 'toxicity_annotations.tsv'))
next(annotation_fp)

raw_annotations, annotations = defaultdict(list), {}

# Clean the labels
for line in csv.reader(annotation_fp, delimiter = '\t'):
    raw_annotations[line[0]].append(float(line[2]))

# Binarize labels
for idx, vals in raw_annotations.items():
    annotations[idx] = 'abuse' if mean(vals) > 0.50 else 'not-abuse'

# Write documents to files
raw_docs = open(os.path.join(base_path, 'toxicity_annotated_comments.tsv'), 'r', encoding = 'utf-8')
next(raw_docs)  # Skip header

for line in csv.reader(raw_docs, delimiter = '\t'):
    doc_id, text, split = line[0], line[1], line[6].strip()
    text = text.replace('NEWLINE_TOKEN', ' ')  # Cleanup documents

    label = annotations[doc_id]
    raw_label = mean(raw_annotations[doc_id])

    out = [doc_id, text, label, raw_label]
    if split == 'train':
        writer = train_f.writerow

    elif split == 'dev':
        writer = dev_f.writerow

    elif split == 'test':
        writer = test_f.writerow

    writer(out)
