import csv
import os

labels = open('../../../data/hate-speech-dataset/annotations_metadata.csv', 'r', encoding = 'utf-8')
train = open('../../../data/garcia_stormfront_train.tsv', 'w', encoding = 'utf-8')
test = open('../../../data/garcia_stormfront_test.tsv', 'w', encoding = 'utf-8')
omitted = open('../../../data/garcia_omitted_docs.log', 'w', encoding = 'utf-8')
train_dir = os.listdir('../../../data/hate-speech-dataset/sampled_train')
test_dir = os.listdir('../../../data/hate-speech-dataset/sampled_test')
print(len(train_dir), len(test_dir))

reader = csv.reader(labels, delimiter = ',')
train_writer = csv.writer(train, delimiter = '\t')
test_writer = csv.writer(test, delimiter = '\t')

header = ["file_id", "user_id", "subforum_id", "num_contexts", "label", "text"]
train_writer.writerow(header)
test_writer.writerow(header)

for line in reader:
    f_idx, u_idx, forum_idx, contexts, label = line
    label = label.rstrip()
    f_idx = f_idx + '.txt'

    if f_idx in train_dir:
        fin = open('../../../data/hate-speech-dataset/sampled_train/' + f_idx)
        data = fin.readlines()
        fin.close()
        data = " ".join(data)
        data = data.replace('\n', '').replace('\t', '')
        out = [f_idx, u_idx, forum_idx, contexts, label, data]
        train_writer.writerow(out)

    elif f_idx in test_dir:
        fin = open('../../../data/hate-speech-dataset/sampled_test/' + f_idx)
        data = fin.readlines()
        fin.close()
        data = " ".join(data)
        data = data.replace('\n', '').replace('\t', '')
        out = [f_idx, u_idx, forum_idx, contexts, label, data]
        test_writer.writerow(out)
    else:
        omitted.write(f_idx.strip('.txt') + '\n')
