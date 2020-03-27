from collections import Counter


def majority_voted(annotations):
    cleaned = []
    candidates = []
    for a in annotations:
        candidates.extend([cand for cand in a.split(',') if cand != ''])
    cleaned.extend(candidates)
    c = Counter(cleaned)
    return c.most_common(1)[0][0]


def clean_mftc():
    fin = open('~/Desktop/Datasets/MFTC_V4_text.json', 'r', encoding = 'utf-8')
    fout = open('MFTC_V4_text_parsed.tsv', 'w', encoding = 'utf-8')
    tweet_data = []
    annotations = []
    annotators = []

    fout.write("\t".join(['tweet_id', 'tweet_text', 'annotator_1', 'annotator_2', 'annotator_3',
                          'annotator_4', 'annotator_5', 'annotator_6', 'annotator_7', 'annotator_8',
                          'annotation_1', 'annotation_2', 'annotation_3', 'annotation_4', 'annotation_5',
                          'annotation_6', 'annotation_7', 'annotation_8', 'majority_label', 'corpus']) + '\n')

    for line in fin:
        if ':' in line:
            line = line.replace('\n', '').replace('"', '').replace('\\n', '')
            if 'Corpus' in line:
                corpus = line.split(':')[1].strip().strip(',')
                continue

            if 'tweet_id' in line:
                value = line.split(':')[1].strip().strip(',')
                if tweet_data == [] and annotations == []:
                    tweet_data.append(value)

                elif tweet_data != [] and annotations != []:  # A new data object has been encountered.

                    if len(annotations) < 8:  # Ensure that there are the same number of annotators for each line
                        while len(annotations) <= 7:
                            annotations.append('')
                            annotators.append('')

                    # Identify Majority voted label
                    majority = majority_voted(annotations)
                    tweet_data.extend(annotators)
                    tweet_data.extend(annotations)  # Add annotations to output line
                    tweet_data.append(majority)
                    tweet_data.append(corpus)  # Add corpus information
                    fout.write("\t".join(tweet_data) + '\n')  # Write the data line

                    # Prepare for a new tweet
                    tweet_data = [value]
                    annotations = []
                    continue

            if 'tweet_text' in line:
                value = ":".join(line.split(':')[1:]).strip().strip('\n')[:-1]
                tweet_data.append(value)
                continue

            if 'annotations' in line:
                annotations = []
                annotators = []
                while ']' not in line:
                    line = next(fin)
                    if 'annotator' in line:
                        annotators.append(line.strip().split(':')[1].strip().replace('"', '').replace(',', ''))
                    if 'annotation' in line:
                        annotations.append(line.strip().split(':')[1].strip().replace('\n', '').replace('"', ''))
                continue


if __name__ == "__main__":
    clean_mftc()
