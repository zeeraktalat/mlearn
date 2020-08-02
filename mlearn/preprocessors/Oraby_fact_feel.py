# Preprocess Oraby Fact-Feel into train, dev, and test tsv files.
import csv
import os


def process_raw(read_path: str, write_path: str):
    """
    Read Oraby et al. (2015) in its raw format.
    (each document is a single file. Direcotory structure are splits and labels).

    :read_path: The name of the top level directory the files are in.
    :write_path: The directory to write the files to.
    """

    for split_dir in os.listdir(read_path):
        outf = open(os.path.join(write_path, f"Oraby_fact_feel_{split_dir}.tsv"), 'w', encoding = 'utf-8')
        writer = csv.writer(outf, delimiter = '\t')
        writer.writerow(["DocID", "Label", "Text"])

        for label_dir in os.listdir(os.path.join(read_path, split_dir)):
            label = label_dir

            for f in os.listdir(os.path.join(read_path, split_dir, label_dir)):
                idx = f.strip('.txt')
                fh = open(os.path.join(read_path, split_dir, label_dir, f), 'r', encoding = 'latin-1')
                contents = " ".join(fh.readlines()).replace('\n', ' ').replace('\r', ' ').rstrip()
                writer.writerow([idx, label, contents])
