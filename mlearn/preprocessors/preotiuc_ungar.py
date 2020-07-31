import os
import csv
import json
from tqdm import tqdm


def preprocess(read_path, write_path):
    """
    Read and process all Preotiuc-Ungar dataset of tweets annotated with gender, race, and age.

    :read_path: File to be read.
    :write_path: Directory of output files.
    """
    read_path = read_path.rstrip('/')
    write_path = write_path.rstrip('/')

    out_file = open(f'{write_path}/all_tweets.json', 'w', encoding = 'utf-8')
    users = open(f'{write_path}/all_users.tsv', 'w', encoding = 'utf-8')
    user_writer = csv.writer(users, delimiter = '\t')

    user_hdr = ['text', 'race', 'gender', 'birth_year', 'user_id']
    user_writer.writerow(user_hdr)

    race_dict = {1: 'African-American', 2: 'Latinx / Hispanic', 3: 'Asian', 4: 'White'}
    gender_dict = {1: 'female', 0: 'male'}

    for user in tqdm(os.listdir(read_path)):
        text = ''
        for tweet in open(f'{read_path}/{user}', 'r', encoding = 'utf-8'):
            tw = json.loads(tweet)
            text += tw['text'].replace('\n', ' ').replace('\r', ' ').strip()  + ' '
            born = tw['birth_year']
            try:
                race = race_dict[int(tw['race'])]
            except (KeyError, ValueError):
                race = 'Unknown'
            try:
                gender = gender_dict[int(tw['is_female'])]
            except (KeyError, ValueError):
                gender = 'Unknown'

            tw['race'] = race
            tw['gender'] = gender
            del tw['is_female']
            out_file.write(json.dumps(tw) + '\n')
        out = [text, race, gender, born, user]
        user_writer.writerow(out)

    out_file.close()
    users.close()
