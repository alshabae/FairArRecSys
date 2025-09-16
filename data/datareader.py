from collections import defaultdict, Counter
from lib2to3.pytree import convert
from typing import Dict, Hashable, List
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import math
from tqdm import tqdm
import pycountry
import pdb
from transformers import pipeline
import torch
from datasets import Dataset


def convert_to_categorical(input_list: List[Hashable], offset=0) -> Dict[Hashable, int]:
    unique_items = sorted(list(set(input_list)))
    out = {item : i + offset for i, item in enumerate(unique_items)}
    return out

def read_BARD(datasets_dir=None, core=3):

    DIALECTS = {"Algeria": 0,
    "Bahrain": 1,
    "Egypt": 2,
    "Iraq": 3,
    "Jordan": 4,
    "Kuwait": 5,
    "Lebanon": 6,
    "Libya": 7,
    "Morocco": 8,
    "Oman": 9,
    "Palestine": 10,
    "Qatar": 11,
    "Saudi_Arabia": 12,
    "Sudan": 13,
    "Syria": 14,
    "Tunisia": 15,
    "UAE": 16,
    "Yemen": 17,
    "MSA" : 18
    }

    data_dir = datasets_dir + '/BARD'
    # checkins_dir = data_dir + '/Processed_LABR.pkl'
    checkins_dir = data_dir + '/Processed_BARD.pkl'

    # df_checkins = pd.read_csv(checkins_dir, sep='\t', dtype=str, header=None, names=['rating', 'reviewID', 'bookID', 'userID', 'review', 'dialect'], encoding='utf-8')
    # df_checkins = pd.read_csv(checkins_dir, sep='\t', dtype=str, header=None, names=['rating', 'reviewID', 'userID', 'bookID', 'review', 'dialect'], encoding='utf-8')
    df_checkins = pd.read_pickle(checkins_dir)
    df_checkins.rename({'userID': 'bookID', 'bookID': 'userID'}, axis=1, inplace=True)
    # pdb.set_trace()
    df_checkins = df_checkins.drop_duplicates(subset=['bookID','userID'])

    # Apply C-core setting
    num_of_removed_rows = 1
    # print(len(df_checkins))
    while(num_of_removed_rows > 0):
        df_lengh_before_filter = len(df_checkins)
        df_checkins = df_checkins[df_checkins.groupby(['userID']).bookID.transform('count') >= core]
        df_checkins = df_checkins[df_checkins.groupby(['bookID']).userID.transform('count') >= core]
        df_lengh_after_filter = len(df_checkins)
        # print(len(df_checkins))
        num_of_removed_rows = df_lengh_before_filter - df_lengh_after_filter

    user_ids = df_checkins['userID'].unique().astype(str).tolist()
    book_ids = df_checkins['bookID'].unique().astype(str).tolist()

    userid_reindexer = convert_to_categorical(user_ids)
    bookid_reindexer = convert_to_categorical(book_ids, offset=len(user_ids))

    item_data = {}
    for i in range(len(book_ids)):
        reindexed_book_id = bookid_reindexer[book_ids[i]]

        item_data[reindexed_book_id] = {'org_book_id' : book_ids[i]}

    reindexed_ratings = defaultdict(list)
    num_dialects = len(DIALECTS)
    ratings_by_edges = {}

    for i, interaction in tqdm(enumerate(df_checkins.itertuples())):
        user_id = userid_reindexer[interaction.userID]
        item_id = bookid_reindexer[interaction.bookID]
        aldi_score = interaction.scores
        if aldi_score>0:
            multi_labels = interaction.multiLabel
        else:
            multi_labels = ['MSA']

        numbered_dialects = list(map(lambda key: DIALECTS[key], multi_labels))
        one_hot = np.zeros((num_dialects,), dtype=float)
        one_hot[numbered_dialects] = 1
        reindexed_ratings[user_id].append((item_id, aldi_score, multi_labels, numbered_dialects, one_hot))
        ratings_by_edges[(user_id,item_id)] = multi_labels
        
    # pdb.set_trace()

    user_data = {}
    for i in range(len(user_ids)):
        reindexed_user_id = userid_reindexer[user_ids[i]]

        counter = Counter()
        for tuple in reindexed_ratings[reindexed_user_id]:
            dialects = tuple[2]
            counter.update(dialects)
        most_common_dialects = [tuple[0] for tuple in counter.most_common(1)]
        numbered_dialects = list(map(lambda key: DIALECTS[key], most_common_dialects))
        one_hot = np.zeros((num_dialects,), dtype=float)
        one_hot[numbered_dialects] = 1
        user_data[reindexed_user_id] = {'org_user_id' : user_ids[i], 'most_common_dilects' : most_common_dialects}
    
    # pdb.set_trace() #(756, 5454)
    return user_data, item_data, reindexed_ratings, ratings_by_edges

if __name__ == '__main__':
    read_BARD(datasets_dir='')
