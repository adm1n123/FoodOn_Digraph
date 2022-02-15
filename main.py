

import os, math
import pickle
import logging as log
from word2vec import WordEmbeddings
from wiki import Wikipedia
from foodon import FoodOn
from utils.utilities import load_pkl
from gensim.models import KeyedVectors
from fdc_preprocess import FDCPreprocess
import nltk
import numpy as np
import pandas as pd

def main():

    # train_embeddings()

    create_and_run_model()
    # testing()


    return None

def train_embeddings():
    we = WordEmbeddings()
    # we.get_pretrain_vectors()
    # wiki = Wikipedia()
    # wiki.parse_wiki()
    we.train_embeddings()
    return None

def create_and_run_model():
    # creating the graph and seeding the class.
    foodon = FoodOn()

    # mapping the rest entities to class.
    foodon.populate_foodon_digraph()
    return None





# ############################################################ TESTING #########################################

def testing():

    pair_pd = pd.read_csv('data/FoodOn/foodonpairs.txt', sep='\t')

    fdc_preprocess = FDCPreprocess()
    l1 = fdc_preprocess.preprocess_columns(pair_pd['Child_label'], False, False).tolist()
    l2 = fdc_preprocess.preprocess_columns(pair_pd['Parent_label'], False, False).tolist()
    ontology_list = l1+l2

    ontology_words = set()
    for word in ontology_list:
        ontology_words.update(word.split())

    print(f'Total distinct words in ontology: {len(ontology_words)}')

    count = 0
    model = KeyedVectors.load_word2vec_format('data/model/word2vec_trained.txt')
    for word in ontology_words:
        try:
            model.get_vector(word)
            count += 1
        except KeyError:
            print(f'word in ontology but not in word2vec:', word)
            pass

    print(f'Total common words in ontology and word2vec are:{count}')

    with open('data/model/bert/assets/vocab.txt', "r", encoding="utf8") as file:
        lines = file.readlines()

    count = 0
    for line in lines:
        if line.strip() in ontology_words:
            count += 1

    print(f'Total common words in ontology and bert are: {count}')




    return None


def _calculate_label_embeddings(index_list, keyed_vectors, fpm):  # make label an index.
    pd_label_embeddings = pd.DataFrame(index=index_list, columns=['preprocessed', 'vector'])
    pd_label_embeddings['preprocessed'] = fpm.preprocess_columns(pd_label_embeddings.index.to_series(), load_phrase_model=True)

    # some preprocessed columns are empty due to lemmatiazation, fill it up with original
    empty_index = (pd_label_embeddings['preprocessed'] == '')
    pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.index.to_series()[empty_index]
    pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.loc[empty_index, 'preprocessed'].apply(lambda x: x.lower())  # at least use lowercase as preprocessing.

    pd_label_embeddings['vector'] = pd_label_embeddings['preprocessed'].apply(lambda x: _calculate_embeddings(x, keyed_vectors))

    return pd_label_embeddings


def _calculate_embeddings(label, keyed_vectors):  # for a label take weighted average of word vectors.
    label_embedding = 0
    num_found_words = 0

    for word, pos in nltk.pos_tag(label.split(' ')):
        # if word in ['food', 'product']:
        #     continue
        try:
            word_embedding = keyed_vectors.get_vector(word)
        except KeyError:
            pass
        else:
            if pos == 'NN': # take NN/NNS if word is noun then give higher weightage in averaging by increasing the vector magnitude.
                multiplier = 1.15
            else:
                multiplier = 1
            label_embedding += (multiplier * word_embedding)    # increase vector magnitude if noun.
            num_found_words += 1

    if num_found_words == 0:
        return np.zeros(300)
    else:
        return label_embedding / num_found_words


def _calculate_class_entities_score(pd_entity_label_embeddings, pd_class_label_embeddings, candidate_classes_info, class_label):  # similarity of entity with average of sibling vectors.
    siblings = candidate_classes_info[class_label][1]

    num_nonzero_siblings = 0
    siblings_embedding = 0

    for sibling in siblings:
        sibling_embedding = pd_entity_label_embeddings.loc[sibling, 'vector']

        if np.count_nonzero(sibling_embedding): # when there are no words in label zero vector is its embedding. so proceed only if it is non-zero.
            siblings_embedding += sibling_embedding
            num_nonzero_siblings += 1

    if num_nonzero_siblings == 0:
        return 0

    siblings_embedding /= num_nonzero_siblings
    class_embeddings = pd_class_label_embeddings.loc[class_label, 'vector']

    score = _cosine_similarity(siblings_embedding, class_embeddings)
    return score


def _cosine_similarity(array1, array2):
    with np.errstate(all='ignore'):
        similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

    if np.isnan(similarity):
        similarity = -1

    return similarity

def remove_same_words():
    with open('data/FoodOn/correction', encoding='utf8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        words = line.split(" correction ")
        if words[0] != words[1]:
            print(f'{words[0]} correction {words[1]}')

if __name__ == '__main__':
    main()
    # remove_same_words()