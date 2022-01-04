

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
    # populated_filepath = 'data/scores/populated.pkl'
    # iteration_populated_dict = load_pkl(populated_filepath)

    # skeleton_and_entities_pkl = 'data/FoodOn/skeleton_candidate_classes_dict.pkl'
    # skeleton_candidate_classes_dict, candidate_entities = load_pkl(skeleton_and_entities_pkl)

    # candidate_ontology_pkl = 'data/FoodOn/candidate_classes_dict.pkl'
    # candidate_classes_dict = load_pkl(candidate_ontology_pkl)


    model = KeyedVectors.load_word2vec_format('data/model/word2vec_trained.txt')
    vectors = []
    for key in ['food', 'product', 'dry', 'fry', 'chicken', 'meat', 'mutton', 'paneer', 'tomato', 'milk', 'cream', 'process', 'papaya', 'whole', 'raw', 'fruit', 'frost', 'mix', 'cheese', 'cake', 'crabmeat', 'packed', 'freeze', 'hen', 'egg']:
        vectors.append([key, math.sqrt(np.sum(np.square(model[key])))])

    vectors = sorted(vectors, key=lambda x: x[1], reverse=True)
    vectors = vectors

    for entry in vectors:
        print(entry)



    print(model.most_similar(positive=['king', 'woman'], negative=['man']))
    print(model.most_similar(positive=['apple']))
    print(model.most_similar(positive=['fruit']))

    class_label = 'beverage food product'

    value = candidate_classes_dict[class_label]
    fdc = FDCPreprocess()
    pd_entities_label_embeddings = _calculate_label_embeddings(value[1], model, fdc)
    pd_class_label_embedding = _calculate_label_embeddings([class_label], model, fdc)

    score = _calculate_class_entities_score(pd_entities_label_embeddings, pd_class_label_embedding, candidate_classes_dict, class_label)
    print(f'beverage class score with its entities: {score}')





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


if __name__ == '__main__':
    main()