import multiprocessing

import nltk
import pandas as pd
import numpy as np
from fdc_preprocess import FDCPreprocess
from gensim.models import KeyedVectors
from time import time

# from foodon import Class


class Scoring:
    """
    TODO: don't consider entities without embeddings ignore them during precision calculation
    """
    def __init__(self, root, class_dict, entity_dict, non_seeds):
        self.root = root
        self.class_dict = class_dict
        self.entity_dict = entity_dict
        self.non_seeds = non_seeds

        self.alpha = 0.8    # for Rc
        self.beta = 0.5     # for Sc


        t1 = time()
        print('Loading word2vec...')
        self.vec_dim = 300
        self.keyed_vectors = KeyedVectors.load_word2vec_format('data/model/word2vec_trained.txt')
        t2 = time()
        print('Elapsed time %.2f minutes', (t2 - t1) / 60)

        self.fpm = FDCPreprocess()

        self.precompute_vectors()

    def precompute_vectors(self):
        t1 = time()
        class_id_labels = list([class_id, self.class_dict[class_id].raw_label] for class_id in self.class_dict.keys())
        df_class = self._calculate_label_embeddings(class_id_labels)
        for idx, row in df_class.iterrows():
            node = self.class_dict[idx]
            node.Lc = row['vector']
            node.label = row['preprocessed']
        t2 = time()
        print('Elapsed time for getting class label vectors %.2f minutes', (t2 - t1) / 60)

        t1 = time()
        entity_id_labels = list(
            [entity_id, self.entity_dict[entity_id].raw_label] for entity_id in self.entity_dict.keys())
        df_entity = self._calculate_label_embeddings(entity_id_labels)
        for idx, row in df_entity.iterrows():
            node = self.entity_dict[idx]
            node.Le = row['vector']
            node.label = row['preprocessed']
        t2 = time()
        print('Elapsed time for getting entity label vectors %.2f minutes', (t2 - t1) / 60, flush=True)

    def reset_nodes(self):
        for _, node in self.class_dict.items():
            node.Rc = None
            node.Sc = None
            node.score = None  # similarity with this class (used from child to set node.can_back = false.)
            node.predicted_entities = []  # predicted entities for this class
            node.backtrack = None  # can backtrack store the entity id or iteration number if iteration no == this iteration then do not backtrack.
            node.visited = 0

        for _, entity in self.entity_dict.items():
            entity.score = None  # list of all the scores during traversal.
            entity.predicted_class = None
        return None


    def run_config(self):
        t1 = time()
        prec = []
        print('Running config ...')

        for alpha in [.8]: #x*.1 for x in range(4, 8)]:
            for beta in [1]: #x*.1 for x in range(4, 8)]:
                self.reset_nodes()
                self.alpha = alpha
                self.beta = beta

                self.precompute_tree_nodes()
                self.predict_entity()

                p = self.calculate_precision()
                self.find_avg_path_length()
                prec.append([alpha, beta, p])
                print(f'alpha: {alpha}, beta: {beta}, prec: {p}')
        t2 = time()
        print('Elapsed time for running config search %.2f minutes', (t2 - t1) / 60)
        prec.sort(key=lambda x: x[2], reverse=True)
        for p in prec:
            print(f'alpha: {p[0]}, beta: {p[1]}, precision: {p[2]}')

        return None





    def precompute_tree_nodes(self):
        print(f'Root is: {self.root.ID}')
        self.post_order_traversal(self.root, depth=0)
        return None

    def post_order_traversal(self, root, depth):

        if len(root.children) == 0: # leaf class
            root.Rc = self._Rc(root)
            root.Sc = self._Sc(root)    # for leaf Sc = Rc
            return None

        for child in root.children:
            self.post_order_traversal(child, depth+1)

        root.Rc = self._Rc(root)
        root.Sc = self._Sc(root)
        return None



    def predict_entity(self):
        failed = 0
        count = 0

        for entity in self.non_seeds:
            entity.score = -2
            # self.traverse_with_blocking(self.root, entity); print('\n Traversing_with_blocking_backtrack\n')
            # self.traverse_all_Rc(self.root, entity); print('\n Traversing_all_classes_Rc\n')
            self.traverse_greedy(self.root, entity); print('\n Traversing_greedily\n')

            pred_class = entity.predicted_class
            if pred_class:
                pred_class.predicted_entities.append(entity)
            else:
                failed += 1
            count += 1
            if count % 500 == 0:
                print(f'\r Total entities predicted: {count}/{len(self.non_seeds)}')

        print('Failed entities to map', failed)
        return None


    def traverse_all_Rc(self, root, entity):
        score = self._cosine_similarity(root.Rc, entity.Le)   # score of class with current entity used for traversal only.

        if score > entity.score:  # compare with any previous class scores
            entity.score = score
            entity.predicted_class = root

        for child in root.children:
            self.traverse_all_Rc(child, entity)

        return None

    def traverse_greedy(self, root, entity):    # traverse all children with high score. don't block backtrack.
        root.score = self._cosine_similarity(root.Sc, entity.Le)   # score of class with current entity used for traversal only.

        if root.score > entity.score:  # compare with any previous class scores
            entity.score = root.score
            entity.predicted_class = root

        if len(root.children) == 0:
            return None

        for child in root.children:
            score = self._cosine_similarity(child.Sc, entity.Le)
            if score >= root.score: # take all child with score >=
                self.traverse_greedy(child, entity)

        return None


    def traverse_with_blocking(self, root, entity):
        root.backtrack = True
        root.score = self._cosine_similarity(root.Sc, entity.Le)   # score of class with current entity used for traversal only.

        if root.score > entity.score:  # compare with any previous class scores
            entity.score = root.score
            entity.predicted_class = root

        if len(root.children) == 0:
            return None

        child_scores = []
        for child in root.children:
            score = self._cosine_similarity(child.Sc, entity.Le)
            if score >= root.score: # take all child with score >=
                child_scores.append([child, score])

        if len(child_scores) == 0:
            return None

        child_scores.sort(key=lambda x: x[1], reverse=True)

        if child_scores[0][1] > root.score: # if child's highest score >
            root.backtrack = False

        for item in child_scores:
            child = item[0]
            self.traverse_with_blocking(child, entity)
            if not child.backtrack: # no need to traverse in rest of the subtrees(children)
                root.backtrack = False
                return None

        return None




    def _Rc(self, node):
        non_zero_entities = []
        for entity in node.seed_entities:
            if np.any(entity.Le):
                non_zero_entities.append(entity.Le)

        if len(non_zero_entities) > 0:
            entity_avg = np.mean(non_zero_entities, axis=0)
            return np.add(self.alpha * node.Lc, (1 - self.alpha) * entity_avg)
        else:
            return np.add(self.alpha * node.Lc, (1 - self.alpha) * np.zeros(self.vec_dim))


    def _Sc(self, node):
        non_zero_subclasses = []
        for subclass in node.children:
            if np.any(subclass.Sc):
                non_zero_subclasses.append(subclass.Sc)

        if len(non_zero_subclasses) > 0:
            subclass_avg = np.mean(non_zero_subclasses, axis=0)
            return np.add(self.beta * node.Rc, (1 - self.beta) * subclass_avg)
        else:
            return np.copy(node.Rc)


    def _caculate_embeddings(self, label):  # for a label take weighted average of word vectors.
        label_embedding = 0
        num_found_words = 0

        for word, pos in nltk.pos_tag(label.split(' ')):
            # if word in ['food', 'product']:
            #     continue
            try:
                word_embedding = self.keyed_vectors.get_vector(word)
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
            return np.zeros(self.vec_dim)
        else:
            return label_embedding / num_found_words

    def _calculate_label_embeddings(self, index_list):  # make label an index.
        pd_label_embeddings = pd.DataFrame(index_list, columns=['ID', 'label'])
        pd_label_embeddings.set_index('ID', inplace=True)

        # pd_label_embeddings = pd.DataFrame(index=index_list, columns=['preprocessed', 'vector'])
        pd_label_embeddings['preprocessed'] = self.fpm.preprocess_columns(pd_label_embeddings['label'], load_phrase_model=False)

        # some preprocessed columns are empty due to lemmatiazation, fill it up with original
        empty_index = (pd_label_embeddings['preprocessed'] == '')
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings['label'][empty_index]
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.loc[empty_index, 'preprocessed'].apply(lambda x: x.lower())  # at least use lowercase as preprocessing.

        pd_label_embeddings['vector'] = pd_label_embeddings['preprocessed'].apply(self._caculate_embeddings)

        return pd_label_embeddings

    def _cosine_similarity(self, array1, array2):
        with np.errstate(all='ignore'):
            similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

        if np.isnan(similarity):
            similarity = -1

        return similarity


    def calculate_precision(self):
        TP, FP = 0, 0
        for entity in self.non_seeds:
            if not np.any(entity.Le):   # ignore entities without labels
                continue
            elif entity.predicted_class is None:    # coud not predict class good to put them
                FP += 1
            elif entity in entity.predicted_class.all_entities:
                TP += 1
            else:
                FP += 1
        print(f'Precision is: {TP/(TP+FP)}, TP: {TP}, TP+FP:{TP+FP}')

        return TP/(TP+FP)


    def find_avg_path_length(self):
        length = 0
        count = 0
        max_len = 0
        for entity in self.non_seeds:
            parent = entity.parents[0]
            pred_parent = entity.predicted_class
            if pred_parent is None:
                continue
            path_p = []
            path_pred = []
            node = parent
            while node != self.root:
                path_p.append(node)
                node = node.parents[0]

            node = pred_parent
            while node != self.root:
                path_pred.append(node)
                node = node.parents[0]

            unique = set(path_p) - set(path_pred)
            length += len(unique) + 1   # +1 for LCA
            count += 1

            if len(unique) + 1 > max_len:
                max_len = len(unique) + 1

        print(f' Average path length between actual and predicted class: {length/count}, for {count} entities, max_length: {max_len}')


    def detect_cycle(self, root):
        for child in root.children:
            self.detect_cycle(child)

        return None

    def break_cycles(self, root):   # remove cycle by depth first search so that lower level nodes could not refer back.
        root.in_path = True
        child_in_path = []
        for child in root.children:
            if child.in_path:   # how can child reached in DFS remove this cycle.
                child_in_path.append(child)

        root.children = [child for child in root.children if child not in child_in_path]    # remove child

        for child in child_in_path:     # remove parent from child
            child.parents.remove(root)

        for child in root.children:
            child.in_path = True
            self.break_cycles(child)
            child.in_path = False

        root.in_path = False
        return None
