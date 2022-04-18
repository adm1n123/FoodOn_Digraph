import math, random

import tensorflow_hub as hub
import tensorflow_text as text
import nltk
from nltk.tree import Tree
import pandas as pd
import numpy as np
from fdc_preprocess import FDCPreprocess
from gensim.models import KeyedVectors
from time import time


class Scoring:
    """
    TODO: don't consider entities without embeddings ignore them during precision calculation
    """
    def __init__(self, root, class_dict, entity_dict, non_seeds):
        self.root = root
        self.class_dict = class_dict
        self.entity_dict = entity_dict
        self.non_seeds = non_seeds

        self.alpha = 0.4    # for Rc
        self.beta = 0.6     # for Sc
        self.bias = 0.08      # for childscore + bias >= parent score.

        self.isEntity = False
        self.vec_dim = 300
        t1 = time()
        # print('Loading BERT...', end='')
        # self.bert_preprocessor, self.bert_model = None, None
        # self.load_BERT()
        self.keyed_vectors = KeyedVectors.load_word2vec_format('data/model/word2vec_trained_new.txt')

        t2 = time()
        print('Elapsed time %.2f minutes' % ((t2 - t1) / 60))

        self.fpm = FDCPreprocess()
        self.type_count = {}
        self.precompute_vectors()

        # print('dumping class:entity pair into file')
        # self.dump_class_entity_pairs(root)

    def precompute_vectors(self):
        print('precomputing vectors...', end='')

        t1 = time()
        class_id_labels = list([class_id, self.class_dict[class_id].raw_label] for class_id in self.class_dict.keys())
        df_class = self._calculate_label_embeddings(class_id_labels)
        print(f'class labels type stats:{self.type_count}, total:{len(class_id_labels)}')
        for idx, row in df_class.iterrows():
            node = self.class_dict[idx]
            node.Lc = row['vector']
            node.label = row['preprocessed']
            node.head = self.find_head(node.raw_label)

        t2 = time()
        print('\rElapsed time for getting class label vectors %.2f minutes' % ((t2 - t1) / 60))

        self.isEntity = True
        self.type_count = {}
        t1 = time()
        entity_id_labels = list(
            [entity_id, self.entity_dict[entity_id].raw_label] for entity_id in self.entity_dict.keys())
        df_entity = self._calculate_label_embeddings(entity_id_labels)
        print(f'entity labels type stats:{self.type_count}, total:{len(entity_id_labels)}')
        print('#######################################################################################################')
        for idx, row in df_entity.iterrows():
            node = self.entity_dict[idx]
            node.Le = row['vector']
            node.label = row['preprocessed']
            node.head = self.find_head(node.raw_label)

        t2 = time()
        print('Elapsed time for getting entity label vectors %.2f minutes' % ((t2 - t1) / 60))

        # self.count_all_words(df_class, df_entity)

    def reset_nodes(self):
        seeds = []
        all = []
        for _, node in self.class_dict.items():
            node.Rc = None
            node.Sc = None
            # node.score = None  # similarity with this class (used from child to set node.can_back = false.)
            node.predicted_entities = []  # predicted entities for this class
            # node.backtrack = None  # can backtrack store the entity id or iteration number if iteration no == this iteration then do not backtrack.
            node.visited = 0
            node.Rc_sum = None
            node.Rc_count = 0
            node.pre_proc = False
            node.visited_for = None

            all.extend(node.all_entities)
            seeds.extend(node.seed_entities)

        self.non_seeds = list(set(all) - set(seeds))
        for _, entity in self.entity_dict.items():
            entity.score = 0  # max score during traversal.
            entity.predicted_class = None
            entity.visited_classes = 0

        return None

    def print_stats(self):
        parents_c = 0
        max_parents = 0
        classes = 0
        for _, node in self.class_dict.items():
            parents_c += len(node.parents)
            classes += 1
            if len(node.parents) > max_parents:
                max_parents = len(node.parents)
        print(f'Average number of parents to a class: {parents_c/classes:.2f}')

        return None

    def run_config(self):
        t1 = time()
        prec = []
        print('\n\nRunning config α * Lc + (1-α) * seeds,  β * Sc + (1-β) * children(Sc avg)')
        self.print_stats()

        for _ in range(20):
            for alpha in [.4]:#[x*.1 for x in range(1, 9)]:
                for beta in [.6]: #[x*.1 for x in range(2, 9, 2)]:
                    self.reset_nodes()
                    self.alpha = alpha
                    self.beta = beta
                    self.bias = .08

                    self.precompute_tree_nodes()
                    self.predict_entity()

                    p = self.calculate_precision()
                    # self.find_avg_path_length()
                    prec.append([alpha, beta, p])
                    print(f'alpha: {alpha:.1f}, beta: {beta:.1f}, bias: {self.bias}, prec: {p:.3f}')
                    self.take_best_seeds(self.root)
                    self.sample_seeds(self.root)


            t2 = time()
            print('Elapsed time for running config search %.2f minutes' % ((t2 - t1) / 60))
            prec.sort(key=lambda x: x[2], reverse=True)
            for p in prec:
                print(f'alpha: {p[0]:.1f}, beta: {p[1]:.1f}, precision: {p[2]:.3f}')

        self.assign_best_seeds(self.root)
        self.reset_nodes()
        self.precompute_tree_nodes()
        self.predict_entity()

        p = self.calculate_precision()
        print(f'Final precision after taking best seeds is:{p}')

        # print('printing best seeds for classes')
        # for _, node in self.class_dict.items():
        #     print(f'class:{node.raw_label}      Best seeds:{[entity.raw_label for entity in node.seed_entities]}')

        return None


    def precompute_tree_nodes(self):
        print(f'Root is: {self.root.ID}')
        # print(f'running post_order_traversal for Rc, Sc computation.')
        print('\nUsing Rc_sum as Sc')
        self.post_order_traversal_avg_Rc(self.root)
        return None

    def take_best_seeds(self, root):

        if len(root.seed_entities) == 0 and len(root.all_entities) != 0:
            return

        if len(set(root.all_entities) - set(root.predicted_entities)) < root.worst_count:
            root.worst_count = len(set(root.all_entities) - set(root.predicted_entities))
            root.best_seeds = root.seed_entities

        root.seed_entities = []

        for child in root.children:
            self.take_best_seeds(child)

    def sample_seeds(self, root):
        if len(root.seed_entities) > 0:
            return

        root.seed_entities = random.sample(root.all_entities, root.seed_count)

        for child in root.children:
            self.sample_seeds(child)

    def assign_best_seeds(self, root):
        root.seed_entities = root.best_seeds
        for child in root.children:
            self.assign_best_seeds(child)

    def post_order_traversal_avg_Rc(self, root):    # find average of all Rc vectors in root subtree.
        if root.pre_proc:   # return because child has many parents.
            return

        root.Rc = self._Rc(root)

        if len(root.all_entities) == 0:
            root.Rc_count = 0
            root.Rc_sum = np.zeros(self.vec_dim)
        else:
            root.Rc_count = 1
            root.Rc_sum = np.copy(root.Rc)

        if len(root.children) == 0:
            root.Sc = np.copy(root.Rc_sum)
            root.pre_proc = True
            return

        for child in root.children:
            self.post_order_traversal_avg_Rc(child)
            root.Rc_sum = np.add(root.Rc_sum, child.Rc_sum)
            root.Rc_count += child.Rc_count

        root.Sc = root.Rc_sum / root.Rc_count   # avg of Rc vectors
        root.pre_proc = True
        return

    def post_order_traversal(self, root, depth):
        if root.pre_proc:   # return because child has many parents.
            return

        if len(root.children) == 0: # leaf class
            root.Rc = self._Rc(root)
            root.Sc = self._Sc(root)    # for leaf Sc = Rc
            root.pre_proc = True
            return None

        for child in root.children:
            self.post_order_traversal(child, depth+1)

        root.Rc = self._Rc(root)
        root.Sc = self._Sc(root)
        root.pre_proc = True
        return None

    def predict_entity(self):
        failed = 0
        count = 0
        visited_classes = 0
        # print('\n Traversing_all_classes_Rc\n')
        print('\nTraversing_greedily all subtree with higher score than parent, predict class using Rc, Traverse subtrees using Sc\n')
        for entity in self.non_seeds:
            entity.score = -1e5
            # entity.score_rmv = -1
            # self.traverse_all_Rc(self.root, entity)

            self.traverse_greedy(self.root, entity, -1e5)
            visited_classes += entity.visited_classes
            pred_class = entity.predicted_class

            if pred_class:
                pred_class.predicted_entities.append(entity)
            else:
                failed += 1
            count += 1
            if count % 100 == 0:
                print(f'\r Total entities predicted: {count}/{len(self.non_seeds)}', end='')
        print(f'\r Total entities predicted: {count}/{len(self.non_seeds)}')
        print(f'Failed to predict:{failed} entities, Average number of classes visited for prediction:{visited_classes/count:.1f}')
        return None


    def traverse_all_Rc(self, root, entity):
        if root.visited_for == entity:  # don't visited same node for same entity.
            return

        score = self._cosine_similarity(root.Rc, entity.Le)   # score of class with current entity used for traversal only.
        entity.visited_classes += 1
        if score > entity.score and len(root.all_entities) > 0:  # compare with any previous class scores
            entity.score = score
            entity.predicted_class = root

        for child in root.children:
            self.traverse_all_Rc(child, entity)

        root.visited_for = entity
        return None

    def dump_class_entity_pairs(self, root):
        file = open('data/FoodOn/class_entity.txt', 'w', encoding="utf-8")
        data = set()
        self.dump_to_file(root, data)
        for root, entity in data:
            file.write(f'{root}:{entity}\n')
        file.close()

    def dump_to_file(self, root, data):
        for entity in root.all_entities:
            data.add((root.label, entity.label))

        for child in root.children:
            self.dump_to_file(child, data)

    def traverse_greedy(self, root, entity, max_path_score):    # traverse all children with high score. don't block backtrack.
        if root.visited_for == entity:  # don't visited same node for same entity.
            return

        score = self._cosine_similarity(root.Rc, entity.Le)   # score of class with current entity used for traversal only.
        # if len(entity.head) >= 1 and entity.head[0] in root.head:
        #     score *= 1.15
        # if len(entity.head) >= 2 and entity.head[1] in root.head:
        #     score *= 1.15

        entity.visited_classes += 1
        if score > entity.score and len(root.all_entities) > 0:  # compare with any previous class scores
            entity.score = score
            entity.predicted_class = root

        if len(root.children) == 0:
            root.visited_for = entity
            return

        score = self._cosine_similarity(root.Sc, entity.Le)
        max_path_score = score if score > max_path_score else max_path_score

        for child in root.children:
            child_score = self._cosine_similarity(child.Sc, entity.Le)
            if child_score + self.bias >= max_path_score or root == self.root: # take all child with score >=, and consider all children of root.
                self.traverse_greedy(child, entity, max_path_score)

        root.visited_for = entity
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

    def find_noun_phrases(self, tree):
        return [subtree for subtree in tree.subtrees(lambda t: t.label() == 'NP')]

    def find_head_of_np(self, np):
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
        ## search for a top-level noun
        top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
        if len(top_level_nouns) > 0:
            ## if you find some, pick the rightmost one, just 'cause
            return top_level_nouns[-1][0]
        else:
            ## search for a top-level np
            top_level_nps = [t for t in top_level_trees if t.label() == 'NP']
            if len(top_level_nps) > 0:
                ## if you find some, pick the head of the rightmost one, just 'cause
                return self.find_head_of_np(top_level_nps[-1])
            else:
                ## search for any noun
                nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
                if len(nouns) > 0:
                    ## if you find some, pick the rightmost one, just 'cause
                    return nouns[-1]
                else:
                    ## return the rightmost word, just 'cause
                    return np.leaves()[-1]

    def get_head(self, label):
        grammar = "NP: {<DT>?<JJ>*<NN>}"
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(nltk.pos_tag(nltk.word_tokenize(label)))
        # print(result)
        for np in self.find_noun_phrases(result):
            head = self.find_head_of_np(np)
            if head is not None:
                return head[0]

    def load_BERT(self):
        preprocess_url = "data/model/bert_preprocessor"  # "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        encoder_url = "data/model/bert"  # "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

        self.bert_preprocessor = hub.KerasLayer(preprocess_url)
        self.bert_model = hub.KerasLayer(encoder_url)

    def _calculate_BERT_embeddings(self, labels):
        bert_embedding = []
        batch_size = 100
        for batch in range(0, math.ceil(len(labels)/batch_size)):
            start, end = batch*100, batch*100+batch_size
            labels_slice = labels[start:end]
            preprocessed_text = self.bert_preprocessor(labels_slice)
            bert_results = self.bert_model(preprocessed_text)
            label_vectors = bert_results['pooled_output']

            for idx in range(len(labels_slice)):
                bert_embedding.append(label_vectors[idx].numpy())

            print(f'\r calculating bert embeddings:{batch/(len(labels)//batch_size):.2f}', end='')

        print()

        return bert_embedding

    def noun_before(self, label, regx):
        strings = label.split(regx)
        left = strings[0].strip().split()
        last = None
        for word, pos in nltk.pos_tag(left):
            if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                last = word
        return [last]

    def noun_after(self, label, regx):
        strings = label.split(regx)
        right = strings[1].strip().split()
        flag = False
        for word, pos in nltk.pos_tag(label.split(' ')):
            if word == regx.strip():
                flag = True
            elif flag and pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                return [word]
        return [right[0]]

    def find_head(self, label):
        label = label.lower()
        label = label.replace("-", " ")

        if "(" in label:
            self.type_count["("] = self.type_count.get("(", 0) + 1
            label = label.replace(",", "")
            head = self.noun_before(label, "(")
            # print(f'type:( label:{label}, head:{head}')
            return head
        if " or " in label:
            self.type_count["or"] = self.type_count.get("or", 0) + 1
            label = label.replace(",", "")
            strings = label.split(" or ")
            left = strings[0].strip().split()
            right = strings[1].strip().split()
            head = [left[-1], right[0]]
            # print(f'type:or label:{label}, head:{head}')
            return head
        if " based " in label:
            self.type_count["based"] = self.type_count.get("based", 0) + 1
            label = label.replace(",", "")
            head = self.noun_after(label, " based ")
            # print(f'type:based label:{label}, head:{head}')
            return head
        if " packed in " in label:
            self.type_count["packed in"] = self.type_count.get("packed in", 0) + 1
            label = label.replace(",", "")
            label = label.replace("packed in", "packed_in")
            head = self.noun_before(label, " packed_in")
            # print(f'type:packed in label:{label}, head:{head}')
            return head
        if " with " in label:
            self.type_count["with"] = self.type_count.get("with", 0) + 1
            label = label.replace(",", "")
            head = self.noun_before(label, " with ")
            # print(f'type:with label:{label}, head:{head}')
            return head
        if " in " in label:
            self.type_count["in"] = self.type_count.get("in", 0) + 1
            label = label.replace(",", "")
            head = self.noun_before(label, " in ")
            # print(f'type:in label:{label}, head:{head}')
            return head
        if " made from " in label:
            self.type_count["made from"] = self.type_count.get("made from", 0) + 1
            label = label.replace(",", "")
            label = label.replace("made from", "made_from")
            head = self.noun_before(label, " made_from ")
            # print(f'type:made from label:{label}, head:{head}')
            return head
        if " for " in label:
            self.type_count["for"] = self.type_count.get("for", 0) + 1
            label = label.replace(",", "")
            head = self.noun_before(label, " for ")
            # print(f'type:for label:{label}, head:{head}')
            return head
        if "," in label:
            self.type_count[","] = self.type_count.get(",", 0) + 1
            head = self.noun_before(label, ",")
            # print(f'type:, label:{label}, head:{head}')
            return head
        # if "food product" in label:
        #     self.type_count["food product"] = self.type_count.get("food product", 0) + 1
        #     label = label.replace(",", "")
        #     label = label.replace("food product", "food_product")
        #     return self.noun_before(label, " food_product")
        # if "family" in label:
        #     self.type_count["family"] = self.type_count.get("family", 0) + 1
        #     label = label.replace(",", "")
        #     return self.noun_before(label, " family")

        self.type_count["rest"] = self.type_count.get("rest", 0) + 1

        # label = label.replace(",", "")
        # label = label.replace("food", "")
        # label = label.replace("products", "")
        # label = label.replace("product", "")
        # last = None
        # for word, pos in nltk.pos_tag(label.split()):
        #     if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        #         last = word
        # if last is None:
        #     return [label.split()[-1]]
        # return [last]
        return []


    def _calculate_embeddings(self, label):  # for a label take weighted average of word vectors.
        preprocessed = self.fpm.preprocess_columns(pd.Series(label), load_phrase_model=False, generate_phrase=False)
        p_label = preprocessed.tolist()[0]
        if p_label == '':
            p_label = label

        label_embedding = 0
        num_found_words = 0
        flag = False
        head = self.find_head(label)
        self.get_head(p_label)
        # print(f'label: {label}, head: {head}')
        for word, pos in nltk.pos_tag(p_label.split(' ')):
            # if word in ['food', 'product']:
            #     continue
            if word == 'food':
                flag = True
            try:
                word_embedding = self.keyed_vectors.get_vector(word)
            except KeyError:
                pass
            else:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS']: # take NN/NNS if word is noun then give higher weightage in averaging by increasing the vector magnitude.
                    multiplier = 1.1
                else:
                    multiplier = 1
                if word in ['dry', 'dried', 'slice', 'sliced', 'fried', 'fry', 'process', 'processed', 'frozen', 'mix', 'cook', 'cooked', 'diced', 'glazed', 'food', 'product', 'based']:
                    continue
                    # multiplier = .2
                    # num_found_words -= 1
                if word in head:
                    multiplier = 1.3
                label_embedding += (multiplier * word_embedding)    # increase vector magnitude if noun.
                num_found_words += 1
        if num_found_words == 0:
            return np.zeros(self.vec_dim)
        else:
            if 0.01 < self._cosine_similarity(self.keyed_vectors.get_vector('food'), label_embedding/num_found_words) < .5:
                label_embedding += self.keyed_vectors.get_vector('food')
                num_found_words += 1
            return label_embedding / num_found_words


    def _calculate_label_embeddings(self, index_list):  # make label an index.
        pd_label_embeddings = pd.DataFrame(index_list, columns=['ID', 'label'])
        pd_label_embeddings.set_index('ID', inplace=True)

        pd_label_embeddings['preprocessed'] = self.fpm.preprocess_columns(pd_label_embeddings['label'],load_phrase_model=False,generate_phrase=False)

        # some preprocessed columns are empty due to lemmatiazation, fill it up with original
        empty_index = (pd_label_embeddings['preprocessed'] == '')
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings['label'][empty_index]
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.loc[empty_index, 'preprocessed'].apply(lambda x: x.lower())

        pd_label_embeddings['vector'] = pd_label_embeddings['label'].apply(self._calculate_embeddings)
        # pd_label_embeddings['vector'] = self._calculate_BERT_embeddings(pd_label_embeddings['preprocessed'].tolist())

        return pd_label_embeddings

    def spell_check(self, list):
        from spellchecker import SpellChecker

        spell = SpellChecker()
        words = []
        for item in list:
            for word in item.split():
                words.append(word)

        misspelled = spell.unknown(words)
        for word in misspelled:
            print(f':{word}: correction :{spell.correction(word)}:')



    def _cosine_similarity(self, array1, array2):
        with np.errstate(all='ignore'):
            similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

        if np.isnan(similarity):
            similarity = -1

        return similarity


    def calculate_precision(self):
        TP, FP = 0, 0
        count = 0
        not_predicted = 0
        for entity in self.non_seeds:
            if not np.any(entity.Le):   # ignore entities without labels
                continue
            elif entity.predicted_class is None:    # coud not predict class good to put them
                FP += 1
                not_predicted += 1
            elif entity in entity.predicted_class.all_entities:
                TP += 1
            else:
                FP += 1
                if len(entity.predicted_class.all_entities) == 0:
                    count += 1

        print(f'Precision is: {TP/(TP+FP):.3f}, TP: {TP}, TP+FP:{TP+FP}')
        print(f'Could not predict:{not_predicted} entities, {count} entities predicted to class without entity')

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
            path_p.append(self.root)

            node = pred_parent
            while node != self.root:
                path_pred.append(node)
                node = node.parents[0]
            path_pred.append(self.root)

            unique = set(path_p+path_pred) - set(path_p).intersection(set(path_pred))
            length += len(unique) + 1   # +1 for LCA
            count += 1

            if len(unique) + 1 > max_len:
                max_len = len(unique) + 1

        print(f' Average path length between actual and predicted class: {length/count:.2f}, for {count} entities, max_length: {max_len}')


    def remove_multiple_parents(self):
        for _, child in self.class_dict.items():
            if len(child.parents) == 0:
                continue
            for parent in child.parents[1:]:
                parent.children = list(set(parent.children) - {child})

            child.parents = [child.parents[0]]  # store only first parent.

        return


    def run_analysis(self):

        for alpha in [.4]:#[x*.1 for x in range(1, 9)]:
            for beta in [.6]: #[x*.1 for x in range(2, 9, 2)]:
                self.reset_nodes()
                self.alpha = alpha
                self.beta = beta

                self.precompute_tree_nodes()
                self.predict_analysis()


        return None

    def predict_analysis(self):
        count = 0
        for entity in self.non_seeds:
            count += 1
            if count % 50 == 0:
                print(f'done:{count}/{len(self.non_seeds)}')
            entity.score = -2
            self.traverse_all_Rc(self.root, entity)
            linear_p = entity.predicted_class

            entity.score = -2
            self.traverse_greedy(self.root, entity)
            tree_p = entity.predicted_class

            if entity not in linear_p.all_entities and entity not in tree_p.all_entities:
                continue

            if linear_p == tree_p:
                continue

            print(f'entity:{entity.ID,entity.raw_label} tree predicted class:{tree_p.raw_label}, linear predicted class:{linear_p.raw_label}')

            if entity in linear_p.all_entities:
                print(f'entity:{entity.ID,entity.raw_label} predicted correctly with linear search but not with tree search')

            if entity in tree_p.all_entities:
                print(f'entity:{entity.ID,entity.raw_label} predicted correctly with tree search but not with linear search')
                continue

            path_p = linear_p
            while path_p != self.root:
                path_p.in_path = True
                path_p = path_p.parents[0]

            entity.score = -2
            self.ext = False
            self.traverse_analysis(self.root, entity)

            path_p = linear_p
            while path_p != self.root:
                path_p.in_path = False
                path_p = path_p.parents[0]

        return

    def traverse_analysis(self, root, entity):    # traverse all children with high score. don't block backtrack.
        if self.ext:
            return
        score = self._cosine_similarity(root.Rc, entity.Le)   # score of class with current entity used for traversal only.
        entity.visited_classes += 1
        if score > entity.score and len(root.all_entities) > 0:  # compare with any previous class scores
            entity.score = score
            entity.predicted_class = root

        if len(root.children) == 0:
            return None

        score = self._cosine_similarity(root.Sc, entity.Le)
        for child in root.children:
            child_score = self._cosine_similarity(child.Sc, entity.Le)
            if child_score >= score: # take all child with score >=
                self.traverse_greedy(child, entity)
            elif child.in_path:
                print(f'Deviating from path')
                self.ext = True


        return None

    def bad_precision(self):
        for entity in self.non_seeds:
            if not np.any(entity.Le):   # ignore entities without labels
                continue
            elif entity.predicted_class is None:    # coud not predict class good to put them
                continue
            elif entity not in entity.predicted_class.all_entities:

                c_class = entity.parents[0]
                all = len(c_class.all_entities)
                if all < 5:
                    continue
                pred_e = len(c_class.predicted_entities)
                if pred_e < all/2:  # if predicted entities are less than half then correct entities are definitely less.
                    if entity.raw_label in ['crabmeat (processed)', 'pineapple (immature, sliced, in brine)', 'frozen nondairy frosting',
                                            'frozen nondairy topping', 'beverage prepared from dry mix', 'fruit (dried, diced, and glazed)',
                                            'dessert mix, dry', 'vegetable (processed)', 'sugar cane (unrefined)', 'crabmeat (cooked, canned)',
                                            'chickpea (cooked, canned)']:
                        z = 2+3

                    c_score = self._cosine_similarity(entity.Le, c_class.Rc)
                    print(f'entity: {entity.raw_label} not predicted correctly. correct class: {c_class.raw_label}, score: {c_score:.2f}, predicted class: {entity.predicted_class.raw_label}, score: {entity.score:.2f}')

    def find_worst_classes(self):
        print('\n\n\n\n\n')
        for _, node in self.class_dict.items():
            if len(node.all_entities) < 10:
                continue

            if len(set(node.all_entities)-set(node.predicted_entities+node.seed_entities)) / len(set(node.all_entities)-set(node.seed_entities)) > .2:
                print(f'\nclass with < 20% precision class: {node.raw_label}')
                print('seeds: ', end=' ')
                for entity in node.seed_entities:
                    print(f'{entity.raw_label}', end='; ')

                print('\n not predicted: ', end=' ')
                for entity in node.all_entities:
                    if entity not in node.seed_entities and entity not in node.predicted_entities:
                        print(f'{entity.raw_label}', end='; ')
                print('')


    def find_diff_old_rmv_method(self):
        print(f'\n\n\nComparison of new and old method ')

        correct_old = 0
        correct_rmv = 0
        correct_both = 0

        incorrect_old = 0
        incorrect_rmv = 0

        for entity in self.non_seeds:
            if not np.any(entity.Le):  # ignore entities without labels
                continue
            if entity.predicted_class is None or entity.predicted_class_rmv is None:  # coud not predict class good to put them
                continue

            if entity in entity.predicted_class_rmv.all_entities and entity in entity.predicted_class.all_entities:
                correct_both += 1
                print(f'Entity predicted correctly for both: {entity.label}:  class: {entity.predicted_class.label}')
            elif entity in entity.predicted_class.all_entities and entity.score >= max(entity.score, entity.score_rmv):
                correct_old += 1
                print(f'Entity remained unchanged: {entity.label}:  class: {entity.predicted_class.label}')
            elif entity in entity.predicted_class_rmv.all_entities and entity.score_rmv >= max(entity.score, entity.score_rmv):
                correct_rmv += 1
                print(f'Entity predicted correctly because of rmv: {entity.label} \ncorrect class: {entity.predicted_class_rmv.label}\n old method class:{entity.predicted_class.label}', end='\n')

            if entity in entity.predicted_class.all_entities and entity.score < max(entity.score, entity.score_rmv) and entity not in entity.predicted_class_rmv.all_entities:
                incorrect_old += 1
                print(f'Entity predicted incorrectly because of rmv: {entity.label} \ncorrect class: {entity.predicted_class.label} \nnew method class: {entity.predicted_class_rmv.label}')


        print(f'correct_old:{correct_old}, correct_new:{correct_rmv}, correct_common:{correct_both}, incorrect_old:{incorrect_old}')

        return None

    def count_all_words(self, df_class, df_entity):
        tc,te,c,e = 0,0,0,0
        for idx, row in df_class.iterrows():
            tc += 1
            node = self.class_dict[idx]
            if len(node.label) < 1:
                node.all_words = False
                c += 1
                print('class:', node.raw_label, ' word: ;')
                continue
            for word in node.label.split():
                try:
                    word_embedding = self.keyed_vectors.get_vector(word)
                except KeyError:
                    node.all_words = False
                    print('class:', node.raw_label, ' word: ', word)
                    pass
            if not node.all_words:
                c += 1


        for idx, row in df_entity.iterrows():
            te += 1
            node = self.entity_dict[idx]
            if len(node.label) < 1:
                node.all_words = False
                e += 1
                print('entity:', node.raw_label, ' word: ;')
                continue
            for word in node.label.split():
                try:
                    word_embedding = self.keyed_vectors.get_vector(word)
                except KeyError:
                    node.all_words = False
                    print('entity:', node.raw_label, ' word: ', word)
                    pass
            if not node.all_words:
                e += 1


        print(f'Total classes with missing label words are: {c}/{tc}, Total entities with missing label words are: {e}/{te}')
        ec = 0
        for idx, row in df_class.iterrows():
            node = self.class_dict[idx]
            if not node.all_words:
                ec += len(node.all_entities)
        print(f'These: {c} classes are responsible for: {ec} entities misclassification')

    def print_test_20(self):
        f = open('logs/test_20.txt', mode='w', encoding='utf8')

        print(f'################################ Printing 20 failed entities .##########################################\n\n\n')
        for _, node in self.class_dict.items():
            failed = set(node.all_entities) - set(node.predicted_entities) - set(node.seed_entities)
            failed = list(failed)
            f.write(f'Class: {node.raw_label}    ##############################\n')
            for entity in failed:
                score = self._cosine_similarity(node.Rc, entity.Le)
                label = entity.predicted_class
                if label is not None:
                    f.write(f'    {entity.raw_label} # predicted_class: {entity.predicted_class.raw_label} score(class:{score}, pred class: {entity.score})\n')
                # else: # if entity is non-seed for this class but seed for other class then it will be None.
                #     f.write(f'    {entity.raw_label} # predicted_class: {None} score{score, None}\n')
        f.close()
