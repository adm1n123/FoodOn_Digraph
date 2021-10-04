import pandas as pd
import logging as log
import os
import random
import networkx as nx
import math
from utils.utilities import file_exists, save_pkl, load_pkl
from scoring import Scoring

class FoodOn:
    """
    TODO: classes at leaf without entities are treated as entities. since they are not parent of any. It does not affect
    TODO: precision because those classes treated as entities in both candidate and skeleton candidate dict. what is the
    TODO: use of leaf class without entity, It should not be created.
    """


    def __init__(self):
        self.csv_file = 'data/FoodOn/FOODON.csv'
        self.pairs_file = 'data/FoodOn/foodonpairs.txt' # save all pairs (parent, child) of FoodOn to here
        self.use_pairs_file = True  # don't generate again
        self.use_pkl = False  # if True, create and overwrite previous pickle file

        self.foodOn_root = 'http://purl.obolibrary.org/obo/FOODON_00001002' # root class we are working with.
        self.digraph_pkl = 'data/FoodOn/digraph.pkl'    # save digraph here
        self.digraph_seeded_pkl = 'data/FoodOn/digraph_seeded.pkl'  # save digraph with generated seeds

        self.num_seeds = 2  # minimum number of labeled data for each class
        self.num_min_seeds = 1 # if total entities are less than num_seeds take this as number of seeds.

        self.pd_foodon_pairs = self.generate_pairs()
        self.all_classes, self.all_entities = self.get_classes_and_entities()   # all_entities are instances(not classes)
        self.digraph_root, self.class_dict, self.entity_dict = self.generate_digraph()


        # self.candidate_ontology_pkl = 'data/FoodOn/candidate_classes_dict.pkl'  # save ground truth ontology (excluding classes without entities) dict to here
        # self.skeleton_and_entities_pkl = 'data/FoodOn/skeleton_candidate_classes_dict.pkl'  # save skeleton ontology dict and entities to populate to here



    def generate_pairs(self):   # tested
        print('generating child parent pairs')
        if os.path.isfile(self.pairs_file) and self.use_pairs_file:
            print('Using pre-generated pairs file.')
            return pd.read_csv(self.pairs_file, sep='\t')

        # read foodon csv file
        foodon = pd.read_csv(self.csv_file, usecols=['Class ID', 'Parents', 'Preferred Label'])
        temp = foodon[['Class ID', 'Preferred Label']].copy()
        labels = temp.set_index('Class ID')['Preferred Label'].to_dict()
        # take child and parents classes
        child_parents = (foodon[['Class ID', 'Parents']].copy()).rename(columns={'Class ID': 'Child'})
        # split parent classes if there are more than one parent to child.

        pairs = []
        for _, row in child_parents.iterrows():
            parents = str(row['Parents'])
            parent_classes = parents.split('|')
            for parent in parent_classes:
                pairs.append([str(row['Child']), parent])

        # take child and parent class there are no duplicates (child, parent) pairs.
        foodonDF = pd.DataFrame(pairs, columns=['Child', 'Parent']) # multiple parents are split over rows
        foodonDF = self.filter_ontology(foodonDF, 'http://purl.obolibrary.org/obo/FOODON_00001872') # this class is under progress so don't include it.

        # after getting subtree there are 14k duplicates (child, parent) pairs out of 30k so total 14k approx pairs remaining.
        foodonDF = self.get_subtree(foodonDF, 'http://purl.obolibrary.org/obo/FOODON_00001002')     # take subtree of FOODON_00001002 class this is 'foodon product type' because we are working with this subtree only.

        foodonDF.drop_duplicates(inplace=True, ignore_index=True)  # there are only 5 duplicates where (child, parent) Id pair was different but pair name was same.

        foodonDF[['Child_label', 'Parent_label']] = None    # add new columns for label.
        for _, row in foodonDF.iterrows():
            row['Child_label'] = labels[row['Child']]
            row['Parent_label'] = labels[row['Parent']]   # if error: item not found in dict use if condition and don't label parent leave class id.

        foodonDF.to_csv(self.pairs_file, sep='\t', index=False)
        return foodonDF


    def filter_ontology(self, df, classname):
        # Remove class and its children from the ontology. Works only if the children are leaf nodes. because grandchild will not be removed. but finding subtree after it will not include grandchild
        indexes = df[df['Parent'] == classname].index
        df.drop(indexes, inplace=True)
        indexes = df[df['Child'] == classname].index
        df.drop(indexes, inplace=True)
        return df

    def get_subtree(self, df, root):
        # return the all the classes in subtree of root
        subtreeDF, nextlevelclasses = self.get_level_classes(df, [root])
        # subtreeDF.drop_duplicates(inplace=True, ignore_index=True)
        # nextlevelclasses = list(set(nextlevelclasses))

        while len(nextlevelclasses) > 0:
            level_pairs, nextlevelclasses = self.get_level_classes(df, nextlevelclasses)
            subtreeDF = pd.concat([subtreeDF, level_pairs], ignore_index=True)
            # subtreeDF.drop_duplicates(inplace=True, ignore_index=True)
            # nextlevelclasses = list(set(nextlevelclasses))
        return subtreeDF

    def get_level_classes(self, df, parent_classes):
        # return the parent child pair of all the parent classes
        pairs = []
        non_leaf_children = []
        for parent in parent_classes:
            selected_pairs = df[df['Parent'] == parent]
            for _, row in selected_pairs.iterrows():
                pairs.append([row['Child'], row['Parent']])
                next_level = df[df['Parent'] == row['Child']]
                if not next_level.empty:
                    non_leaf_children.append(row['Child'])

        level_pairs = pd.DataFrame(pairs, columns=['Child', 'Parent'])
        return level_pairs, non_leaf_children

    def get_classes_and_entities(self):
        classes = self.pd_foodon_pairs['Parent'].tolist()   # every non-leaf is a class.
        classes = list(set(classes))
        classes.sort()
        print('Found %d classes.' % len(classes))

        child = self.pd_foodon_pairs['Child'].tolist()
        child = list(set(child))
        child.sort()
        entities = [c for c in child if c not in classes]   # child which is also parent is not leaf(instance)
        print('Found %d entities.' % len(entities))
        return classes, entities

    def generate_digraph(self):
        print('Generating Digraph')
        if os.path.isfile(self.digraph_pkl) and self.use_pkl:
            print('Using pre-generated digraph file: %s', self.digraph_pkl)
            return load_pkl(self.digraph_pkl)

        class_dict = {class_id: Class(class_id) for _, class_id in enumerate(self.all_classes)}
        entity_dict = {entity_id: Entity(entity_id) for _, entity_id in enumerate(self.all_entities)}

        for _, row in self.pd_foodon_pairs.iterrows():
            if row['Parent'] == row['Child']:
                print(row['Parent'], 'parent is child of itself')
                exit(0)
            if row['Parent'] in self.all_classes and row['Child'] in self.all_classes:  # both are classes
                parent = class_dict[row['Parent']]
                child = class_dict[row['Child']]
                parent.raw_label = row['Parent_label']
                child.raw_label = row['Child_label']
                parent.children.append(child)
                child.parents.append(parent)
            elif row['Parent'] in self.all_classes and row['Child'] in self.all_entities:
                parent = class_dict[row['Parent']]
                child = entity_dict[row['Child']]
                parent.raw_label = row['Parent_label']
                child.raw_label = row['Child_label']
                parent.all_entities.append(child)
                child.parents.append(parent)

        digraph = (class_dict[self.foodOn_root], class_dict, entity_dict)
        # save_pkl(digraph, self.digraph_pkl)
        return digraph


    def seed_digraph(self):
        print('Seeding digraph.')
        if file_exists(self.digraph_seeded_pkl) and self.use_pkl:
            print('Using pickled seeded digraph file: %s', self.digraph_seeded_pkl)
            return load_pkl(self.digraph_seeded_pkl)

        seeds = set()
        count = 0
        for _, node in self.class_dict.items():
            if len(node.all_entities) > self.num_seeds:
                node.seed_entities = random.sample(node.all_entities, self.num_seeds)
            elif len(node.all_entities) > self.num_min_seeds:
                node.seed_entities = random.sample(node.all_entities, self.num_min_seeds)
            else:
                node.seed_entities = node.all_entities.copy()
            seeds = seeds.union(set(node.seed_entities))

            if len(node.all_entities) == 0:
                count += 1

        non_seeds = set(self.entity_dict.values()) - seeds

        print(f'Classes without entities: {count}, classes with entities: {len(self.all_classes)-count}')
        digraph_seeded = (self.class_dict, list(non_seeds))
        # print('Saving seeded digraph to file: %s', self.digraph_seeded_pkl)
        # save_pkl(digraph_seeded, self.digraph_seeded_pkl)
        print('seeds %d, Found %d non-seed entities to populate out of %d all entities.' % (len(seeds), len(non_seeds), len(self.all_entities)))
        return digraph_seeded

    def seed_digraph2(self):
        print('Seeding digraph.')
        if file_exists(self.digraph_seeded_pkl) and self.use_pkl:
            print('Using pickled seeded digraph file: %s', self.digraph_seeded_pkl)
            return load_pkl(self.digraph_seeded_pkl)

        seeds = set()
        count = 0
        # min_non_seeds = 1
        threshold, fraction = 1, .8 # if entities are more than 5 take 80% as seeds.
        for _, node in self.class_dict.items():
            if len(node.all_entities) > threshold:
                node.seed_entities = random.sample(node.all_entities, math.floor(len(node.all_entities)*fraction))
            # elif len(node.all_entities) > min_non_seeds:
            #     node.seed_entities = random.sample(node.all_entities, len(node.all_entities)-min_non_seeds)
            else:
                node.seed_entities = node.all_entities.copy()

            seeds = seeds.union(set(node.seed_entities))

            if len(node.all_entities) == 0:
                count += 1

        non_seeds = set(self.entity_dict.values()) - seeds

        print(f'Classes without entities: {count}, classes with entities: {len(self.all_classes)-count}')
        digraph_seeded = (self.class_dict, list(non_seeds))
        # print('Saving seeded digraph to file: %s', self.digraph_seeded_pkl)
        # save_pkl(digraph_seeded, self.digraph_seeded_pkl)
        print('seeds %d, Found %d non-seed entities to populate out of %d all entities.' % (len(seeds), len(non_seeds), len(self.all_entities)))
        return digraph_seeded


    def populate_foodon_digraph(self):

        class_dict, non_seed_entities = self.seed_digraph2()

        scoring = Scoring(
            root=self.digraph_root,
            class_dict=class_dict,
            entity_dict=self.entity_dict,
            non_seeds=non_seed_entities)    # entity assumed to belongs to one class.

        scoring.run_config()
        # scoring.run_analysis()
        scoring.bad_precision()
        scoring.find_worst_classes()
        # scoring.find_diff_old_rmv_method()

        return

"""
All the parents of class/entity are considered.
"""
class Class:
    def __init__(self, ID):
        self.ID = ID  # class id
        self.raw_label = None
        self.label = None   # class label processed
        self.Lc = None
        self.Rc = None
        self.Sc = None
        # self.score = None # similarity with this class (used from child to set self.can_back = false.)
        self.seed_entities = [] # seed entities for this class
        self.predicted_entities = []    # predicted entities for this class
        self.all_entities = []  # all entities in this class
        self.children = []  # all child subclasses(not entities)
        self.parents = []    # all parents
        self.visited = 0
        self.Rc_sum = None  # sum of all Rc vectors of subclasses with non-zero entities.
        self.Rc_count = 0   # count of all the classes in subtree with non-zero entities.
        self.in_path = False
        self.pre_proc = False   # vector Rc, Sc computed.
        self.visited_for = None


class Entity:
    def __init__(self, ID):
        self.ID = ID  # class id
        self.raw_label = None
        self.label = None   # entity label processed
        self.Le = None
        self.score = None   # list of all the scores during traversal.
        self.predicted_class = None
        self.parents = []  # all parent classes
        self.visited_classes = 0



