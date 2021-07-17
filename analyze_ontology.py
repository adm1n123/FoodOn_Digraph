# standard imports
import os
import sys

# local imports

from utils.utilities import load_pkl


class AnalyzeOntology:
    def __init__(self):
        self.gt_ontology = load_pkl('data/FoodOn/candidate_classes_dict.pkl')   # GT: ground truth, it is actual(not predicted) foodon ontology.

    def get_stats(self, predictedDF, allow_distance=0, match_only=None):    # predictedDF has predicted parent child pairs
        TP = 0
        FP = 0
        tp_list = []
        fp_list = []
        distance_distribution = []

        for idx, pair in predictedDF.iterrows():
            predicted_class = pair['Parent']    # predicted class of child.
            gt_classes = []                     # actual classes in which predicted child is present.
            for key, value in self.gt_ontology.items():
                if pair['Child'] in value[1]:
                    gt_classes.append(key)

            if match_only:
                if any(gt_class not in match_only for gt_class in gt_classes):
                    continue

            distance_list = []  # store the path distance.
            for gt_class in gt_classes:
                predicted_paths = [path[::-1] for path in self.gt_ontology[predicted_class][0]]
                gt_paths = [path[::-1] for path in self.gt_ontology[gt_class][0]]

                for pred_path in predicted_paths:
                    for gt_path in gt_paths:
                        common_path = set(pred_path).intersection(gt_path)
                        common_path = [c for c in gt_path if c in common_path]

                        distance = len(pred_path) + len(gt_path) - 2 * len(common_path)
                        distance_list.append(distance)

            idx_shortest = distance_list.index(min(distance_list))
            shortest_distance = distance_list[idx_shortest]
            distance_distribution.append(shortest_distance)

            if shortest_distance <= allow_distance: # assume it is mapped correctly take it TP: True Positive
                TP += 1
                tp_list.append((gt_classes, pair['Parent'], pair['Child']))
            else:   # else take it FP: False Positive.
                FP += 1
                fp_list.append((gt_classes, pair['Parent'], pair['Child']))

        return TP, FP, tp_list, fp_list, distance_distribution
