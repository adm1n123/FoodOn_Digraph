1.
Install packages
gensim
numpy
pandas
matplotlib
scikit-learn
scipy
argparse
configparser
cython
pattern
wikipedia
networkx
textdistance


2.
run main.py


similarity of entity with all sibling is taken and also similarity with class label is taken 
then score = alpha * sibling + (1-alpha) * class label

Noun words is given 15% more weightage than non-noun words in class/instance label.

TODO:
wiki corpus has only 4349 sentences to train collect more (try firing failed queries again). Use wordnet for synonyms
and use all words+synonyms to query wikipedia. use stack overflow API and twitter API to query food label

get_seeded_skeleton() this method is generating skeleton graph and some entities are taken as seed and some are kept
as candidate entities(which has to be populated/mapped to class).
there are some entities which are in more than one class but in this method only unique entities are taken i.e. author
is assuming an entity belongs to one class only(in paper) 
[ improvement could be store the count of classes entity belongs to and then during population assign this entity to 
top k classes(by matching). This is not improvement because you don't know when you are adding new entity in ontology 
that is why method is called automated, or for every entity map it to one or more classes i.e. don't use information
from ontology that an entity belongs to more than one classes. first analyse entities belonging to more than one class
and assign if similarity with top 2 classes are same]

If an entity is belonging to more than one class then it might be ambiguous so assigning it to more than one class
might inccur more loss.


change similarity_method= use weighted similarity method (i.e. word embeddings with text distance etc.)


we are not using phrases. all the labels are space separated and average of word vector is taken as label vector.
foodonpairs.txt columns are not preprocessed and phrases generation is also skipped.


calculate_parents_score  here class label is used for similarity calculation between entity and class(not sibling) 
entities similarity is taken with siblings of class and with class label.
siblings vector is average of (seeds vectors and seeds vector is avg of words vectors in seed).
class vector is average of label word vectors.
Parent class means class in which entity is categorized because entity itself is called class hence parent class is 
class in which entity and siblings are present.


Since parent class vector(class label vectors) are precomputed
and for similarity purpose take array of class vectors and array of entity vectors then do multiplication to get
result fast due to vectorization.

for sibling vectors store the sum of sibling vectors and when new sibling added just add vector 
then take array of sibling vectors and array of entities do vector multiplication.


Don't fix the seed number of seed instances take it as 5-10% percent etc.


sometimes result defer by 4% because of random seeds.(bad seed selection cause this may be for any particular class)


Stats:  (alpha .4, beta .6), score = [α * Lc + (1-α) * seeds,  β * Sc + (1-β) * children(Sc avg)]

Considering all children of root(foodon product type)   
Predicted   Traversed   Avg node(visited)   bias    Precision
Rc          Sc(Rc_sum)  30                  0       20%
Rc          Sc          537                 .05     31%
Rc          Sc          1127                .1      33%
Rc          Sc          188                 0       27%
Rc          Sc(Rc_sum)  154                 .05     32.9%
Rc          Sc(Rc_sum)  310                 .08     33.4


Not considering all children of root(foodon product type)
Predicted   Traversed   Avg node(visited)   bias    Precision
Rc          Sc          100                 0       20% (sometimes 16% due to random seeds)
Rc          Sc(Rc_sum)                      0       14%




Traversing all node(brute force) (alpha .4, beta .6)
precision 35%

Before bug fixes(some seed were also in non-seeds) (alpha .4, beta .6)
precision 38%




















