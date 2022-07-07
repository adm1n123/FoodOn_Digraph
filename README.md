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


Stats:  (alpha .4, beta .6), score = [α * Lc + (1-α) * seeds,  β * Sc + (1-β) * children(Sc avg)]

Ra = La  + Seed_a
Sa = Sb + Sc
Rs_a = Ra + Rb + Rc + Rd + Re + Rf + Rg
Sd = Rd



==> Considering all children of root(foodon product type)   
Predicted   Traversed   Avg node(visited)   bias    Precision
Rc          Rc_sum      30                  0       20%
Rc          Rc_sum      154                 .05     32.9%
Rc          Rc_sum      310                 .08     33.4
Rc          Rc_sum      251                 .08     34.2
Rc          Sc          537                 .05     31%
Rc          Sc          1127                .1      33%
Rc          Sc          188                 0       27%
Rc          -           all                         35.6%

Rc = 

==> Not considering all children of root(foodon product type)
Predicted   Traversed   Avg node(visited)   bias    Precision
Rc          Sc          100                 0       20% (sometimes 16% due to random seeds)
Rc          Sc(Rc_sum)                      0       14%




Traversing all node(brute force) (alpha .4, beta .6)
precision 35%

Before bug fixes(some seed were also in non-seeds) (alpha .4, beta .6)
precision 38%




















