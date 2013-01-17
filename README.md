
Latent Dirichlet Allocation 
================================

A library for the LDA topic modelling algorithm in Python and C.

<pre>
 __    ____  _____ 
|  |  |    \|  _  |
|  |__|  |  |     |
|_____|____/|__|__|
</pre>


Usage
-----

The best way to use `liblda` is through the command line: 

   ./run.py --docs docs.txt --numT 40 --vocab vocab.txt --seed 3 --iter 400 --alpha 0.1 --beta 0.01 --save_probs --print_topics 10

where: 
  * `docs.txt` contains one document per line,
  * `vocab.txt` contains the vocabulary (one word per line)
  * `--save_probs` indicates that you want to output the probs phi and theta



Installation
------------

Place the directory `liblda` somewhere in your Python path.


Features
--------

We have implemented the Gibbs sampling approach which is fairly efficient when done in C.
All the rest of the functionality is done in Python so it is very hackable.


Requirements
------------
  * `numpy` (for arrays)
  * `scipy` (for weave)


Project status
--------------

The code base works, but is a bit of a mess right now.
A rewrite has begun -- in cython.


Author
--------

Ivan Savov, first dot last at gmail



