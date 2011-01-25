

Latent Dirichlet Allocation library
===================================

Yet another one.. I know...
But it seems it is necessary for my needs.
Luckily I can stand on the shoulders of talanted hackers
that came before me David Newman and Radim Řehůřek.



Example
-------

Here is a small example of how to use liblda

  * Prepare the corpus of documents
  * Prepare the vocabulary
  * Run the lda model

::
  ./run.py \
  --docs liblda/test/arXiv_docs.txt \
  --vocab liblda/test/arXiv_vocab.txt \
  --numT 20 \
  --seed 3 \
  --iter 300 \
  --alpha 0.01 \
  --beta 0.01 \
  --save_z --save_probs --print_topics 12



Yeah more info as we I know...


You can also use the LdaModel class in your own code.

::
    import sys
    sys.path.insert(1, '/Projects/LatentDirichletAllocation/')  
    from gensim import corpora, models, similarities
    import liblda
    from liblda.LDAmodel import LdaModel

    c = corpora.mmcorpus.MmCorpus("liblda/test/test_corpus.mm")
    lda = LdaModel( numT=3, corpus=c)
    lda.train()

    # results in 
    lda.phi
    lda.theta




Current status
--------------
At this point we are calling out to Dave Newman's gibbs
sampler. We can do 20 topics for 6M very short documents 
over a 1M vocabulary.



