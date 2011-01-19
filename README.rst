

Latent Dirichlet Allocation library
===================================

Yet another one.. I know...
But it seems it is necessary for my needs.
Luckily I can stand on the shoulders of talanted hackers
that came before me David Newman and Radim Řehůřek.



Example
-------

Here is a small example of how to use liblda

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



Yeah more info as we I know...



Current status
--------------
At this point we are calling out to Dave Newman's gibbs
sampler. We can do 20 topics for 6M very short documents 
over a 1M vocabulary.



