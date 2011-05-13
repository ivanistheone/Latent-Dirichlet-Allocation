#!/usr/bin/env python
import sys,os

posterior_dir = os.path.dirname(__file__)
posterior_dir = os.path.realpath( posterior_dir )
print posterior_dir
os.chdir(os.path.join(posterior_dir,"../../..") )
# ok now we are in the LatentDirichletAllocation root /
LDAdir = os.getcwd()


from liblda.low2corpus import Low2Corpus

execfile('mycmds.py')

testdir = os.path.realpath(os.path.join(LDAdir, "liblda/test/"))
# 1/10th of the quant-ph arXiv papers 2016 docs, vocab size of 10000
INFILE="arXiv_docs.txt"     # 2016 docs
VOCABFILE="arXiv_vocab.txt" # ~ 10 000
#arXiv_ids.txt
infilename = os.path.join(testdir, INFILE)
vfilename  = os.path.join(testdir, VOCABFILE)
tcorpus3 = Low2Corpus(infilename)
tcorpus3.setVocabFromList( [w.strip() for w in open(vfilename, 'r').readlines() ] )
tcorpus3.doCounts()


execfile('liblda/math/dirichlet_sparse_stats.py')
from liblda.LDAmodel import LdaModel


os.chdir(posterior_dir)

# T = 100
lda = LdaModel( numT=10, alpha=0.1, beta=0.01, corpus=tcorpus3)
watch_sparseness(lda, steps=[0,1,1,8,10,50,130], seed=7, filename='theta_and_phi_sparseness_for_alpha0.1beta0.01T10.png', initialize=True, pause=False )

lda = LdaModel( numT=10, alpha=0.01, beta=0.01, corpus=tcorpus3)
watch_sparseness(lda, steps=[0,1,1,8,10,50,130], seed=8, filename='theta_and_phi_sparseness_for_alpha0.01beta0.01T10.png', initialize=True, pause=False )


# T=400
#lda = LdaModel( numT=400, alpha=0.1, beta=0.01, corpus=tcorpus3)
#watch_sparseness(lda, steps=[0,1,1,8,10,50,130,200], seed=9, filename='theta_and_phi_sparseness_for_alpha0.1beta0.01T400.png', initialize=True, pause=False )


#lda = LdaModel( numT=400, alpha=0.01, beta=0.01, corpus=tcorpus3)
#watch_sparseness(lda, steps=[0,1,1,8,10,50,130,200], seed=2, filename='theta_and_phi_sparseness_for_alpha0.01beta0.01T400.png', initialize=True, pause=False )

