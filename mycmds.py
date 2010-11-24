# things i need to run each time I stary ipython
import numpy, scipy
import os,sys

# INSTALLED APPS ;)
####################################

# original gensim
sys.path.insert(1, '/Projects/LatentDirichletAllocation/gensim/trunk/src')
from gensim import corpora, models, similarities

# ldalib
sys.path.insert(1, '/Projects/LatentDirichletAllocation/')
import liblda


# settings file with RUNDIRS path, topicmodel location and PROJECT_HOME
# from liblda import settings ?
from liblda import local_settings


# to see logging...
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)



