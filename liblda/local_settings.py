import os

# this is where we currently run things from...
# /Projects/LatentDirichletAllocation
PROJECT_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),".."))

# this is the location where we downloaded Dave Newman's code
# and ran make clean; make all in that dir

# in the local case
# /Projects/LatentDirichletAllocation/topicmodel
topicmodel_DIR=os.path.realpath(os.path.join(os.path.dirname(__file__),"../topicmodel/"))

# Location which we use to store run data
#
RUNDIRS_ROOT = os.path.join(PROJECT_PATH, "data/runs/")

