
from collections import defaultdict
import os,sys
import numpy as np





from scikits.learn.linear_model import  LogisticRegression



nips_dir = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
catsf = open( os.path.join(nips_dir, 'NIPS_categories.txt'), 'r' )
categories = np.array( [ int( cid.strip() ) for cid in catsf.readlines() ] )
catsf.close()




def random_split(theta, categories, size=10):
    

    CATEGORIES = range(1,9) # 1... 8


    # FROM NOW ON evething is in terms of indices into categories/theta
    # idx=0  is document0
    # idx2 = labeled[0] is the indx of the first labeled doc
    labeled = [ w[0] for w in enumerate(categories) if w[1] in CATEGORIES ]
    
    labeled_for = defaultdict(list)
    label_count = np.zeros(9, dtype=np.int32)
    # 0 category is uncategorized in there...
    for d_id in labeled:
        labeled_for[categories[d_id]].append(d_id)
        label_count[categories[d_id]] += 1

    permuted_for = {}
    for cat_id in CATEGORIES:
        permuted_for[cat_id] = np.random.permutation(  labeled_for[cat_id] )

    #split
    train_ids= []
    test_ids = []
    for cat_id in CATEGORIES:
        test_ids.extend( permuted_for[cat_id][0:size] )
        train_ids.extend( permuted_for[cat_id][size:] )
        

    #print train_ids
    #print test_ids

    # slice theta and categories as necessary
    trainX = theta[ train_ids, : ]
    trainY = categories[ train_ids, : ]
    testX = theta[ test_ids, : ]
    testY = categories[ test_ids, : ]

    return ( (trainX,trainY,train_ids), (testX, testY,test_ids), )






def load_cat_labels(dir):
    """ Reads the files
            NIPS_category_label.txt and 
            NIPS_labels.txt
        produces the file
            NIPS_categories.txt
    """
    
    # map
    mapf = open( os.path.join(dir, 'NIPS_category_label.txt') ) 
    label_id=0
    cat2label = defaultdict(list)
    label2cat = {}
    for line in mapf.readlines():
        cat_id, label_name  = line.split(" ", 1)
        label2cat[int(label_id)] = int(cat_id)
        cat2label[int(cat_id)].append( int(label_id) )
        label_id += 1


    # names of things
    lnamesf = open( os.path.join(dir, 'NIPS_label_names.txt') )
    labels = dict(enumerate( [w.strip() for w in lnamesf.readlines() ] ))
    cnamesf = open( os.path.join(dir, 'NIPS_category_names.txt') )
    categories = dict(enumerate( [w.strip().split(" ",1)[1] for w in cnamesf.readlines() ] ))
    categories[99]=categories[9]
    del(categories[9])

    

    for cat in [0,1,2,3,4,5,6,7,8, 99]:
        print cat, categories[cat]
        for l in cat2label[cat]:
            print "    ", labels[l]


    # IN labels file
    labelsf = open( os.path.join(dir, 'NIPS_labels.txt') ) 

    # OUT cats file
    catsf = open( os.path.join(dir, 'NIPS_categories.txt'), 'w' ) 
    
    for line in labelsf.readlines():

        label_id = int(line.strip())
        cat_id = label2cat[label_id]
        catsf.write( "%d\n"% cat_id )


    print "FIle NIPS_categories.txt written"
    print "done."



