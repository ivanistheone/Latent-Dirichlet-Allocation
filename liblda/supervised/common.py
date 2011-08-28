
from collections import defaultdict
import os,sys
import numpy as np





from scikits.learn.linear_model import  LogisticRegression
from scikits.learn import metrics



nips_dir = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
catsf = open( os.path.join(nips_dir, 'NIPS_categories.txt'), 'r' )
categories = np.array( [ int( cid.strip() ) for cid in catsf.readlines() ] )
catsf.close()




def reg_param_search( th1000, categories, dir=None):

    if not dir:
        dir = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
    lnamesf = open( os.path.join(dir, 'NIPS_label_names.txt') )
    label_names = dict(enumerate( [w.strip() for w in lnamesf.readlines() ] ))
    cnamesf = open( os.path.join(dir, 'NIPS_category_names.txt') )
    category_names = dict(enumerate( [w.strip().split(" ",1)[1] for w in cnamesf.readlines() ] ))
    category_names[99]=category_names[9]
    del(category_names[9])

    labelsf = open( os.path.join(dir, 'NIPS_labels.txt') )
    labels  = [ int(w.strip()) for w in labelsf.readlines() ]


    ((Trx,Try,train_ids), (tex,tey,test_ids) ) = random_split( th1000, categories )

    Cs =  [1, 2, 3, 5, 10 ] #, 23, 10, 20, 30, 40]
    #Cs = [30]
    #np.arange(0.1,120,5)
    Cscore = np.zeros( len(Cs) )

    for Cid in range(0,len(Cs) ):

        cp = CategoryPredictor(Trx, Try, reg_param=Cs[Cid])

        score=0
        item_id=0
        for d in range(0,len(tey)):
            doc_id = test_ids[item_id]
            #print cp.classify(tex[d,:][np.newaxis]), tey[d]
            if cp.classify(tex[d,:][np.newaxis])==tey[d]:
                score+=1
            else:
                print "CLASSIF ERR: ", \
                        "predicted ", cp.classify(tex[d,:][np.newaxis]),  \
                        ",      true cat=", tey[d], ", label=", label_names[labels[doc_id]]
            item_id += 1


        Cscore[Cid] = score

    return (Cs, Cscore)




def evaluate(model, testX, testY, testTitles=None):
    """ Shows all the performance of `model` at predicting
        the testY from the testX
    """

    dir = '/CurrentPorjects/LatentDirichletAllocation/data/NIPS1-17/'
    cnamesf = open( os.path.join(dir, 'NIPS_category_names.txt') )
    names_of_categories = dict(enumerate( [w.strip().split(" ",1)[1] for w in cnamesf.readlines() ] ))
    cnamesf.close()
    n_of_cats=names_of_categories


    predicted = model.predict(testX)

    print metrics.confusion_matrix(testY, predicted)
    print metrics.classification_report(testY, predicted, target_names=names_of_categories.values()[1:9])

    size = len(testY)

    if testTitles is not None:
        print ("_"*80),                "_________", "_________"
        print "Paper title".ljust(80), "predicted", "true     "
        print ("_"*80),                "_________", "_________"
        for i in range(0, size):
            pred = predicted[i]
            true = testY[i]
            if pred != true:
                print testTitles[i][0:80].ljust(80), n_of_cats[pred][0:9].ljust(9), n_of_cats[true][0:9].ljust(9)



            





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



