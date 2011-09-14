

import numpy as np

# Phi based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix, find_closest
# Theta  based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix2, find_closest2





# data exploration, plotting  and reporting
from liblda.topicviz.show_top import show_top
from liblda.topicviz.show_top import top_words_for_topic



#### Fri  2 Sep 2011 15:07:15 EDT
#
# Ok here's the plan. We run a kmeans clustergin on the combined Phi's
# of the three /runs/repeatedN 


#%run loadNIPS.py
ph1 = np.load('/Projects/runs/reduced1/phi.npy')
ph2 = np.load('/Projects/runs/reduced2/phi.npy')
ph3 = np.load('/Projects/runs/reduced3/phi.npy')

ph = np.vstack((ph1,ph2,ph3))


from kmeans import Kmeans
km= Kmeans(ph, k=40, nsample=20, centres=None, metric='euclidean')



model_ids = np.concatenate( 
                (1*np.ones(ph1.shape[0]),  2*np.ones(ph2.shape[0]), 3*np.ones(ph3.shape[0]))  )
model_lookup = {1:ph1, 2:ph2, 3:ph3}

def print_clusters(km, id2word):
    for jc,jids in km:
        print ""
        print "==== Cluster", jc, "  ====="
        for tid in np.nonzero(jids)[0]:
            model_id = model_ids[tid]
            realt = tid - 40*(model_id-1)
            print tid,"(%d)"%model_id, top_words_for_topic(model_lookup[model_id], realt, num=10, id2word=id2word)





def similar_topic(mstar, t):
    m1 = getCostMatrix(mstar.phi, mstar.phi)
    m2 = getCostMatrix2(mstar.theta, mstar.theta)

    ph_similar = sorted([ k for k in enumerate(m1[t])], key=lambda k: k[1] )[1:6]
    th_similar = sorted([ k for k in enumerate(m2[t])], key=lambda k: k[1] )[1:6]
    
    print "Phi similar topics:"
    for ii, d_t_ii in ph_similar:
        print ii, d_t_ii, ",".join( top_words_for_topic(mstar.phi, ii, num=9, id2word=mstar.corpus.id2word) )

    print "Theta similar topics:"
    for ii, d_t_ii in th_similar:
        print ii, d_t_ii, ",".join( top_words_for_topic(mstar.phi, ii, num=9, id2word=mstar.corpus.id2word) )




import pylab as pl
from itertools import cycle
def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
    pl.legend()
    pl.show()


import mpl_toolkits.mplot3d.axes3d as p3

def plot_3D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    for i, c, label in zip(target_ids, colors, target_names):
        ax.scatter3D(data[target == i, 0], data[target == i, 1],data[target == i, 2], 
            c=c, label=label)
    pl.legend()
    pl.show()



# visualize topics via PCA to 2D
from scikits.learn.decomposition import PCA
#pca = PCA(n_components=2, whiten=True).fit(X)




#http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means
# copied to kmeans.py




