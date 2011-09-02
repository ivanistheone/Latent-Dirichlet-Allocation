


# Phi based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix, find_closest
# Theta  based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix2, find_closest2





# data exploration, plotting  and reporting
from liblda.topicviz.show_top import show_top
from liblda.topicviz.show_top import top_words_for_topic




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








#http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means
# copied to kmeans.py


