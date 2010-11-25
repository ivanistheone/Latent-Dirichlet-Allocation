
import operator

def show_top(phi, num=20, corpus=None):

    numT,numTerms = phi.shape

    topics = []

    for t in range(numT):

        pw_gt = phi[t,:]
        topwords = sorted(enumerate(pw_gt), key=operator.itemgetter(1), reverse=True)
        words = [corpus.id2word[id] for id,prb in topwords[0:num] ]
        topics.append(words)

    return topics



