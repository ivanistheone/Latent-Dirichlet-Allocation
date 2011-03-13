
import pylab as p
import numpy as np


def hist_of_topics_in_docs(th):
    bins = np.arange(-0.001,1.001,0.004)
    numDocs,numT = th.shape

    for t in range(0,numT):

        p.clf()

        n, bins, patches = p.hist(th[:,t], bins=bins)
        p.title("Topic %d" % t )
        p.ylim([0,200] )
        p.xlim([0,0.7] )

        bigcounts = [c for c in n if c>200]
        p.text(0, 200, ",".join( [str(c) for c in bigcounts]) )


        # try to classify
        t2  =   1.0/numT*40     # 0.2 in our case
        t3  =   1.5*t2          # 0.3

        low_bins = np.nonzero(    bins < t2)[0]
        high_bins= np.nonzero(t3 <bins     )[0]
        mid_bins = np.arange( low_bins[-1]+1,  high_bins[0] )


        p.plot([t2,t2], [0,200] )
        p.plot([t3,t3], [0,200] )

        totalc =  numDocs
        lowc  = np.sum( n[low_bins] )
        midc  = np.sum( n[mid_bins] )
        highc = totalc - lowc - midc    # != np.sum( n[high_bins] )
                                        #    since some p could be > histogram max bin
        p.text(0.3*t2, 170, "low=%d" % lowc )
        p.text(t2, 100, "mid=%d" % midc )
        p.text(2*t3, 15, "high=%d" % highc )

        garb = raw_input()


        p.clf()
