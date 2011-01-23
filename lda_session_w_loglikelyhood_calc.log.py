#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True, 'logfile': 'lda_session_w_loglikelyhood_calc.log.py'})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
_ip.magic("run run.py")
_ip.magic("run run.py")
_ip.magic("clear ")
_ip.magic("run run.py")
_ip.magic("run run.py")
_ip.magic("run run.py")
_ip.magic("run run.py --numT 3")
c = corpora.mmcorpus.MmCorpus("liblda/test/test_corpus.mm")
c
#[Out]# <gensim.corpora.mmcorpus.MmCorpus object at 0x102823c10>
list(c)
#[Out]# [[(0, 1.0), (1, 1.0), (2, 1.0)], [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)], [(2, 1.0), (5, 1.0), (7, 1.0), (8, 1.0)], [(1, 1.0), (5, 2.0), (8, 1.0)], [(3, 1.0), (6, 1.0), (7, 1.0)], [(9, 1.0)], [(9, 1.0), (10, 1.0)], [(9, 1.0), (10, 1.0), (11, 1.0)], [(4, 1.0), (10, 1.0), (11, 1.0)]]
c[:][]
c[:]
_ip.magic("clear ")
_ip.system("ls -F ")
_ip.magic("pwd ")
#[Out]# '/Users/ivan/Homes/master/Documents/Projects/LatentDirichletAllocation'
_ip.magic("run mycmds.py")
from liblda.LDAmodel import LdaModel
_ip.magic("run liblda/test/test_LDAmodel.py")
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
tcorpus
#[Out]# <gensim.corpora.mmcorpus.MmCorpus object at 0x1029186d0>
tcorpus3
#[Out]# <liblda.low2corpus.Low2Corpus object at 0x1029187d0>
lda = LdaModel( numT=20, alpha=0.1, beta=0.1, corpus=tcorpus3 )
tcorpus3.numTerms
#[Out]# 10011
tcorpus3.numDocs
#[Out]# 2016
lda.train(iter=10, seed=7)
lda.phi
#[Out]# array([[  9.85691241e-03,   8.74656946e-03,   1.10510548e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  1.13991153e-02,   6.45613610e-03,   1.11973611e-03, ...,
#[Out]#           2.01754253e-05,   1.51315690e-05,   0.00000000e+00],
#[Out]#        [  8.11648035e-03,   3.93418917e-03,   6.39394348e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#[Out]#        ..., 
#[Out]#        [  2.26204834e-02,   1.31288004e-02,   1.25069099e-02, ...,
#[Out]#           1.88451681e-05,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  5.48189993e-02,   2.85911921e-02,   1.36254900e-02, ...,
#[Out]#           2.16163246e-05,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  3.12442553e-02,   1.62791489e-02,   4.24170213e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
def loglike(self):
lda.phi
#[Out]# array([[  9.85691241e-03,   8.74656946e-03,   1.10510548e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  1.13991153e-02,   6.45613610e-03,   1.11973611e-03, ...,
#[Out]#           2.01754253e-05,   1.51315690e-05,   0.00000000e+00],
#[Out]#        [  8.11648035e-03,   3.93418917e-03,   6.39394348e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#[Out]#        ..., 
#[Out]#        [  2.26204834e-02,   1.31288004e-02,   1.25069099e-02, ...,
#[Out]#           1.88451681e-05,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  5.48189993e-02,   2.85911921e-02,   1.36254900e-02, ...,
#[Out]#           2.16163246e-05,   0.00000000e+00,   0.00000000e+00],
#[Out]#        [  3.12442553e-02,   1.62791489e-02,   4.24170213e-03, ...,
#[Out]#           0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
lda.phi.shape
#[Out]# (20, 10011)
lda.theta.shape
#[Out]# (20, 2016)
lda.phi[:][lda.w[2]]
#[Out]# array([ 0.00985691,  0.00874657,  0.00110511, ...,  0.        ,
#[Out]#         0.        ,  0.        ])
np.dot( lda.phi[:][lda.w[2]], lda.theta[:][lda.d[2]] )
#np.dot( lda.phi[:][lda.w[2]], lda.theta[:][lda.d[2]] )
#?np.inner
s("")
_ip.system("ls -F ")
np.inner( lda.phi[:][lda.w[2]], lda.theta[:][lda.d[2]] )
lda.d[2]
#[Out]# 0
 lda.theta[:][lda.d[2]]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[:][lda.d[2]]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[:][lda.d[2]].shape
#[Out]# (2016,)
lda.theta[lda.d[2]][:]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[lda.d[2]][:].shape
#[Out]# (2016,)
lda.theta
#[Out]# array([[ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#          0.008006  ,  0.02572097],
#[Out]#        [ 0.08142629,  0.004914  ,  0.1575    , ...,  0.02162162,
#[Out]#          0.00850638,  0.04208885],
#[Out]#        [ 0.        ,  0.11302211,  0.05333333, ...,  0.05540541,
#[Out]#          0.02076557,  0.02494154],
#[Out]#        ..., 
#[Out]#        [ 0.01117616,  0.        ,  0.00805556, ...,  0.0027027 ,
#[Out]#          0.01225919,  0.0568979 ],
#[Out]#        [ 0.01703034,  0.02211302,  0.04111111, ...,  0.00405405,
#[Out]#          0.0445334 ,  0.00857366],
#[Out]#        [ 0.03725386,  0.        ,  0.08111111, ...,  0.00810811,
#[Out]#          0.11208406,  0.00077942]])
lda.theta.shape
#[Out]# (20, 2016)
lda.theta[:][0]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[:][0].shape
#[Out]# (2016,)
lda.theta[:][0]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[0][:]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[0][:].shape
#[Out]# (2016,)
lda.theta[0][:]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[:][0]
#[Out]# array([ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#         0.008006  ,  0.02572097])
lda.theta[:][:]
#[Out]# array([[ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#          0.008006  ,  0.02572097],
#[Out]#        [ 0.08142629,  0.004914  ,  0.1575    , ...,  0.02162162,
#[Out]#          0.00850638,  0.04208885],
#[Out]#        [ 0.        ,  0.11302211,  0.05333333, ...,  0.05540541,
#[Out]#          0.02076557,  0.02494154],
#[Out]#        ..., 
#[Out]#        [ 0.01117616,  0.        ,  0.00805556, ...,  0.0027027 ,
#[Out]#          0.01225919,  0.0568979 ],
#[Out]#        [ 0.01703034,  0.02211302,  0.04111111, ...,  0.00405405,
#[Out]#          0.0445334 ,  0.00857366],
#[Out]#        [ 0.03725386,  0.        ,  0.08111111, ...,  0.00810811,
#[Out]#          0.11208406,  0.00077942]])
lda.theta[:][:].shape
#[Out]# (20, 2016)
lda.theta
#[Out]# array([[ 0.3943587 ,  0.05159705,  0.07277778, ...,  0.01351351,
#[Out]#          0.008006  ,  0.02572097],
#[Out]#        [ 0.08142629,  0.004914  ,  0.1575    , ...,  0.02162162,
#[Out]#          0.00850638,  0.04208885],
#[Out]#        [ 0.        ,  0.11302211,  0.05333333, ...,  0.05540541,
#[Out]#          0.02076557,  0.02494154],
#[Out]#        ..., 
#[Out]#        [ 0.01117616,  0.        ,  0.00805556, ...,  0.0027027 ,
#[Out]#          0.01225919,  0.0568979 ],
#[Out]#        [ 0.01703034,  0.02211302,  0.04111111, ...,  0.00405405,
#[Out]#          0.0445334 ,  0.00857366],
#[Out]#        [ 0.03725386,  0.        ,  0.08111111, ...,  0.00810811,
#[Out]#          0.11208406,  0.00077942]])
lda.phi[:][lda.w[2]]
#[Out]# array([ 0.00985691,  0.00874657,  0.00110511, ...,  0.        ,
#[Out]#         0.        ,  0.        ])
lda.phi[:][lda.w[2]].shape
#[Out]# (10011,)
lda.phi[:,lda.w[2]].shape
#[Out]# (20,)
np.inner( lda.phi[:,lda.w[2]], lda.theta[:,lda.d[2] )
np.inner( lda.phi[:,lda.w[2]], lda.theta[:,lda.d[2]] )
#[Out]# 0.011079595485477016
#np.inner( lda.phi[:,lda.w[2]], lda.theta[:,lda.d[2]] )
sum=0
i=0
np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] )
#[Out]# 0.011079595485477016
lda.corpus.totalNwords
#[Out]# 3813938L
#for i in range(0,lda.corpus.totalNwords
for i in range(0,lda.corpus.totalNwords):
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] )
sun
sum
#[Out]# 7849.5301507637487
# that is after 10
# this was after 10 iterations of gibbs sampling
lda.train(iter=50, seed=7)
sum=0.0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
sum
#[Out]# -27396870.707167484
ll=dict()
ll[50]=sum
lda.gibbs_sample(iter=50, seed=7)
lda.wpdt_to_probs()
sum=0.0; for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
sum=0.0; for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
sum=0
sum=0.0; for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
usm
sum
#[Out]# -27365855.738786738
ll[50]
#[Out]# -27396870.707167484
sum
#[Out]# -27365855.738786738
ll[50]/lda.corpus.totalNwords
#[Out]# -7.183355027577135
ll[100]=sum
lda.gibbs_sample(iter=50, seed=7); lda.wpdt_to_probs()
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
sum
#[Out]# -27354221.034531552
ll[150]=sum
lda.gibbs_sample(iter=50, seed=7); lda.wpdt_to_probs()
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
sum
#[Out]# -27350874.492436238
ll[200]=sum
ll.values()
#[Out]# [-27350874.492436238, -27396870.707167484, -27365855.738786738, -27354221.034531552]
plot(ll.values())
#from liblda.low2corpus import Low2Corpus
import pylab as p
p.plot(ll.values())
#[Out]# [<matplotlib.lines.Line2D object at 0x111039bd0>]
p.show()
ll
#[Out]# {200: -27350874.492436238, 50: -27396870.707167484, 100: -27365855.738786738, 150: -27354221.034531552}
ll.iteritems()
#[Out]# <dictionary-itemiterator object at 0x11102e928>
ll.items()
#[Out]# [(200, -27350874.492436238), (50, -27396870.707167484), (100, -27365855.738786738), (150, -27354221.034531552)]
sorted(ll.items())
#[Out]# [(50, -27396870.707167484), (100, -27365855.738786738), (150, -27354221.034531552), (200, -27350874.492436238)]
llike = [l for i,l in sorted(ll.items())]
llike
#[Out]# [-27396870.707167484, -27365855.738786738, -27354221.034531552, -27350874.492436238]
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x1111eced0>]
lda.gibbs_sample(iter=50, seed=7); lda.wpdt_to_probs()
sum
#[Out]# -27350874.492436238
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
ll[250]=sum
llike = [l for i,l in sorted(ll.items())]
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x1111f7ed0>]
ll
#[Out]# {200: -27350874.492436238, 50: -27396870.707167484, 100: -27365855.738786738, 250: -27349769.814838134, 150: -27354221.034531552}
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x111229e90>]
lda.gibbs_sample(iter=200, seed=8); lda.wpdt_to_probs()
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
ll[450]=sum
llike = [l for i,l in sorted(ll.items())]
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x11123a2d0>]
lda.gibbs_sample(iter=300, seed=8); lda.wpdt_to_probs()
sum=0
for i in range(0,lda.corpus.totalNwords):
    sum += np.log( np.inner( lda.phi[:,lda.w[i]], lda.theta[:,lda.d[i]] ) )
    
ll[750]=sum
llike = [l for i,l in sorted(ll.items())]
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x1112c0d10>]
p.show()
p.plot( llike )
#[Out]# [<matplotlib.lines.Line2D object at 0x1112ea510>]
#p.title("log likelyhood for 50 100 150
ll
#[Out]# {450: -27347333.008298151, 100: -27365855.738786738, 200: -27350874.492436238, 750: -27340782.369485803, 50: -27396870.707167484, 150: -27354221.034531552, 250: -27349769.814838134}
#p.title("log likelyhood for 50 100 1
#?p.plot
man zip
#?zip
x,y = zip(*ll)
ll
#[Out]# {450: -27347333.008298151, 100: -27365855.738786738, 200: -27350874.492436238, 750: -27340782.369485803, 50: -27396870.707167484, 150: -27354221.034531552, 250: -27349769.814838134}
p.title("log likelyhood for 50 100 150 200 250 450 and 750 iterations")
#[Out]# <matplotlib.text.Text object at 0x1112d6c50>
p.xlabel('iterations')
#[Out]# <matplotlib.text.Text object at 0x1112cffd0>
p.ylabel('log likelyhood of data')
#[Out]# <matplotlib.text.Text object at 0x1112d3e50>
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda
#[Out]# <liblda.LDAmodel.LdaModel object at 0x1112f3510>
lda.train(30)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda.train(30)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda.train(30)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda.train(30)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda.train(30)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
#lda.random_initialize
lda.allocate_arrays()
lda
#[Out]# <liblda.LDAmodel.LdaModel object at 0x11130da90>
lda.wp
#[Out]# array([[0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0]], dtype=int32)
lda.dp
#[Out]# array([[0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0],
#[Out]#        [0, 0, 0]], dtype=int32)
lda.train(10)
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda.train(10)
#lda.train(10)
beta = lda.berta
beta = lda.beta
beta
#[Out]# array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1])
beta[3]=0.4
beta
#[Out]# array([ 0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1])
beta.reshape( 3*len(beta) )
beta.resize( 3*len(beta) )
np.resize( beta, 3*len(beta) )
#[Out]# array([ 0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1,  0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1,  0.1,  0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1,  0.1,  0.1])
np.resize( beta, (len(beta),3) )
#[Out]# array([[ 0.1,  0.1,  0.1],
#[Out]#        [ 0.4,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.4,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.4,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1]])
np.resize( beta, (3,len(beta)) )
#[Out]# array([[ 0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#          0.1],
#[Out]#        [ 0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#          0.1],
#[Out]#        [ 0.1,  0.1,  0.1,  0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#          0.1]])
np.resize( beta, (3,len(beta)) ).transpose()
#[Out]# array([[ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.4,  0.4,  0.4],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1],
#[Out]#        [ 0.1,  0.1,  0.1]])
reload(liblda.LDAmodel); from liblda.LDAmodel import LdaModel;
lda = LdaModel( numT=3, alpha=0.1, beta=0.1, corpus=tcorpus )
lda.train(10)
lda.beta
#[Out]# array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
#[Out]#         0.1])
sum(lda.wp,1)
np.sum(lda.wp,1)
#[Out]# array([2, 2, 2, 2, 4, 2, 3, 2, 3, 3, 2, 2])
np.sum(lda.wp,0)
#[Out]# array([ 9, 11,  9])
lda.phi
#[Out]# array([[ 0.00980392,  0.00980392,  0.20588235,  0.10784314,  0.10784314,
#[Out]#          0.20588235,  0.30392157,  0.00980392,  0.00980392,  0.00980392,
#[Out]#          0.00980392,  0.00980392],
#[Out]#        [ 0.00819672,  0.00819672,  0.00819672,  0.09016393,  0.00819672,
#[Out]#          0.00819672,  0.00819672,  0.00819672,  0.25409836,  0.25409836,
#[Out]#          0.17213115,  0.17213115],
#[Out]#        [ 0.20588235,  0.20588235,  0.00980392,  0.00980392,  0.30392157,
#[Out]#          0.00980392,  0.00980392,  0.20588235,  0.00980392,  0.00980392,
#[Out]#          0.00980392,  0.00980392]])
lda.theta
#[Out]# array([[ 0.33333333,  0.03030303,  0.63636364],
#[Out]#        [ 0.96825397,  0.01587302,  0.01587302],
#[Out]#        [ 0.25581395,  0.02325581,  0.72093023],
#[Out]#        [ 0.02325581,  0.02325581,  0.95348837],
#[Out]#        [ 0.93939394,  0.03030303,  0.03030303],
#[Out]#        [ 0.07692308,  0.84615385,  0.07692308],
#[Out]#        [ 0.04347826,  0.91304348,  0.04347826],
#[Out]#        [ 0.03030303,  0.93939394,  0.03030303],
#[Out]#        [ 0.03030303,  0.93939394,  0.03030303]])
