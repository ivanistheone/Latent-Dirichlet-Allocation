#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True,
 'logfile': 'lots_of_alpha_plots_const_and_exp.log.py',
 'pylab': 1})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
_ip.magic("run mycmds.py")
help(pylab)
import pylab as p
np.arange(0,
x=np.arange(0,100,0.01)
_ip.system("ls -F ")
plot(x, np.exp(x) )
#[Out]# [<matplotlib.lines.Line2D object at 0x1035affd0>]
plot(x, np.exp(-x) )
#[Out]# [<matplotlib.lines.Line2D object at 0x1035bac90>]
plot(x, np.exp(-x) )
#[Out]# [<matplotlib.lines.Line2D object at 0x1035a3f90>]
plot(x, np.exp(-0.01x) )
plot(x, np.exp(-0.01*x) )
#[Out]# [<matplotlib.lines.Line2D object at 0x1036c14d0>]
x
#[Out]# array([  0.00000000e+00,   1.00000000e-02,   2.00000000e-02, ...,
#[Out]#          9.99700000e+01,   9.99800000e+01,   9.99900000e+01])
plot(x, np.exp(-0.1*x) )
#[Out]# [<matplotlib.lines.Line2D object at 0x1036c4d50>]
hasattr(plot, "__call__")
#[Out]# True
_ip.magic("cd math/")
_ip.system("ls -F ")
_ip.magic("run dirichlet_sparse_stats.py")
_ip.magic("run dirichlet_sparse_stats.py")
#?r = get_sparse_stats
#?get_sparse_stats
r = get_sparse_for_alpha(0.1)
_ip.magic("run dirichlet_sparse_stats.py")
r = get_sparse_for_alpha(0.1)
_ip.magic("run dirichlet_sparse_stats.py")
r = get_sparse_for_alpha(alphaval=0.1)
#r = get_sparse_for_alpha(alphaval=0.1)
_ip.magic("run dirichlet_sparse_stats.py")
r = get_sparse_for_alpha(alphaval=0.1)
plot_results(r)
# p.title('Sparseness of Dirichlet distribution (prior)' )
_ip.magic("run dirichlet_sparse_stats.py")
plot_results(r, case="0.1")
_ip.magic("run dirichlet_sparse_stats.py")
plot_results(r, case="0.1")
_ip.magic("run dirichlet_sparse_stats.py")
plot_results(r, case="0.1")
p.show()
p.figure()
#[Out]# <matplotlib.figure.Figure object at 0x118a80e90>
_ip.magic("run dirichlet_sparse_stats.py")
plot_results(r, case="0.1")
#?p.save
#?p.savefig
_ip.magic("pwd ")
#[Out]# '/Users/ivan/Homes/master/Documents/Projects/LatentDirichletAllocation/math'
p.save("alpha0.1sparseness.pdf")
p.savefig(fname="alpha0.1sparseness.pdf")
#?p.savefig
p.savefig("alpha0.1sparseness.pdf")
r = get_sparse_for_alpha(alphaval=0.01)
plot_results(r, case="0.01")
p.savefig("Dir_sparseness_a0dot01.pdf")
r01 = get_sparse_for_alpha(alphaval=0.1)
plot_results(r, case="0.1")
#p.savefig("Dir_sparseness_a0dot1.pdf")
plot_results(r01, case="0.1")
p.savefig("Dir_sparseness_a0dot1.pdf")
x = np.zeros(n)
#x = np.zeros(n)
n=4
x = np.zeros(n)
x
#[Out]# array([ 0.,  0.,  0.,  0.])
x = np.range(1,n)
x = np.arange(1,n)
x
#[Out]# array([1, 2, 3])
x = np.arange(0,n)
x
#[Out]# array([0, 1, 2, 3])
n
#[Out]# 4
x
#[Out]# array([0, 1, 2, 3])
a=1
a = 1.0
def expf(n):
    

    
    pass
x
#[Out]# array([0, 1, 2, 3])
exp(-a*x)
#[Out]# array([ 1.        ,  0.36787944,  0.13533528,  0.04978707])
np.exp(-a*x)/sum( np.exp(-a*x) )
#[Out]# array([ 0.64391426,  0.23688282,  0.08714432,  0.0320586 ])
#np.exp(-a*x)/sum( np.exp(-a*x) )
def expf(n):
    x = np.arange(0,n)
    return np.exp(-a*x)/sum( np.exp(-a*x) )
#rExp1 =
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="0.1")
#plot_results(rExp1, case="exp(-t)/sum(-t)")
def expf(n):
    x = np.arange(0,n)
    return np.exp(-0.1*x)/sum( np.exp(-0.1*x) )
rExp1 = get_sparse_for_alpha(alpha=expf)
#def expf(n):
    x = np.arange(0,n)
    return np.exp(-0.1*x)/sum( np.exp(-0.1*x) )
plot_results(rExp1, case="exp(-0.1t)/sum")
def expf(n):
    x = np.arange(0,n)
    return np.exp(-0.01*x)/sum( np.exp(-0.01*x) )
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="exp(-0.01t)/sum")
#plot_results(rExp1, case="exp(-0.0001t)/sum")
#rExp1 = get_sparse_for_alpha(alpha=expf)
def expf(n):
    x = np.arange(0,n)
    return np.exp(-0.00001*x)/sum( np.exp(-0.00001*x) )
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="exp(-0.00001t)/sum")
def expf(n):
    x = np.arange(0,n)
    return 10.0*np.exp(-0.00001*x)/sum( np.exp(-0.00001*x) )
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="10*exp(-0.00001t)/sum")
p.savefig("Dir_sparseness_a10Exp0dot00001.pdf")
np.exp(-a*x)/sum( np.exp(-a*x) )
#[Out]# array([ 0.64391426,  0.23688282,  0.08714432,  0.0320586 ])
3*3,0
#[Out]# (9, 0)
3*3.0
#[Out]# 9.0
_ip.magic("run dirichlet_sparse_stats.py")
expf = exp_function_maker(0.00001, 10.0)
expf(5)
#[Out]# array([ 2.00004,  2.00002,  2.     ,  1.99998,  1.99996])
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="10*exp(-0.00001t)/sum rerun")
#rExp1 = get_sparse_for_alpha(alpha=expf)
expf = exp_function_maker(0.00001, 30.0)
expf = exp_function_maker(0.00001, 50.0)
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="50*exp(-0.00001t)/sum rerun")
p.savefig("Dir_sparseness_a50Exp0dot00001.pdf")
expf = exp_function_maker(0.01, 30.0)
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="30*exp(-0.01t)/sum rerun")
expf = exp_function_maker(0.01, 50.0)
rExp1 = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="50*exp(-0.01t)/sum")
p.savefig("Dir_sparseness_a50Exp0dot01.pdf")
_ip.magic("run dirichlet_sparse_stats.py")
expf = n_scaled_exp_function_maker(0.01, 0.1)
r = get_sparse_for_alpha(alpha=expf)
plot_results(rExp1, case="0.1*n*exp(-0.01t)/sum")
plot_results(r, case="0.1*n*exp(-0.01t)/sum")
expf(10)
#[Out]# array([ 0.10455965,  0.10351926,  0.10248923,  0.10146945,  0.10045981,
#[Out]#         0.09946022,  0.09847057,  0.09749077,  0.09652072,  0.09556032])
expf(100)
#[Out]# array([ 0.15740931,  0.15584306,  0.1542924 ,  0.15275716,  0.1512372 ,
#[Out]#         0.14973237,  0.14824251,  0.14676747,  0.14530711,  0.14386128,
#[Out]#         0.14242984,  0.14101264,  0.13960954,  0.1382204 ,  0.13684508,
#[Out]#         0.13548345,  0.13413537,  0.1328007 ,  0.13147931,  0.13017107,
#[Out]#         0.12887584,  0.12759351,  0.12632393,  0.12506699,  0.12382255,
#[Out]#         0.1225905 ,  0.1213707 ,  0.12016304,  0.1189674 ,  0.11778365,
#[Out]#         0.11661169,  0.11545138,  0.11430262,  0.11316529,  0.11203928,
#[Out]#         0.11092447,  0.10982075,  0.10872802,  0.10764615,  0.10657506,
#[Out]#         0.10551462,  0.10446473,  0.10342529,  0.10239619,  0.10137733,
#[Out]#         0.10036861,  0.09936992,  0.09838118,  0.09740227,  0.0964331 ,
#[Out]#         0.09547357,  0.0945236 ,  0.09358307,  0.0926519 ,  0.09173   ,
#[Out]#         0.09081727,  0.08991363,  0.08901897,  0.08813322,  0.08725628,
#[Out]#         0.08638806,  0.08552849,  0.08467746,  0.08383491,  0.08300074,
#[Out]#         0.08217487,  0.08135721,  0.0805477 ,  0.07974623,  0.07895274,
#[Out]#         0.07816715,  0.07738938,  0.07661934,  0.07585696,  0.07510217,
#[Out]#         0.07435489,  0.07361505,  0.07288257,  0.07215738,  0.0714394 ,
#[Out]#         0.07072856,  0.0700248 ,  0.06932804,  0.06863822,  0.06795526,
#[Out]#         0.06727909,  0.06660965,  0.06594688,  0.06529069,  0.06464104,
#[Out]#         0.06399785,  0.06336106,  0.06273061,  0.06210643,  0.06148846,
#[Out]#         0.06087664,  0.06027091,  0.0596712 ,  0.05907746,  0.05848963])
#plot_results(r, case="0.1*n*exp(-3*t)/sum")
expf = n_scaled_exp_function_maker(5.0, 0.1)
expf(50)
#[Out]# array([  4.96631027e+000,   3.34627353e-002,   2.25470137e-004,
#[Out]#          1.51920583e-006,   1.02363284e-008,   6.89718382e-011,
#[Out]#          4.64728590e-013,   3.13131661e-015,   2.10986453e-017,
#[Out]#          1.42161554e-019,   9.57877017e-022,   6.45412457e-024,
#[Out]#          4.34875493e-026,   2.93016802e-028,   1.97433168e-030,
#[Out]#          1.33029422e-032,   8.96345197e-035,   6.03952643e-037,
#[Out]#          4.06940090e-039,   2.74194076e-041,   1.84750515e-043,
#[Out]#          1.24483918e-045,   8.38766041e-048,   5.65156113e-050,
#[Out]#          3.80799193e-052,   2.56580478e-054,   1.72882566e-056,
#[Out]#          1.16487357e-058,   7.84885637e-061,   5.28851782e-063,
#[Out]#          3.56337528e-065,   2.40098338e-067,   1.61776987e-069,
#[Out]#          1.09004477e-071,   7.34466386e-074,   4.94879558e-076,
#[Out]#          3.33447224e-078,   2.24674972e-080,   1.51384805e-082,
#[Out]#          1.02002279e-084,   6.87285953e-087,   4.63089632e-089,
#[Out]#          3.12027340e-091,   2.10242368e-093,   1.41660193e-095,
#[Out]#          9.54498873e-098,   6.43136282e-100,   4.33341818e-102,
#[Out]#          2.91983420e-104,   1.96736881e-106])
expf = n_scaled_exp_function_maker(3.0, 0.1)
expf(5)
#[Out]# array([  4.75106611e-01,   2.36541653e-02,   1.17767155e-03,
#[Out]#          5.86328138e-05,   2.91915591e-06])
expf = n_scaled_exp_function_maker(1.0, 0.1)
expf(5)
#[Out]# array([ 0.31820432,  0.11706083,  0.04306427,  0.01584246,  0.00582812])
r = get_sparse_for_alpha(alpha=expf)
plot_results(r, case="0.1*n*exp(-1*i)/sum")
#?p.find
expf = n_scaled_exp_function_maker(0.6, 0.1)
r = get_sparse_for_alpha(alpha=expf)
plot_results(r, case="0.1*n*exp(-0.6*i)/sum")
p.savefig("Dir_sparseness_a0dot1nExp0dot6.pdf")
