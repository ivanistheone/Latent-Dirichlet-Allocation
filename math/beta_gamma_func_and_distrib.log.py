#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True, 'logfile': 'beta_gamma_func_and_distrib.py', 'pylab': 1})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
import scipy
#import scipy
from scipy import special
x=arange(1,10,0.1)
y=special.gamma(x)
plot(x,y)
#[Out]# [<matplotlib.lines.Line2D object at 0x10347ecd0>]
plot(x,log(y))
#[Out]# [<matplotlib.lines.Line2D object at 0x1026e68d0>]
clf 
#[Out]# <function clf at 0x103768500>
clf()
plot(x,log(y))
#[Out]# [<matplotlib.lines.Line2D object at 0x102705990>]
title("Plot of log of $\Gamma$ from 0 to 10")
#[Out]# <matplotlib.text.Text object at 0x1026dd510>
from scipy import stats
#?stats.beta
stats.beta.pdf(1)
stats.beta.pdf(1, 2,5)
#[Out]# 0.0
stats.beta.pdf(0, 2,5)
#[Out]# 0.0
stats.beta.pdf(4.0, 2,5)
#[Out]# 0.0
stats.beta.pdf(0.4, 2,5)
#[Out]# 1.5551999999999999
#stats.beta.pdf(0.4, 2,5)
x=arange(0,1,0.01)
#y=arange(0,1,0.01)
y = stats.beta.pdf(x, 2,5)
plot(x,y)
#[Out]# [<matplotlib.lines.Line2D object at 0x11401d910>]
plot(x,y, title="Beta distribution")
plot(x,y)
#[Out]# [<matplotlib.lines.Line2D object at 0x11ab67d90>]
plot(x,stats.beta(x, 1,5))
plot(x,stats.beta.pdf(x,1,5))
#[Out]# [<matplotlib.lines.Line2D object at 0x11b3b4dd0>]
plot(x,stats.beta.pdf(x,3,5))
#[Out]# [<matplotlib.lines.Line2D object at 0x114030050>]
plot(x,stats.beta.pdf(x,1,1))
#[Out]# [<matplotlib.lines.Line2D object at 0x11402b190>]
plot(x,stats.beta.pdf(x,1,2))
#[Out]# [<matplotlib.lines.Line2D object at 0x11402b150>]
plot(x,stats.beta.pdf(x,1,3))
#[Out]# [<matplotlib.lines.Line2D object at 0x114030b90>]
plot(x,stats.beta.pdf(x,0.1,3))
#[Out]# [<matplotlib.lines.Line2D object at 0x114030350>]
plot(x,stats.beta.pdf(x,3,0.1))
#[Out]# [<matplotlib.lines.Line2D object at 0x11402d110>]
plot(x,stats.beta.pdf(x,1,0.9))
#[Out]# [<matplotlib.lines.Line2D object at 0x11b3c6c90>]
clf()
plot(x,stats.beta.pdf(x,1,0.9))
#[Out]# [<matplotlib.lines.Line2D object at 0x11b3da910>]
plot(x,stats.beta.pdf(x,1,0.99))
#[Out]# [<matplotlib.lines.Line2D object at 0x11b84ccd0>]
clf()
plot(x,1/sqrt(x)
)
#[Out]# [<matplotlib.lines.Line2D object at 0x11b86a550>]
stats.import mpl_toolkits.mplot3d.axes3d as p3
import mpl_toolkits.mplot3d.axes3d as p3
fig=p.figure()
fig=p.figure()
fig=p3.figure()
fig=figure()
ax = p3.Axes3D(fig)
#numpy.random.dirichlet
#?numpy.random.dirichlet
(x,y,z)=numpy.random.dirichlet([0.5,3,3], 3)
(x,y,z)=numpy.random.dirichlet([0.5,3,3], 5)
s=numpy.random.dirichlet([0.5,3,3], 5)
s
#[Out]# array([[ 0.08619444,  0.62634005,  0.28746551],
#[Out]#        [ 0.12159434,  0.54585735,  0.33254831],
#[Out]#        [ 0.3518952 ,  0.5364836 ,  0.11162121],
#[Out]#        [ 0.30257971,  0.2858588 ,  0.41156149],
#[Out]#        [ 0.02345192,  0.17496258,  0.8015855 ]])
s[1:]
#[Out]# array([[ 0.12159434,  0.54585735,  0.33254831],
#[Out]#        [ 0.3518952 ,  0.5364836 ,  0.11162121],
#[Out]#        [ 0.30257971,  0.2858588 ,  0.41156149],
#[Out]#        [ 0.02345192,  0.17496258,  0.8015855 ]])
s[1,:]
#[Out]# array([ 0.12159434,  0.54585735,  0.33254831])
x=s[:,0]; y=s[:,1], z=s[:
x=s[:,0]; y=s[:,1]; z=s[:,2]
ax.scatter3D(x,y,z)
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11b895e10>
ax.show()
p3.show()
show()
s=numpy.random.dirichlet([0.5,3,3], 5000)
clf()
x=s[:,0]; y=s[:,1]; z=s[:,2]
ax.scatter3D(x,y,z)
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11b6c1210>
show()
fig=figure()
ax = p3.Axes3D(fig)
ax.scatter3D(x,y,z)
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11b6c16d0>
show()
ax.set_xlabel('X')
#[Out]# <matplotlib.text.Text object at 0x11b369290>
ax.set_ylabel('$x_2$')
#[Out]# <matplotlib.text.Text object at 0x11b36c210>
ax.set_zlabel('$x_3$')
#[Out]# <matplotlib.text.Text object at 0x11b36cf10>
clf();
fig=figure(); ax = p3.Axes3D(fig); 
s=numpy.random.dirichlet([0.5,0.5,3], 5000)
x=s[:,0]; y=s[:,1]; z=s[:,2]; ax.scatter3D(x,y,z); show()
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11c4192d0>
fig=figure(); ax = p3.Axes3D(fig); 
s=numpy.random.dirichlet([0.5,0.5,0.9], 5000)
x=s[:,0]; y=s[:,1]; z=s[:,2]; ax.scatter3D(x,y,z); show()
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11b939390>
fig=figure(); ax = p3.Axes3D(fig); 
s=numpy.random.dirichlet([0.5,0.9,0.9], 5000)
x=s[:,0]; y=s[:,1]; z=s[:,2]; ax.scatter3D(x,y,z); show()
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11c65f510>
fig=figure(); ax = p3.Axes3D(fig); 
s=numpy.random.dirichlet([0.1,0.1,0.1], 5000)
x=s[:,0]; y=s[:,1]; z=s[:,2]; ax.scatter3D(x,y,z); show()
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11c41b490>
fig=figure(); ax = p3.Axes3D(fig); 
s=numpy.random.dirichlet([0.01,0.01,0.01], 5000)
x=s[:,0]; y=s[:,1]; z=s[:,2]; ax.scatter3D(x,y,z); show()
#[Out]# <mpl_toolkits.mplot3d.art3d.Patch3DCollection object at 0x11b95a610>
ones(4)
#[Out]# array([ 1.,  1.,  1.,  1.])
