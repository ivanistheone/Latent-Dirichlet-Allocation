import pylab
from pylab import arange,pi,sin,cos,sqrt


fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)
# Generate data
x = pylab.arange(-2*pi,2*pi,0.01)
y1 = sin(x)
y2 = cos(x)
# Plot data
pylab.figure(1)
pylab.clf()
pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.plot(x,y1,'g:',label='$\sin(x)$')
pylab.plot(x,y2,'-b',label='$\cos(x)$')
pylab.xlabel('$x$ (radians)')
pylab.ylabel('$y$')
pylab.legend()
pylab.savefig('fig1.eps')


