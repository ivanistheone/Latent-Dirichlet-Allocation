from pylab import *
from numpy import mod

dashes = ['--', #    : dashed line
          '-', #     : solid line
          '-.', #   : dash-dot line
          ':', #    : dotted line
           '-']

def NR(C,C50,n):
    return (  ( C / C50)**n /( 1 + ( C / C50)**n ))

figure(1)
C = arange(0.0,3.0,0.1)
C50 = arange(0.5,2.5,0.5)

for i,c50 in enumerate(C50):
    plot(C, NR(C,c50,2), label = 'C50 = ' + str(c50), c= 'k', linestyle = dashes[mod(i,len(dashes))] )

xlabel('Contrast')
ylabel('gain')
legend(loc='lower right')
savefig('naka-rushton.png')


