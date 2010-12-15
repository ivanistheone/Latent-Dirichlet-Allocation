#!/usr/bin/env python

"""
Comparison and test of scipy.weave ...
"""

import numpy
from scipy import weave
from scipy.weave import converters
import sys, time



def inlineTimeStep(self, dt=0.0):
    """Takes a time step using inlined C code -- this version uses
    blitz arrays."""
    g = self.grid
    nx, ny = g.u.shape
    dx2, dy2 = g.dx**2, g.dy**2
    dnr_inv = 0.5/(dx2 + dy2)
    u = g.u

    code = """
           #line 120 "laplace.py"
           double tmp, err, diff;
           err = 0.0;
           for (int i=1; i<nx-1; ++i) {
               for (int j=1; j<ny-1; ++j) {
                   tmp = u(i,j);
                   u(i,j) = ((u(i-1,j) + u(i+1,j))*dy2 +
                             (u(i,j-1) + u(i,j+1))*dx2)*dnr_inv;
                   diff = u(i,j) - tmp;
                   err += diff*diff;
               }
           }
           return_val = sqrt(err);
           """
    # compiler keyword only needed on windows with MSVC installed
    err = weave.inline(code,
                       ['u', 'dx2', 'dy2', 'dnr_inv', 'nx','ny'],
                       type_converters = converters.blitz,
                       compiler = 'gcc')
    return err

def fastInlineTimeStep(self, dt=0.0):
    """Takes a time step using inlined C code -- this version is
    faster, dirtier and manipulates the numeric array in C.  This
    code was contributed by Eric Jones.  """
    g = self.grid
    nx, ny = g.u.shape
    dx2, dy2 = g.dx**2, g.dy**2
    dnr_inv = 0.5/(dx2 + dy2)
    u = g.u

    code = """
           #line 151 "laplace.py"
           double tmp, err, diff;
           double *uc, *uu, *ud, *ul, *ur;
           err = 0.0;
           for (int i=1; i<nx-1; ++i) {
               uc = u+i*ny+1;
               ur = u+i*ny+2;     ul = u+i*ny;
               ud = u+(i+1)*ny+1; uu = u+(i-1)*ny+1;
               for (int j=1; j<ny-1; ++j) {
                   tmp = *uc;
                   *uc = ((*ul + *ur)*dy2 +
                          (*uu + *ud)*dx2)*dnr_inv;
                   diff = *uc - tmp;
                   err += diff*diff;
                   uc++;ur++;ul++;ud++;uu++;
               }
           }
           return_val = sqrt(err);
           """
    # compiler keyword only needed on windows with MSVC installed
    err = weave.inline(code,
                       ['u', 'dx2', 'dy2', 'dnr_inv', 'nx','ny'],
                       compiler='gcc')
    return err
