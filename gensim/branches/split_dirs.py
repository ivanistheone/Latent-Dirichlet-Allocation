#!/usr/bin/env python2.5

import urllib
import os
import os.path
import re
import shutil

fnames = list(os.listdir('.'))

for fname in fnames:
    if not fname.startswith('#'):
        continue
    moveto = fname[-2:]
    if not os.path.exists(moveto):
        os.mkdir(moveto)
    print "attempting to move", fname
    shutil.move(fname, "%s/%s" % (moveto, fname))
