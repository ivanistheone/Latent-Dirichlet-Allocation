#!/usr/bin/env python2.5

import urllib
import os
import re

pat = re.compile('<td><a href="([0-9]*)/">')

for num, line in enumerate(open('index.html')):
    parts = pat.findall(line)
    if not parts or len(parts) > 1:
        continue
    part = parts[0]
    url = 'http://arxmliv.kwarc.info/files/math/papers/%s/%s.tex.xml' % (part, part)
    print url
