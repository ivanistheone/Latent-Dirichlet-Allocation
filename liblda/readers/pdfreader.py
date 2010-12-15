

from liblda.readers.interfaces import ReaderABC
import tempfile


import os,sys
import subprocess


import logging
logger = logging.getLogger('readers:')
logger.setLevel(logging.INFO)


class PDFReader(ReaderABC):
    """
    This will read the contents of a pdf
    using pdftotext (which comes with Xpdf and
    must be installed separately.

        pdftotext --help

        pdftotext version 3.02
        Copyright 1996-2007 Glyph & Cog, LLC
        Usage: pdftotext [options] <PDF-file> [<text-file>]
          -f <int>          : first page to convert
          -l <int>          : last page to convert
          -layout           : maintain original physical layout
          -raw              : keep strings in content stream order
          -htmlmeta         : generate a simple HTML file, including the meta information
          -enc <string>     : output text encoding name
          -eol <string>     : output end-of-line convention (unix, dos, or mac)
          -nopgbrk          : don't insert page breaks between pages
          -upw <string>     : user password (for encrypted files)
          -q                : don't print any messages or errors
          -cfg <string>     : configuration file to use in place of .xpdfrc
          -v                : print copyright and version info
          -h                : print usage information
          -help             : print usage information
          --help            : print usage information
          -?                : print usage information


    and shoot all of it back to you This is the abstract class
    """

    def __init__(self, path):
        """
        Prepare to read contents of `path`
        """
        self.path = path     #absolute path

    def get_path(self):
        return self.path


    def __iter__(self):
        """
        Return a string of tokens, ex. readline.
        For now we dump the whole PDF in one shot
        by getting the output of the utilituy pdftotext
        """

        tmpfn = tempfile.mktemp()
        tmpf = open(tmpfn,"w")         # touch
        tmpf.close()

        runcommand = "pdftotext " + self.path + " " + tmpf.name
        logger.info("run command: "+runcommand )

        (out,err)  = subprocess.Popen(runcommand, shell=True, \
                                                   stdout=subprocess.PIPE, \
                                                   stderr=subprocess.PIPE \
                                                   ).communicate()

        # lets get the text from the tmp file now
        tmpf = open(tmpfn,"r")
        contents =  tmpf.read()

        # kill the temp file
        tmpf.close()
        os.unlink(tmpfn)

        yield contents







