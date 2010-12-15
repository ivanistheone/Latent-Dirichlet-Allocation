

from liblda.readers.interfaces import ReaderABC


import os,sys
import subprocess


import logging
logger = logging.getLogger('readers:')
logger.setLevel(logging.INFO)


class WeblocReader(ReaderABC):
    """
    This will read the contents of a Mac OS .webloc
    -- a link to a website.
    There are two formats, the safari native:

        bplist00##SURL_http://localhost/

    and the XML style (fireforx, chrome):

        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC ...
        <plist version="1.0">
        <dict>
            <key>URL</key>
            <string>http://localhost/miniref/doku.php</string>
        </dict>
        </plist>

    I guess we will have to handle both kinds.

    Get url, visit using urllib2, remove HTML tags and code...
    for the moment we just get a dump from w3m

    """

    def __init__(self, path):
        """
        Prepare to read contents of `path`
        """
        self.path = path     #absolute path of .webloc file

        # open and read first few bytes






    def get_path(self):
        return self.path


    def get_url(self):
        return self.url


    def __iter__(self):
        """
        Return a string of text which is what the
        user is most likely to see when going to the page.
        If possible skip ads.
        What would the URL look like in a text mode browser?
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







