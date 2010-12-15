

class ReaderABC(object):
    """
    This is the abstract class
    """

    def __init__(self, path):
        """
        Prepare to read contents of `path`
        """
        pass


    def __iter__(self):
        """
        Return a string of tokens, ex. readline.
        Should return a string -- which will be parsed and tokenized  furher down the line...
        """
        pass




