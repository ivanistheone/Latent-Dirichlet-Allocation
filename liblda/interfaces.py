

class LdaModelABC(object):
    """
    This is an abstract base class for all LDA model implementations.

    Parameters:
        numT        : number of topics in model
        alpha       : Dirichlet prior on theta (topics in docs)
                      len: numT   or  constant scalar
        beta        : Dirichlet prior on phi (words in topics)


    Input data:
        corpus      : reference to a corpus object (anything that looks like [[ (termid,count) ]]
        numDocs     : len(corpus)

    Optional input:
        vocab       : ref to Vacabulary object (which can map word id to word string)
        numTerms    : len(vocab)

    API:
        corpus=[[(1,1),(2,4)],[(1,1)]]
        lda = LdaModel?( corpus=corpus, numT=17, alpha=50.0/17, beta=0.01 )
        lda.train()

    Then some time goes by...
    and the following things appear.

    Inferred model:
        phi         : topic-word distribution
                      size: numT x numTerms
                      alias: prob_w_given_t
        theta       : document-topic distibution
                      size: numDocs x numT
                      alias: prob_t_given_d
        z           : the topic assignment
                      size: ?

    """

    def __init__(self):
        """
        This creates the LDA model.
        Minimum required arguments are corpus and numT.
        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def is_trained(self):
        """
        Tells you whether model has been trained or not.
        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def train(self):
        """
        Runs the algorithm.
        """
        raise NotImplementedError('LDA Model must supply train function')


