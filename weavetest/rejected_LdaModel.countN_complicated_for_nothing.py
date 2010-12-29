
# this file can serve as an example for someone
# trying to play with scipy.weave.inline
#
#  Uses several APIs:
#   -- standard Python C API (PyObject, PyIter_Next, PyLong_FromLong ...
#   -- the CXX types py::tuple -- which gives the [ ] to access elements
#
#  Much of the CXX objects are have their ref-counting "automatically"
#  automatically taken care of.
#  For the PythonC-API vars, we have to do manual reference counting


# Note:     This code is completely USELESS since it calls back
#           into the Python C api -- PyIter_Next etc...
#           so might as well do it in python ;)
#           I am including it here for posterity's sake
#           and to show how tough I am -- I can handle ref counting ;)


# Note 2:   The input to this function is a `corpus` which is
#           an array of arrays of tuples like:
#           (actually an iterable, of iterables of tuples)
#
#                [[(0, 1.0), (1, 1.0), (2, 1.0)],
#                 [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)],
#                 [(2, 1.0), (5, 1.0), (7, 1.0), (8, 1.0)],
#                 [(1, 1.0), (5, 2.0), (8, 1.0)],
#                 [(3, 1.0), (6, 1.0), (7, 1.0)],
#                 [(9, 1.0)],
#                 [(9, 1.0), (10, 1.0)],
#                 [(9, 1.0), (10, 1.0), (11, 1.0)],
#                 [(4, 1.0), (10, 1.0), (11, 1.0)]]
#
#           where each row represents a doc, each tuple is of the form (term_id, count(term_id in doc) )
#           countN finds the total number of words in corpus


    def countN(self):
        """ Count the total number of words in corpus """


        # need to give:
        corpus = self.corpus
        numDocs = self.numDocs

        # the C code
        code =  """
                //line 203 in LDAmodel.py

                int m;      // doc m in corpus    0 .. numDocs-1
                int n;      // term n-in doc m    0 .. #numTerms in doc m

                long total = 0;


                // from http://docs.python.org/c-api/iter.html
                PyObject *iterator = PyObject_GetIter(corpus);
                PyObject *doc;

                PyObject *inneriterator;
                PyObject *term_tuple_ptr;
                py::tuple term_tuple;

                m=0;
                while (doc = PyIter_Next(iterator)) {
                    /* doc is a list-like */

                    inneriterator = PyObject_GetIter(doc);
                    n=0;
                    while (term_tuple_ptr = PyIter_Next(inneriterator)) {

                        term_tuple = py::tuple((PyObject*)term_tuple_ptr);
                        total =  total + (int) term_tuple[1];

                        Py_DECREF(term_tuple_ptr);
                        n++;
                    }
                    Py_DECREF(inneriterator);



                    /* release reference when done */
                    Py_DECREF(doc);
                    m++;
                }

                Py_DECREF(iterator);

                return_val = PyLong_FromLong(total);
                """

        # compiler keyword only needed on windows with MSVC installed
        err = sp.weave.inline( code,
                               ['corpus', 'numDocs'],
                               compiler='gcc')
        return err


