
from scipy import weave
from scipy.weave import converters

import numpy


def simpleforloop(a):

    z=[long(0)]    # return value

    #n=int(len(a))
    code="""
    int i;
    long int counter;
    counter=0;
    for(i=0; i< a; i++){
        counter=counter+i;
    }
    return_val=3;
    """

    err=weave.inline(code, ['a','z'],
                    #type_converters=converters.blitz,
                    headers=["<stdio.h>","<stdlib.h>","<string.h>", "<math.h>"],
                    compiler='gcc')

    print "z after = " + str(z)

    return err




