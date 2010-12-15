
from scipy import weave
from scipy.weave import converters

import numpy


def simpleforloop(a):

    #n=int(len(a))
    code="""
    int i;
    long int counter;
    counter=0;
    for(i=0; i< 10000; i++){
        counter=counter+i;
    }
    return_val=counter;
    """

    err=weave.inline(code)# ['a','n'] ,type_converters=converters.blitz, compiler='gcc')
    return err




def mysum(a):
    n=int(len(a))
    code="""
    int i;
    long int counter;
    counter =0;
    for(i=0;i<n;i++){
        counter=counter+ (int)a[i];
    }
    return_val=counter;
    """

    err=weave.inline(code,['a','n'], compiler='gcc')
    return err


# a really long lis
a=range(6000000)

print "this is how long it takes to add up the first 6000000 integers with numpy"



