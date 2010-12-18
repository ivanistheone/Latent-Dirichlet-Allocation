# from presentation
# Speeding up Python with C/C++
# by Fernando Perez


from  numpy import *
from scipy import weave
from scipy.weave import converters

import cmath


def quantum_cat_python(N,kappa):
    # First initialize complex matrix with NxN elements
    mat=zeros((N,N), complex)
    # precompute a few things outside the loop
    sqrt_N_inv = 1.0/sqrt(N)
    alpha = 2.0*pi/N
    kap_al = kappa/alpha

    # now we fill each element
    for k in range(0,N):
        for l in range(0,N):
            mat[k,l] =  sqrt_N_inv * \
                        cmath.exp(1j*(alpha*(k*k-k*l+l*l) + \
                          kap_al*sin(alpha*l)))
    return(mat)


def quantum_cat_numeric(N,kappa):
    alpha = 2.0*pi/N
    mat_fn = lambda k,l: alpha*(k*k-k*l+l*l)
    phi =fromfunction(mat_fn,(N,N)) + \
            (kappa/alpha)*sin(alpha*arange(N))
    return (1.0/sqrt(N))*exp(1j*phi)

def quantum_cat_weave(N,kappa):
    phi = zeros((N,N), float)   # Initialize phase matrix
    support = "#include <math.h>"
    code = """
        float alpha = 2.0*pi/N;
        float kap_al = kappa/alpha;
        for (int k=0;k<N;++k)
            for(int l=0;l<N;++l)
                phi(k,l) = alpha*(k*k-k*l+l*l) + kap_al*sin(alpha*l);
        """

    # Call weave to fill in phi
    weave.inline(code,['N','kappa','pi','phi'],
                    type_converters = converters.blitz,
                    support_code = support,
                    libraries = ['m'])

    return (1.0/sqrt(N))*exp(1j*phi)


# Why is there this discrepancy ???
# a = quantum_cat_python(300,0.1)
# w = quantum_cat_weave(300,0.1);
#  norm(a-w)
#      Out[77]: 0.00042109772604762279
# ?





