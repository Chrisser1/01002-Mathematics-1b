from sympy import Matrix, simplify, factorial
from sympy import hessian



def evaluateFunction(f,x,x0):
    for i in range(len(x)):
        f = f.subs({x[i]:x0[i]})
    return f

def taylorDegree2(f,x,x0): # What is called taylorN in the assignment
    N = len(x)
    # Konstant-led
    const = evaluateFunction(f,x,x0)
    # Førstegrads-led
    J = Matrix([f]).jacobian(x)
    J0 = evaluateFunction(J,x,x0)
    first = J0*(x-x0)
    # Andengrads-led
    H = hessian(f,x)
    H0 = evaluateFunction(H,x,x0)
    second = 1/2*(x-x0).T*H0*(x-x0)
    # Resultat
    Pk = simplify(Matrix([const]) + first + second)[0]
    return Pk

def multi_indices(N, k):
    """
    Generate all N-tuples of nonnegative integers that sum to k.
    """
    if N == 1:
        yield (k,)
    else:
        for i in range(k+1):
            for tail in multi_indices(N-1, k-i):
                yield (i,) + tail

def taylorN(f, x, x0, n):
    """
    Compute the Taylor polynomial of degree n for the function f in variables x
    around the point x0.

    Parameters:
    - f: sympy expression representing the function.
    - x: list (or Matrix) of sympy symbols.
    - x0: list (or Matrix) of values (the expansion point).
    - n: integer, the degree of the Taylor polynomial.

    Returns:
    - The simplified Taylor polynomial (sympy expression).
    """
    P = 0
    N = len(x)

    # Sum over all orders from 0 up to n
    for k in range(n+1):
        # Sum over all multi-indices (alpha) with |alpha| = k
        for alpha in multi_indices(N, k):
            # Compute the k-th order partial derivative corresponding to alpha
            deriv = f
            for i, a in enumerate(alpha):
                if a > 0:
                    deriv = deriv.diff(x[i], a)
            # Evaluate the derivative at x0
            deriv_at_x0 = evaluateFunction(deriv, x, x0)

            # Compute the factorial product: alpha! = a1! * a2! * ... * aN!
            denom = 1
            for a in alpha:
                denom *= factorial(a)

            # Construct the monomial: (x[0]-x0[0])^a1 * ... * (x[N-1]-x0[N-1])^aN
            term = 1
            for i, a in enumerate(alpha):
                term *= (x[i]-x0[i])**a

            # Add the term to the polynomial
            P += deriv_at_x0/denom * term
    return simplify(P)
