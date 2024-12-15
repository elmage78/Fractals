import numpy as np
from manimlib import *


def roots_to_coefficients(roots):
    return np.poly(roots)

coefs = [1.0,0.0,0.0,-1.0]
roots = np.roots(coefs)
roots = [complex(3.0,0.0000121333),complex(3.1233,-1.21313),complex(-3.13333,3.322)]

full_coefs = [*coefs] + [0] * (3 - len(coefs) + 1)
full_roots = [*roots] + [0] * (3 - len(roots))

print (np.poly([complex(-3,3),complex(-3,-3),complex(1,0)]))