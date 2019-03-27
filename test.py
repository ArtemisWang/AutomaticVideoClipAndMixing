from sympy import *
from sympy.abc import x,y,z
print(solve([(1-x)**2+y**2-z**2,x**2+(1-y)**2-z**2,x**2+(-1-y)**2-z**2],[x,y,z])[1])