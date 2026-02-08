import numpy as np
import matplotlib.pyplot as plt

# Input
P = 10      # kN (example, change as needed)
L = 6       # m
a = 2       # m
E = 2e5     # kN/m^2 (convert if needed)
I = 8e-3    # m^4 (example)

b = L-a

# Reactions
RA = P*(L-a)/L
RB = P*a/L

print("RA =",RA)
print("RB =",RB)

# x array
x = np.linspace(0,L,400)

# Shear
V = np.piecewise(x, [x<a, x>=a], [lambda x: RA, lambda x: RA-P])

# Bending moment
M = np.piecewise(x, 
    [x<a, x>=a],
    [lambda x: RA*x,
     lambda x: RA*x - P*(x-a)]
)

# Deflection and slope using formula (Appendix A)
deflection = np.zeros_like(x)
slope = np.zeros_like(x)

for i,xi in enumerate(x):
    if xi<=a:
        deflection[i] = -P*b*xi*(L**2 - b**2 - xi**2)/(6*L*E*I)
    else:
        deflection[i] = -P*b*xi*(L**2 - b**2 - xi**2)/(6*L*E*I) - P*(xi-a)**3/(6*E*I)

# Plot diagrams
plt.figure()
plt.plot(x,V)
plt.title("Shear Force Diagram")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x,M)
plt.title("Bending Moment Diagram")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x,deflection)
plt.title("Deflection Curve")
plt.grid(True)
plt.show()
