import numpy as np
import matplotlib.pyplot as plt

# Inputs
w = 5       # kN/m
L = 6       # m
E = 2e5
I = 8e-3

# reactions
RA = RB = w*L/2

x = np.linspace(0,L,400)

# Shear
V = RA - w*x

# Bending moment
M = RA*x - w*x**2/2

# Deflection from Appendix-B
v = -w*x*(L**3 - 2*L*x**2 + x**3)/(24*E*I)

print("Maximum deflection =", np.min(v))

# Plotting
plt.figure()
plt.plot(x,V)
plt.title("Shear Force Diagram (UDL)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x,M)
plt.title("Bending Moment Diagram (UDL)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x,v)
plt.title("Deflection Curve (UDL)")
plt.grid(True)
plt.show()
