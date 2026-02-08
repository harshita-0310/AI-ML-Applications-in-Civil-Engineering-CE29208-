import numpy as np

# Geometry
L = 4.0
H1 = 4.5
H2 = 3.5
H3 = 3.5

# Material
E = 20  # kN/mm2
E = E * 1e6  # convert to kN/m2

# Section
b = 0.3
D = 0.4
I = b*D**3/12

# Storey stiffness
def K(E,I,H):
    return 2*(12*E*I/H**3)

K1 = K(E,I,H1)
K2 = K(E,I,H2)
K3 = K(E,I,H3)

Kmat = np.array([
    [K1+K2, -K2,     0],
    [-K2,   K2+K3, -K3],
    [0,    -K3,     K3]
])

print("Stiffness matrix:\n",Kmat)

# Load vector
P = np.array([1000e3,1000e3,1000e3])  # kN → N scale consistent

# Displacements
disp = np.linalg.solve(Kmat,P)

print("Floor displacements (m):\n",disp)
