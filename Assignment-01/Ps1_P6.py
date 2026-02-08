import numpy as np

# ---------------------------
# Geometry
# ---------------------------
L = 4.0         # m
H1 = 4.5        # m
H2 = 3.5        # m
H3 = 3.5        # m

# ---------------------------
# Material
# ---------------------------
E = 20          # kN/mm^2
# convert kN/mm2 → N/m2
E = E * 1e6 * 1e6   # 1 kN = 1000 N, 1 mm = 1e-3 m

# ---------------------------
# Section properties
# ---------------------------
b = 0.3         # m
D = 0.4         # m
I = b * D**3 / 12

# ---------------------------
# Story stiffness terms
# ---------------------------
def story_stiffness(E,I,H):
    return 2 * (12 * E * I / H**3)

K1 = story_stiffness(E,I,H1)
K2 = story_stiffness(E,I,H2)
K3 = story_stiffness(E,I,H3)

# ---------------------------
# Global stiffness matrix
# ---------------------------
K = np.array([
    [K1 + K2,   -K2,        0],
    [-K2,       K2 + K3,   -K3],
    [0,        -K3,         K3]
])

print("Stiffness matrix K =\n",K)

# ---------------------------
# Mass matrix
# ---------------------------
w = 300.0       # kg/m
m1 = w * L
m2 = w * L
m3 = w * L

M = np.diag([m1, m2, m3])

print("\nMass matrix M =\n",M)

# ---------------------------
# Eigenvalue problem
# ---------------------------
# Solve [K - ω^2 M]{φ}=0
evals, evecs = np.linalg.eig(np.linalg.inv(M).dot(K))

# natural circular frequencies
omega = np.sqrt(evals)

# natural frequencies in Hz
freq = omega / (2 * np.pi)

print("\nNatural circular frequencies ω (rad/s):\n",omega)
print("\nNatural frequencies f (Hz):\n",freq)

# mode shape normalization (first component = 1)
modes = evecs / evecs[0,:]

print("\nMode shapes (normalized):\n",modes)
