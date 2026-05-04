
import numpy as np
import matplotlib.pyplot as plt

# Given data
L = 4.0   # m
w = 20.0  # kN/m

# Reactions (standard results for two equal spans)
RA = RC = (3/8) * w * L
RB = (5/4) * w * L

print("Reactions:")
print(f"RA = {RA:.2f} kN")
print(f"RB = {RB:.2f} kN")
print(f"RC = {RC:.2f} kN")

# Bending moment at B
MB = -w * L**2 / 8
print(f"\nBending Moment at B = {MB:.2f} kNm")

# -----------------------------
# SHEAR FORCE DIAGRAM (SFD)
# -----------------------------
x = np.linspace(0, 2*L, 400)
V = np.zeros_like(x)

for i, xi in enumerate(x):
    if xi <= L:
        V[i] = RA - w * xi
    else:
        V[i] = RA - w * xi + RB

plt.figure()
plt.plot(x, V)
plt.axhline(0)
plt.xlabel("Length (m)")
plt.ylabel("Shear Force (kN)")
plt.title("Shear Force Diagram (SFD)")
plt.grid()

# Mark supports
plt.axvline(0, linestyle='--')
plt.axvline(L, linestyle='--')
plt.axvline(2*L, linestyle='--')

plt.show()

# -----------------------------
# BENDING MOMENT DIAGRAM (BMD)
# -----------------------------
M = np.zeros_like(x)

for i, xi in enumerate(x):
    if xi <= L:
        M[i] = RA * xi - w * xi**2 / 2
    else:
        # Correct expression using continuity
        M[i] = RA * xi - w * xi**2 / 2 + RB * (xi - L)

plt.figure()
plt.plot(x, M)
plt.axhline(0)
plt.xlabel("Length (m)")
plt.ylabel("Bending Moment (kNm)")
plt.title("Bending Moment Diagram (BMD)")
plt.grid()

# Mark supports
plt.axvline(0, linestyle='--')
plt.axvline(L, linestyle='--')
plt.axvline(2*L, linestyle='--')

plt.show()

# -----------------------------
# PART (e): M_B vs x
# -----------------------------
x_pos = np.linspace(0, L, 100)
MB_x = -10 * x_pos * (4 - x_pos)

plt.figure()
plt.plot(x_pos, MB_x)
plt.axhline(0)
plt.xlabel("Position x from A (m)")
plt.ylabel("Moment at B (kNm)")
plt.title("Variation of M_B with Load Position")
plt.grid()
plt.show()
