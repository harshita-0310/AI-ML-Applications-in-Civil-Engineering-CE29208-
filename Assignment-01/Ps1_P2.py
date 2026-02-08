import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5,500)
f = np.exp(-t) * np.cos(2*np.pi*t)

plt.figure()
plt.plot(t,f)
plt.xlabel("t (s)")
plt.ylabel("f(t)")
plt.title("f(t)=e^{-t}cos(2πt) for 5 seconds")
plt.grid(True)
plt.show()
