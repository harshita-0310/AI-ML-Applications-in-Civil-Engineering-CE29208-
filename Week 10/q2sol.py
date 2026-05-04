import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Data
RH = np.array([24, 26, 28, 34, 38, 38])
UPV = np.array([3.2, 3.0, 3.2, 4.0, 3.8, 4.2])
y = np.array([0, 0, 0, 1, 1, 1])  # Poor=0, Good=1

X = np.column_stack((RH, UPV))

# Train model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Plot data points
plt.figure()
for i in range(len(y)):
    if y[i] == 0:
        plt.scatter(RH[i], UPV[i], color='red', label='Poor' if i == 0 else "")
    else:
        plt.scatter(RH[i], UPV[i], color='green', label='Good' if i == 3 else "")

# Support vectors
plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=100, facecolors='none', edgecolors='black',
            label='Support Vectors')

# Decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[0], colors='blue')
ax.contour(XX, YY, Z, levels=[-1, 1], linestyles=['--','--'])

# Predictions
new_point1 = np.array([[30, 3.6]])
pred1 = model.predict(new_point1)

new_point2 = np.array([[32, 3.2]])
pred2 = model.predict(new_point2)

print("Point (RH=30, UPV=3.6):", "Good" if pred1[0]==1 else "Poor")
print("Point (RH=32, UPV=3.2):", "Good" if pred2[0]==1 else "Poor")

# Plot new points
plt.scatter(30, 3.6, color='blue', marker='x', s=100, label='New Point 1')
plt.scatter(32, 3.2, color='purple', marker='x', s=100, label='New Point 2')

plt.xlabel("RH")
plt.ylabel("UPV (km/s)")
plt.title("SVM Classification of Concrete Quality")
plt.legend()
plt.grid()
plt.show()
