import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
y = np.arange(0, 11) ** 3
x = np.arange(0, 11)
plt.plot(x, y, color='r')

plt.show();
=======
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, c='m', alpha=0.5, marker='o')

plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")

plt.show();
>>>>>>> a748f355a43aab76c255a25a0f151b5ff5eb28fc
