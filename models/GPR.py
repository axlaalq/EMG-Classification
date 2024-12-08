import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


TV = np.loadtxt('Valores_esperados.txt')
c = np.loadtxt('Predicciones.txt')



# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([0.000075,0.0001,0.00002,0.00005,0.00001,0.000005]).T

# Observations
y = np.array([0.9,0.9405,0.9 ,0.9524,0.8333,0.7024])



# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 0.0001, 10000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.title('Learning rate optimization')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='lower right')
plt.show()
print(y)
print(y_pred)
