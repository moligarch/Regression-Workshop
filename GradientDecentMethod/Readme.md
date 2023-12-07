# Linear Regression using Gradient Descent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wRB8gI1fC_wvBjdSYAg6KAVP3jgzPlpr?usp=sharing)

The provided code implements linear regression using gradient descent. Linear regression is a machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables and aims to find the best-fitting line that minimizes the difference between the predicted and actual values.

The code defines a `LinReg` class that encapsulates the functionality of linear regression. Let's go through the code and explain its usage.

### Class Initialization

```python
class LinReg:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.w = np.zeros(self.n)
        self.b = 0
```

The `LinReg` class has an initializer (`__init__`) that takes two parameters: `X` and `y`. `X` is a numpy array representing the input features, and `y` is a numpy array representing the corresponding target values. The initializer initializes the class variables `self.X`, `self.y`, `self.m`, and `self.n`. `self.m` represents the number of samples in the dataset, and `self.n` represents the number of features.

The class also initializes `self.w` as a numpy array of zeros with shape `(self.n,)`, representing the weights of the linear regression model, and `self.b` as 0, representing the bias term.

### Cost Derivative Calculation

```python
    def GetCostDerivation(self, spec):
        #calculate f_{w,b}(X)
        def f(self, index):
            return np.dot(self.w, self.X[index]) + self.b

        if spec == 'w': # Return dJ/dw
            dJdW = np.zeros(self.n)
            for dim in range (self.n):
                for sample in range(self.m):
                    dJdW[dim] += ((f(self,sample)-y[sample]) * X[sample][dim])/self.m
            return  dJdW
        elif spec == 'b': # Return dJ/db
            dJdB = 0
            for sample in range(self.m):
                dJdB += (f(self, sample)-y[sample])/self.m
            return dJdB
```

The `GetCostDerivation` method calculates the derivative of the cost function with respect to the weights (`dJ/dw`) or the bias term (`dJ/db`). It takes a parameter `spec` that specifies which derivative to compute.

The method defines an inner function `f(self, index)` that calculates the linear regression hypothesis for a given sample at index `index`. It computes the dot product between the weights `self.w` and the feature vector `self.X[index]`, and adds the bias term `self.b`.

If `spec` is `'w'`, the method initializes `dJdW` as a numpy array of zeros with shape `(self.n,)`. It then iterates over the dimensions of the feature vector (`dim`) and the samples in the dataset (`sample`). For each dimension and sample, it updates `dJdW[dim]` by adding the product of the difference between the predicted value and the actual value (`f(self, sample) - y[sample]`) and the corresponding feature value (`self.X[sample][dim]`) divided by the number of samples (`self.m`).

If `spec` is `'b'`, the method initializes `dJdB` as 0. It iterates over the samples in the dataset and updates `dJdB` by adding the difference between the predicted value and the actual value divided by the number of samples.

Finally, the method returns the calculated derivative.

### Gradient Descent Optimization

```python
    def GradDecent(self, lr):
        self.w = self.w - lr * self.GetCostDerivation('w')
        self.b = self.b - lr * self.GetCostDerivation('b')
```

The `GradDecent` method performs one step of gradient descent optimization. It takes a learning rate (`lr`) as a parameter.

The method updates the weights `self.w` by subtracting the learning rate multiplied by the derivative of the cost function with respect to the weights (`self.GetCostDerivation('w')`). It also updates the bias term `self.b` in a similar way using the derivative of the cost function with respect to the bias term (`self.GetCostDerivation('b')`).

### Computing Linear Regression

```python
    def Compute(self, method, lr, n_iter):
        for _ in range(n_iter):
            method(lr)
```

The `Compute` method performs the linear regression calculation using gradient descent. It takes three parameters: `method`, `lr`, and `n_iter`. 

- `method` is a function that performs one step ofgradient descent optimization (in this case, `linreg.GradDecent`).
- `lr` is the learning rate, which determines the step size in each iteration of gradient descent.
- `n_iter` is the number of iterations to perform gradient descent.

The method iterates `n_iter` times and calls the `method` function (in this case, `linreg.GradDecent`) with the learning rate `lr` in each iteration.

### Example Usage

```python
# Define a simple dataset
X = np.array([[1,2], [2,4], [3,5], [4,3], [5,7]])
y = np.array([2, 3, 1, 4, 8])

# Create an instance of the LinReg class
linreg = LinReg(X, y)

# Perform gradient descent to find linear regression
linreg.Compute(linreg.GradDecent, lr=0.05, n_iter=1000)

# Print the final weights and bias
print("Weights:", linreg.w)
print("Bias:", linreg.b)
```

In this example, a simple dataset is defined with input features `X` and target values `y`. An instance of the `LinReg` class is created with the dataset.

The `Compute` method is called on the `linreg` instance with `linreg.GradDecent` as the optimization method, a learning rate of 0.05, and 1000 iterations. This performs gradient descent optimization to find the optimal weights and bias for linear regression.

Finally, the final weights and bias are printed, representing the learned linear regression model.

Output:
```
Weights: [1.07161243 0.25365249]
Bias: -0.6801122411924745
```

The final weights are approximately `[1.07, 0.25]` and the bias is approximately `-0.68`, indicating that the learned linear regression model is `y = 1.07*x_1 + 0.25*x_2 - 0.68`.
