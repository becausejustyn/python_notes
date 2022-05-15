# Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/student_scores.csv')
df.head(10)
```

- `Hours` is the feature variable (`X`)
- `Scores` is our target variable (`y`)

```python
# Taking the Hours and Scores column of the dataframe as X and y 
# respectively and coverting them to numpy arrays.
X = np.array(df['Hours']).reshape(-1, 1)
y = np.array(df['Scores'])

plt.figure(figsize = (8, 6))
plt.scatter(X, y)
plt.title('Hours vs Scores')
plt.xlabel('X (Input) : Hours')
plt.ylabel('y (Target) : Scores')
```

y_hat = wX + b

```python
y_hat = np.dot(X, weights) + bias
```

For two dimensions e.g. simple linear regression

- w: weights
   - slope of the line 
- b: bias
   - y-intercept 

Note: because this is two dimensional we fit the data with a line. If you had more than one feature variable, you would be using a plane instead of a line.

We want to find the parameters for our weights and biases that get y_hat to y as close as possible.

### Notation

- `n`: number of features
- `m`: number of training examples
- `X`: features
- `y`: target/label
- `(X(i), y(i))`: ith example in the training set

### Representation of X, y, w, and b

- `X` is a matrix of size `(m, n)`; 
   - rows represent the training examples 
   - columns represent the features
- `y` is a matrix of size `(m, 1)`
- each row is a label for the corresponding set of features in `X`
- `w` is a vector of size `(n, 1)`, and the parameter `b` is a scalar that can be broadcasted


If we had 3 features and 3 training examples, this is what the representation would look like `(m = 3, n = 3)`

$$
X_{j}^{i} \to j^{th} \text{ feature of } i^{th} \text{ training example }
$$



$$
\mathbf{{m}\big\updownarrow} =
\begin{bmatrix}
x_{1}^{1} \; x_{2}^{1} \; x_{3}^{1} \\ 
x_{1}^{2} \; x_{2}^{2} \; x_{3}^{2} \\
x_{1}^{3} \; x_{2}^{3} \; x_{3}^{3}
\end{bmatrix}_{\; m \times n}
\;\;
\mathbf{w} =
\begin{bmatrix}
w_{1} \\ w_{2} \\ w_{3}
\end{bmatrix}_{\; n \times 1}
\;\;
\mathbf{b} =
\begin{bmatrix}
\vdots \\ b \\ \vdots
\end{bmatrix}
$$

```python
X.shape, y.shape
```

### Loss/Cost Function

$$
J = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}-y)^{2} \\
\text{Mean Squared Error}
$$

```python
loss = np.mean((y_hat - y)**2)
```

`https://medium.com/analytics-vidhya/linear-regression-from-scratch-in-python-b6501f91c82d`

`https://towardsai.net/p/machine-learning/closed-form-and-gradient-descent-regression-explained-with-python`