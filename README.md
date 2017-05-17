# multithreaded-cv
K-Folds Cross Validation based on a inputted function with implementation of multithreading.

 - - - -

The function can be simply called using the following syntax.
```python
cross_val_function(function,x,y,folds)
```
The parameters function, x, y and folds are described below.

**function**  
Function that returns a score with type float or integer.

**x**  
Numpy array that contains the predictor variables.

**y**  
Numpy array that contains the target variable.

**folds**  
Number of folds used in cross validation.

**What to do when your function has parameters?**  
You can simply add your function's parameters to the parameters of the cross_val_function.  
For instance we want to cross validate the following function:
```python
def regression(alpha)
```
We can cross validate this function using alpha=1 like this:
```python
cross_val_function(regression,x,y,folds,alpha=1)
```
