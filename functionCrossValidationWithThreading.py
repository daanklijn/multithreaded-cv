def cross_val_function(function,x,y,folds,*args,**kwargs):
	"""
	This is an implementation of K-folds cross validation. Which is used to gain an accurate performance measure
	for a certain model or estimator. This estimator or model should be in form of a function which returns a
	score. The cross_val_function creates the given number of folds and then saves the score in an array for each
	of these folds. For each of the folds the function creates a separate thread to make optimal use of the
	processor.

    Parameters
    ----------
    function : function
        Function that returns a score with type float or integer.
        NOTE: Parameters of function should be added to the cross_val_function's parameters
    x        : numpy array
        Numpy array that contains the predictor variables.
    y        : numpy array
        Numpy array that contains the target variable.
    folds    : integer
        Number of folds used in cross validation.
	*args    : 
        Parameters of function
	*kwargs  : 
        Parameters of function
    
	
    Returns
    -------
    cross_val_scores : array
        Array containing the cross validation scores of all folds.
    """
	import numpy as np
	from threading import Thread
	
	num_rows=x.shape[0]
	fold_size=round(num_rows/folds)
	cross_val_scores=[]
	threads=[]

	def calculate_and_append_score(train_x,train_y,test_x,test_y):
		score=function(train_x,train_y,test_x,test_y,*args,**kwargs)
		cross_val_scores.append(score)

	for i in range(0,num_rows,fold_size):
		endIndex=i+fold_size if i+fold_size<=num_rows else num_rows
		indices=range(i,endIndex)
		mask = np.ones(num_rows,dtype=bool)
		mask[indices] = False
		train_x=x[mask]
		train_y=y[mask]
		test_x=x[~mask]
		test_y=y[~mask]
		thread=Thread(target=calculate_and_append_score, args=(train_x,train_y,test_x,test_y))
		thread.start()
		threads.append(thread)

	for thread in threads:
		thread.join()

	return cross_val_scores
