#sample_submission.py
import numpy as np
import sys
class regressor(object):
    """
    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data):
        self.x, self.y = data
        tempX=np.c_[ np.ones(len(self.x)),self.x]
        learningstep=.00001
        W=np.random.random((tempX.shape[1],1))
        validsize=int(.2*len(self.x))
        validsetY=self.y[0:validsize,:]
        trainY = self.y[validsize:, :]
        validsetX=tempX[0:validsize,:]
        trainX=tempX[validsize:,:]
        previouserror=sys.maxint
        count=0
        while True:
            prevW=W
            error=np.add(np.dot(-2,np.dot(trainX.T,trainY)),np.dot(2,np.dot(np.dot(trainX.T,trainX),W)))
            W=np.subtract(W,np.dot(learningstep,error))
            predY = np.dot(W.T,validsetX.T)
            predY=predY.T
            currenterror=(np.mean((validsetY- predY) ** 2))/2
            if currenterror > previouserror:
                    if count==0:
                        finalW=prevW
                    count=count+1
                    if count>=3:
                        break
            else:
                count=0
                previouserror = currenterror
        W=finalW
        self.w = W[1:,:]
        self.b = W[0,0]
        
    def get_params (self):
        """
        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a W and b value of the trained linear regression model.

        """
        return (self.w, self.b)

    def get_predictions (self, x):
        """
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of test data x
        """
        tempX = np.c_[np.ones(len(x)),x]
        W= np.insert(self.w, 0, self.b, axis=0)
        Y=np.dot(W.T,tempX.T)
        return Y.T

if __name__ == '__main__':
    pass 
