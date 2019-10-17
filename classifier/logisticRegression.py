class LogisticRegression:
    def __init__(self, regulariser = 'l2', lmbda = 0.01, num_steps = 50, learning_rate = 0.01, initial_wts=None):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        self.regulariser = regulariser
        self.lmbda = lmbda
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.initial_wts = initial_wts
        return
    
    def sigmoid(self,z):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        z = np.array(z, dtype = 'float')
        t = np.array((1/(1 + np.exp(-1*z))), dtype = 'float')
        return t
    
    def normalize(self):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        maxi = float(np.max(self.initial_wts))
        mini = float(np.min(self.initial_wts))
        rangei = float(maxi - mini)
        normalized_initial_wts = (np.array(self.initial_wts) - mini)/rangei
        self.initial_wts = np.array(normalized_initial_wts, dtype = 'float')
        
    def updated_weights(self, X_tr, y_tr):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        X = np.ones(shape = (X_tr.shape[0],X_tr.shape[1]+1) ,dtype = 'float')
        X[:,1:] = X_tr
        y = np.array(y_tr)
        n = len(X_tr)
        
        h = self.sigmoid(np.dot(X,self.initial_wts))
        h =  h.reshape(y.shape[0])
        e = np.subtract(h,y)
        deriv = np.dot(np.transpose(X), e)
        deriv = deriv.reshape(deriv.shape[0],1)
       
        if self.regulariser == 'l1':
            self.initial_wts = self.initial_wts - self.learning_rate*(deriv + (self.lmbda*self.initial_wts))

        if self.regulariser == 'l2':
            self.initial_wts = self.initial_wts - self.learning_rate*(deriv + ((self.lmbda/2)*(self.initial_wts**2))) 
        return
    
    def train(self,X_tr, y_tr):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        if self.initial_wts is None:
            self.initial_wts = np.ones(shape=(X_tr.shape[1]+1,1))
        else:
            self.normalize()
        for i in range(self.num_steps):
            self.updated_weights(X_tr, y_tr)

    def predict(self, X_test):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        X = np.ones(shape = (X_test.shape[0],X_test.shape[1]+1) ,dtype = 'float')
        X[:,1:] = X_test
        pred = []
        t = self.sigmoid(np.dot(X,self.initial_wts))
        for j in t:
            if j > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)