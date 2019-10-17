class Stack:
    def __init__(self, args = []):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        self.allclf = []
        self.args = args
        for i in self.args:
            for j in range(i[1]):
                self.allclf.append(copy.deepcopy(i[0]))
        return
    
    def train(self, X_tr, y_tr):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        for model in self.allclf:
            model.train(X_tr,y_tr)
            x = np.ndarray(shape=(X_tr.shape[0], X_tr.shape[1]+1), dtype = 'float')
            x[:,:-1] = X_tr
            x[:,-1] = model.predict(X_tr)
            X_tr = x
            
    def predict(self, x_test):
        import numpy as np
        import pandas as pd
        import scipy as sp
        import copy
        for model in self.allclf:
           
            x = np.ndarray(shape=(x_test.shape[0], x_test.shape[1]+1), dtype = 'float')
            x[:,:-1] = x_test
            x[:,-1] = model.predict(x_test)
            x_test = x
        return x_test[:,-1]