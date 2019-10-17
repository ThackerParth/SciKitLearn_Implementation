class DecisionTree:
    def __init__(self,max_depth = 10000, split_val_metric = 'median',min_info_gain = 1e-7,split_node_criterion = 'gini', depth = 0):
        import pandas as pd
        import numpy as np
        import scipy as sy
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain
        self.split_node_criterion = split_node_criterion
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.depth = depth
        self.target = None
        #print(self.max_depth, self.split_val_metric, self.min_info_gain, self.split_node_criterion, self.depth)
        return
    
    def _gini_ind(self, col):
        import pandas as pd
        import numpy as np
        import scipy as sy
        if len(col) == 0:
            return 1
        total = float(len(col))
        classes, counts = np.unique(col, return_counts=True)
        c_counts = dict(zip(classes, counts))
        if len(classes) == 0:
            return 1
        if len(classes) == 1:
            return 0
        gini = 1.0
        for cla in c_counts.keys():
            p = float(c_counts.get(cla) / total)
            gini += float(-1.0*p*p)
        
        return float(gini)
    
    def _entropy(self, col):
        import pandas as pd
        import numpy as np
        import scipy as sy
        if len(col) == 0:
            return 0
        total = float(len(col))
        classes, counts = np.unique(col, return_counts=True)
        c_counts = dict(zip(classes, counts))
        if len(classes) == 0:
            return 0
        if len(classes) == 1:
            return 0
        ent = 0.0
        for cla in c_counts.keys():
            p = float(c_counts.get(cla) / total)
            ent += float(-1.0 * p * np.log2(p))
            
        return float(ent)
    
    def split_data(self, data, fkey, fval):
        import pandas as pd
        import numpy as np
        import scipy as sy
        left = list()
        right = list()
        
        if len(data) == 0:
             return np.array(left, dtype = 'float'), np.array(right, dtype = 'float')
        
        for row in data:
            try:
                val = row[fkey]
            except:
                print (row)
                return IndexError
            
            if val > fval:
                right.append(row)
            else:
                left.append(row)
        
        return np.array(left, dtype = 'float'), np.array(right, dtype = 'float')
    
    def information_gain(self, data, fkey, fval):
        import pandas as pd
        import numpy as np
        import scipy as sy
        try:
            left, right = self.split_data(data, fkey, fval)
        except:
            return IndexError
        
        if left.shape[0] == 0 or right.shape[0] == 0:
            return -10000
        total = float(len(data))
        ig = 0.0
        if self.split_node_criterion == 'gini':
            ig += float(self._gini_ind(data[:,-1]))
            ig -= float(self._gini_ind(left[:,-1]) * float(left.shape[0]) / total)
            ig -= float(self._gini_ind(right[:,-1]) * float(right.shape[0]) / total)
        elif self.split_node_criterion == 'entropy':
            ig += float(self._entropy(data[:,-1]))
            ig -= float(self._entropy(left[:,-1]) * float(left.shape[0]) / total)
            ig -= float(self._entropy(right[:,-1]) * float(right.shape[0]) / total)
            
        return float(ig)
    
    def train(self, x_train, y_train):
        import pandas as pd
        import numpy as np
        import scipy as sy
       
        if x_train.shape[0] != y_train.shape[0]:
            return IndexError
        if x_train.shape[0] == 0:
            return
        x_train = np.array(x_train, dtype = 'float')
        y_train = np.array(y_train, dtype= 'float')
        data = np.ndarray(shape=(x_train.shape[0],x_train.shape[1]+1), dtype='float')
        data[:,:-1] = x_train
        data[:,-1] = y_train
        gains=[]
        if self.split_val_metric == 'mean':
            for i in range(data.shape[1]-1):
                gains.append(self.information_gain(data, i, np.mean(data[:,i])))
        elif self.split_val_metric == 'median':
            for i in range(data.shape[1]-1):
                gains.append(self.information_gain(data, i, np.median(data[:,i])))
        
        self.fkey = np.argmax(gains)
        #print(gains)
        ig = np.max(np.array(gains, dtype='float'))
        #print(self.fkey,ig)
        if self.split_val_metric == 'mean':
            self.fval = np.mean(data[:,self.fkey])
        elif self.split_val_metric == 'median':
            self.fval = np.median(data[:,self.fkey])
        
        left, right = self.split_data(data, self.fkey, self.fval)
        
        if left.shape[0] == 0 or right.shape[0] == 0:
            classes, counts = np.unique(data[:,-1], return_counts=True)
            c_counts = dict(zip(classes, counts))
            self.target = max(c_counts, key=c_counts.get)
            return
        if not(self.max_depth is None):
            if self.depth >= self.max_depth:
                classes, counts = np.unique(data[:,-1], return_counts=True)
                c_counts = dict(zip(classes, counts))
                self.target = max(c_counts, key=c_counts.get)
                return
        
        if ig < float(self.min_info_gain):
            classes, counts = np.unique(data[:,-1], return_counts=True)
            c_counts = dict(zip(classes, counts))
            self.target = max(c_counts, key=c_counts.get)
            return
        
        self.left = DecisionTree(max_depth = self.max_depth, split_val_metric = self.split_val_metric, min_info_gain = self.min_info_gain, split_node_criterion = self.split_node_criterion, depth = self.depth+1)
        self.left.train(left[:,:-1],left[:,-1])
        
        self.right = DecisionTree(max_depth = self.max_depth, split_val_metric = self.split_val_metric, min_info_gain = self.min_info_gain, split_node_criterion = self.split_node_criterion, depth = self.depth+1)
        self.right.train(right[:,:-1],right[:,-1])
        
        classes, counts = np.unique(data[:,-1], return_counts=True)
        c_counts = dict(zip(classes, counts))
        self.target = max(c_counts, key=c_counts.get)
        #print(self.fkey, self.fval)
        return
    
    def _predict_class(self, x_test):
        import pandas as pd
        import numpy as np
        import scipy as sy
        if x_test[self.fkey] > self.fval:
            if self.right is None:
                return self.target
            return self.right._predict_class(x_test)
            
        else:
            if self.left is None:
                return self.target
            return self.left._predict_class(x_test)
    
    def predict(self, X_test):
        import pandas as pd
        import numpy as np
        import scipy as sy
        pred = []
        X_test = np.array(X_test,dtype = 'float')
        for x_test in X_test:
            pred.append(float(self._predict_class(x_test)))
        return np.array(pred, dtype='float')