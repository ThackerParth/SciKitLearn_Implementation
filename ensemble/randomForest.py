class RandomForest:

    def __init__(self, n_trees = 100, max_depth = 10000, split_val_metric = 'median',min_info_gain = 1e-7,split_node_criterion = 'gini', max_features = "auto", bootstrap=True, n_cores = 1, random_state = None):
        import pandas as pd
        import numpy as np
        import scipy as sp
        import multiprocessing as mp
        from classifier.decisionTree import DecisionTree
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain
        self.split_node_criterion = split_node_criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_cores = n_cores
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.trees = []
        return
    def _train_dt(self,X, y, dtree, features_to_select, q):
        import pandas as pd
        import numpy as np
        import scipy as sp
        import multiprocessing as mp
        from classifier.decisionTree import DecisionTree
        dtree.train(X,y)
        q.put((dtree,features_to_select))
    def _set_max_features(self):
        import pandas as pd
        import numpy as np
        import scipy as sp
        import multiprocessing as mp
        from classifier.decisionTree import DecisionTree
        if type(self.max_features) is type(0.5):
            self.max_features = int(self.max_features * self.no_features)
            if self.max_features > self.no_features:
                self.max_features = int(self.no_features)
        elif self.max_features == "auto":
            self.max_features = int(np.sqrt(self.no_features))
        elif self.max_features == "sqrt":
            self.max_features = int(np.sqrt(self.no_features))
        elif self.max_features == "log2":
            self.max_features = int(np.log2(self.no_features))
        elif self.max_features is None:
            self.max_features = int(self.no_features)
        
        elif type(self.max_features) is type(1):
            if self.max_features > self.no_features:
                self.max_features = int(self.no_features)
        
    
    
    def train(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp
        import multiprocessing as mp
        from classifier.decisionTree import DecisionTree
        X = np.array(X, dtype = 'float')
        y = np.array(y, dtype = 'float')
        self.no_features = int(X.shape[1])
        
        self._set_max_features()
        treess = []
        p = mp.current_process()
        if not(__name__ == "__main__"):
            
            i = 0
            while (i<self.n_trees):
                q = mp.Queue()
                jobs = []
                j = 0
                while(i<self.n_trees and j<self.n_cores):
                    rows_to_select = list(np.random.choice(len(X), len(X), replace = (self.bootstrap)))
                    features_to_select = list(np.random.choice(self.no_features, self.max_features, replace = False))
                    new_X = X[rows_to_select,:][:,features_to_select]
                    new_y = y[rows_to_select]
                    jobs.append(mp.Process(target=self._train_dt, args=(new_X, new_y, DecisionTree(max_depth=self.max_depth, split_val_metric = self.split_val_metric, min_info_gain = self.min_info_gain, split_node_criterion = self.split_node_criterion, depth = 0),features_to_select, q)))
                    i+=1
                    j+=1
                
                for f in jobs:
                    f.start()
                    
                for k in range(self.n_cores):
                    treess.append(q.get())
                    
                for f in jobs:
                    f.join()
                    
        
        self.trees = treess
        print(len(self.trees))
            
    def predict(self, X_test):
        import pandas as pd
        import numpy as np
        import scipy as sp
        import multiprocessing as mp
        from classifier.decisionTree import DecisionTree
        X_test = np.array(X_test)
        pred_all = list()
        for (tree,features_to_select) in self.trees:
            X_t = X_test[:,features_to_select]
            pred_all.append(tree.predict(X_t))
        pred_all = np.array(pred_all)
        pred = list()
        for i in range(pred_all.shape[1]):
            classes, counts = np.unique(pred_all[:,i], return_counts=True)
            c_counts = dict(zip(classes, counts))
            pred.append(max(c_counts, key=c_counts.get))
        return np.array(pred)
        