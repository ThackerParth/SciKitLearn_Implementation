class AdaBoost():
    def __init__(self, n_trees, learning_rate):
        import pandas as pd
        import numpy as np
        import scipy as sp
        from classifier.decisionTree import DecisionTree

        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.base_estimator = DecisionTree()
        
    def fit(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp
        from classifier.decisionTree import DecisionTree
        self.n = X.shape[0]
        classes = list(set(y))
        sort_Class = np.array(sorted(classes))
        self.labels = sort_Class
        for i in range(self.n_trees):
            if i==0:
                s_wt = np.ones(self.n)/self.n
            s_wt, e_wt, e_error = self.adaboost(X, y, s_wt)
            self.errors[i] = e_error
            self.eweight[i] = e_wt
        return self
    
    def adaboost(self, X, y, s_wt):
        import pandas as pd
        import numpy as np
        import scipy as sp
        from classifier.decisionTree import DecisionTree
        clf = deepcopy(self.base_estimator)
        clf.train(X,y, s_wt)
        y_p = clf.predict(X)
        misclassified = (y_p != y)
        error_Rate = np.dot(misclassified*s_wt)/np.sum(s_wt, axis=0)
        e_wt = self.learning_rate * np.log((1 - error_rate) / error_rate) + np.log(self.n - 1)
        s_wt *= np.exp(e_wt * misclassified)
        s_wt/=np.sum(s_wt, axis=0)
        self.classifiers.append(clf)
        return s_wt, e_wt, e_error
    
    def predict(self, X):
        import pandas as pd
        import numpy as np
        import scipy as sp
        from classifier.decisionTree import DecisionTree
        pd=[]
        lab = self.labels[:, np.newaxis]
        for clf, wt in zip(self.classifiers,self.eweight):
            pd = sum((clf.predict(X) == lab).T * wt)
        pd/=self.eweights.sum()
        if self.n == 2:
            pd[:, 0] *= -1
            pd = pd.sum(axis=1)
            return self.labels.take(pd > 0, axis=0)

        return self.labels.take(np.argmax(pd, axis=1), axis=0)