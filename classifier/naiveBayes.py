class NaiveBayes():
    def __init__(self, typec, prior):
        import pandas as pd
        import numpy as np
        import scipy as sp

        self.typec = typec
        self.prior = prior
    
    def mean(self, values):
        import pandas as pd
        import numpy as np
        import scipy as sp

        sumv = np.sum(values, dtype = np.float32)
        return float(sumv/float(len(values)))
    
    def variance(self, values):
        import pandas as pd
        import numpy as np
        import scipy as sp

        avg = self.mean(values)
        num = np.sum(pow(values-avg, 2))
        return float(num/float(len(values)-1))
    
    def train(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp

        Xarr = np.array(X)
        yarr = np.array(y)
        if self.typec == 'Gaussian':
            self.GaussianNB(Xarr, yarr)
        else:
            self.MultinomialNB(Xarr, yarr)
    
    def DatabyClass(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp

        divclass = dict()
        for i in range(len(y)):
            #print(y)
            if y[i][0] not in divclass:
                divclass[y[i][0]] = []
            divclass[y[i][0]].append(X[i])
        for label, instances in divclass.items():
            divclass[label] = np.array(instances)
        return divclass
    
    def Counts(self, values):
        import pandas as pd
        import numpy as np
        import scipy as sp

        unique, counts = np.unique(values, return_counts=True)
        dic = dict(zip(unique, counts))
        return dict(zip(unique, counts))
    
    def CountVals(self, instances):
        import pandas as pd
        import numpy as np
        import scipy as sp

        vals = dict()
        for j in range(len(instances[0])):
            values = []
            vals[j] = []
            for i in range(len(instances)):
                values.append(instances[i][j])
            dic = self.Counts(values)
            for i in range(self.minnum[j], self.maxnum[j]+1):
                if i not in dic:
                    dic[i]=0
                dic[i]+=1
            vals[j].append(dic)
        #print(vals)
        return vals
    
    def CalcClassprob(self, ClassDiv):
        import pandas as pd
        import numpy as np
        import scipy as sp

        probab = dict()
        values = []
        for label, instances in ClassDiv.items():
            probab[label] = self.CountVals(instances)
        #print(probab)
        #print('HERE')
        return probab
    
    def MultinomialNB(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp

        self.ClassDiv = self.DatabyClass(X, y)
        self.maxnum = np.amax(X, axis=0)
        self.minnum = np.amin(X, axis=0)
            
        self.probabilities = self.CalcClassprob(self.ClassDiv)
     
    def calcPred(self, row):
        import pandas as pd
        import numpy as np
        import scipy as sp

        cl =0
        ma = 0
        #print(self.probabilities)
        for label, instances in self.probabilities.items():
            prob = 1;
            n = len(self.ClassDiv[label])
            for lab, ins in instances.items():
                #print(ins)
                #print(row)
                count = ins[0][np.array(row)[int(lab)]]
                #print(count)
                prob *= float(count/float(n))
            prob*= self.prior[label]
            #print(prob, label)
            if prob > ma:
                ma = prob
                cl = label
        return cl
    
    
    def predict(self, X_test):
        import pandas as pd
        import numpy as np
        import scipy as sp

        yp = []
        X_t = np.array(X_test)
        if self.typec == 'Multinomial':
            for i in range(len(X_t)):
                yp.append(self.calcPred(X_t[i]))
        else:
            for i in range(len(X_t)):
                yp.append(self.calcPredGNB(X_t[i]))
            
        return yp
    
    def GaussianNB(self, X, y):
        import pandas as pd
        import numpy as np
        import scipy as sp

        self.ClassDiv = self.DatabyClass(X, y)
        self.probabilities = self.GNBProb(self.ClassDiv)
    
    def AttrProb(self, instances):
        import pandas as pd
        import numpy as np
        import scipy as sp

        atprob = dict()
        for j in range(len(instances[0])):
            values = []
            atprob[j] = []
            for i in range(len(instances)):
                values.append(instances[i][j])
            avg = self.mean(np.array(values))
            var = self.variance(np.array(values))
            atprob[j].append(avg)
            atprob[j].append(var)
        return atprob
    
    def GNBProb(self, ClassDiv):
        import pandas as pd
        import numpy as np
        import scipy as sp

        prob = dict()
        for label, instances in ClassDiv.items():
            prob[label] = self.AttrProb(instances)
        return prob
    
    def calcPGNB(self, x, avg, var):
        import pandas as pd
        import numpy as np
        import scipy as sp

        num = float(np.exp(-(np.power(x-avg, 2))/float(2*var)))
        den = float(1/float(np.sqrt(float(2*np.pi*var))))
        return num*den
    
    def calcPredGNB(self, row):
        import pandas as pd
        import numpy as np
        import scipy as sp

        ma = 0
        cl =0
        row = np.array(row)
        for label, instances in self.probabilities.items():
            prob=1
            for lab, ins in instances.items():
                avg, var = instances[lab]
                prob*= self.calcPGNB(row[int(lab)], avg, var)
            #print(prob, label)
            prob*=self.prior[label]
            if prob>ma:
                ma = prob
                cl = label
        return cl