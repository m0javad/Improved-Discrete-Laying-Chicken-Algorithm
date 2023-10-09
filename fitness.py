
class Fitness():

    def __init__(self, ZFeatures, TF_IDF_Vec, label, method = 'svm', score = 'precision_macro', n_neighbors = 5):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.y_encode = label #le.fit_transform(label)
        self.method = method #'naive_bayes', 'svm', 'decisionTree', 'KNN'
        self.score = score #'precision_macro', 'recall_macro','f1_weighted'
        self.n_neighbors = n_neighbors
        self.initial = np.delete(TF_IDF_Vec, ZFeatures, 1) #from  TF_IDF_Vec delete Features 0 

    def naive_bayes(self):
        from sklearn.naive_bayes import GaussianNB
        scoring = [self.score]
        clf = GaussianNB()
        scores = cross_validate(clf, self.initial, self.y_encode, cv=5 , scoring= scoring ) 
        score_method = 'test_'+ self.score
        return sum(scores[score_method])/5

    def svm(self):
        from sklearn.svm import SVC
        scoring = [self.score]
        clf = SVC(C = 100 , kernel="poly" , degree=1)
        scores = cross_validate(clf, self.initial, self.y_encode, cv=2 , scoring= scoring ) 
        score_method = 'test_'+ self.score
        return sum(scores[score_method])/2

    def decisionTree(self):
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        scoring = [self.score]
        scores = cross_validate(clf, self.initial, self.y_encode, cv=5 , scoring= scoring ) 
        score_method = 'test_'+ self.score
        return sum(scores[score_method])/5

    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(self.n_neighbors)
        scoring = [self.score]
        scores = cross_validate(clf, self.initial, self.y_encode, cv=5 , scoring= scoring ) 
        score_method = 'test_'+ self.score
        return sum(scores[score_method])/5

    def fitt(self):
        if self.method == 'naive_bayes':
            fit = self.naive_bayes()
        elif self.method == 'svm':
            fit = self.svm()
        elif self.method == 'decisionTree':
            fit = self.decisionTree()
        elif self.method == 'KNN':
            fit = self.KNN()
        return fit