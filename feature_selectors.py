#imports
# !pip install category_encoders
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
import category_encoders as ce
from skrebate import ReliefF
from tqdm import tqdm
import pandas as  pd
import numpy as np
import functools
import itertools
import warnings
import random
import math
warnings.filterwarnings("ignore")
FLOOR = .001

class MRMR():

    def __init__(self, labels, TF_IDF_vector, k= 1000):
        self.y = labels
        self.TF_IDF_vector = TF_IDF_vector
        self.K = k
    
    def dataset(self):
        scaler = StandardScaler(with_mean=False)
        X_tf_idf = scaler.fit_transform(self.TF_IDF_vector)
        X_df = pd.DataFrame(X_tf_idf)
        y_df = pd.DataFrame(self.y)
        return X_df, y_df

    
    def groupstats2fstat(self,avg, var, n):

        avg_global = (avg * n).sum() / n.sum()  # global average of each variable
        numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
        denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
        f = numerator / denominator
        return f.fillna(0.0)


    def mrmr_base(self, K, relevance_func, redundancy_func,
                relevance_args={}, redundancy_args={},
                denominator_func=np.mean, only_same_domain=False,
                return_scores=False, show_progress=True):
    
        relevance = relevance_func(**relevance_args)
        features = relevance[relevance.fillna(0) > 0].index.to_list()
        relevance = relevance.loc[features]
        redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
        K = min(K, len(features))
        selected_features = []
        not_selected_features = features.copy()
        score_selected = []
        selected = []
        for i in tqdm(range(K), disable=not show_progress):

            score_numerator = relevance.loc[not_selected_features]

            if i > 0:

                last_selected_feature = selected_features[-1]

                if only_same_domain:
                    not_selected_features_sub = [c for c in not_selected_features if
                                                c.split('_')[0] == last_selected_feature.split('_')[0]]
                else:
                    not_selected_features_sub = not_selected_features

                if not_selected_features_sub:
                    redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                        target_column=last_selected_feature,
                        features=not_selected_features_sub,
                        **redundancy_args
                    ).fillna(FLOOR).abs().clip(FLOOR)
                    score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                        denominator_func, axis=1).replace(1.0, float('Inf'))

            else:
                score_denominator = pd.Series(1, index=features)

            score = score_numerator / score_denominator

            best_feature = score.index[score.argmax()]
            selected_features.append(best_feature)
            score_selected.append(score[best_feature])
            not_selected_features.remove(best_feature)
            
        for o in range(0,len(selected_features)):
            selected.append([selected_features[o],score_selected[o]])

        if not return_scores:
            return selected
        else:
            return (selected_features, relevance, redundancy)

    def parallel_df(self,func, df, series, n_jobs):
        n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
        col_chunks = np.array_split(range(len(df.columns)), n_jobs)
        lst = Parallel(n_jobs=n_jobs)(
            delayed(func)(df.iloc[:, col_chunk], series)
            for col_chunk in col_chunks
        )
        return pd.concat(lst)

    def _f_classif(self, X, y):
        def _f_classif_series(x, y):
            x_not_na = ~ x.isna()
            if x_not_na.sum() == 0:
                return 0
            return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

        return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)

    def _f_regression(self, X, y):
        def _f_regression_series(x, y):
            x_not_na = ~ x.isna()
            if x_not_na.sum() == 0:
                return 0
            return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

        return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)

    def f_classif(self, X, y, n_jobs):
        return parallel_df(_f_classif, X, y, n_jobs=n_jobs)

    def f_regression(self, X, y, n_jobs):
        return parallel_df(_f_regression, X, y, n_jobs=n_jobs)

    def _ks_classif(self, X, y):
        def _ks_classif_series(x, y):
            x_not_na = ~ x.isna()
            if x_not_na.sum() == 0:
                return 0
            x = x[x_not_na]
            y = y[x_not_na]
            return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

        return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)

    def ks_classif(self, X, y, n_jobs):
        return parallel_df(_ks_classif, X, y, n_jobs=n_jobs)

    def random_forest_classif(self, X, y):
        forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
        return pd.Series(forest.feature_importances_, index=X.columns)

    def random_forest_regression(self, X, y):
        forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
        return pd.Series(forest.feature_importances_, index=X.columns)


    def correlation(self, target_column, features, X, n_jobs):
        
        def _correlation(X, y):
            return X.corrwith(y).fillna(0.0)

        return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)

    def mrmr_classif(
            self,X, y, K,
            relevance='f', redundancy='c', denominator='mean',
            cat_features=None, cat_encoding='leave_one_out',
            only_same_domain=False, return_scores=False,
            n_jobs=-1, show_progress=True
    ):
        if relevance == "f":
            relevance_func = functools.partial(f_classif, n_jobs=n_jobs)
        else:
            relevance_func = relevance

        redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
        denominator_func = np.mean if denominator == 'mean' else (
            np.max if denominator == 'max' else denominator)

        relevance_args = {'X': X, 'y': y}
        redundancy_args = {'X': X}

        return mrmr_base(self, K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                        relevance_args=relevance_args, redundancy_args=redundancy_args,
                        denominator_func=denominator_func, only_same_domain=only_same_domain,
                        return_scores=return_scores, show_progress=show_progress)


    def feature_selection(self):
        X_df, y_df = self.dataset()
        X = pd.DataFrame(X_df)
        y = pd.DataFrame(y_df)
        # use mrmr classification
        selected_features = mrmr_classif(X=X , y=y , K = self.K)
        features_scores = pd.DataFrame([selected_features[x][1] for x in range(0,len(selected_features))])
        cumulativesum = features_scores.cumsum()
        mrmrscore = cumulativesum / features_scores.sum()
        mrmr_score = mrmrscore.loc[mrmrscore[0] <= 0.9]
        del selected_features[len(mrmr_score):]
        return selected_features

class CFS():
    
    def __init__(self, TF_IDF_vector, k= 1000):
        self.TF_IDF_vector = TF_IDF_vector
        self.K = k
    def feature_selection(self):
        corr_matrix = np.corrcoef(self.TF_IDF_vector, rowvar=False)
        selected_features = np.argsort(np.sum(corr_matrix, axis=1))[-self.K:]
        selected_features = sorted(list(selected_features), key=lambda x: -np.sum(corr_matrix[x,:]))
        return selected_features

class relieff():
    
    def __init__(self, TF_IDF_vector, labels, n_neighbors=20, k= 1000):
        self.y = labels
        self.n_neighbors = n_neighbors
        self.TF_IDF_vector = TF_IDF_vector
        self.K = k
    def feature_selection(self):
        fs = ReliefF(n_neighbors=self.n_neighbors, verbose=True, n_jobs=-1)
        X_relief = fs.fit_transform(self.TF_IDF_vector, self.y)
        top_k_idx = np.argsort(fs.feature_importances_)[::-1][:self.K]
        return top_k_idx


