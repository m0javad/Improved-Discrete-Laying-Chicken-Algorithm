#imports
# !pip install category_encoders
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn import preprocessing
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

%%time
#test with mrmr
# filter = MRMR(label, TF_IDF_vector, k =len(TF_IDF_vector[0]))
# selected_features = filter.feature_selection()
# mrmr_features = [selected_features[x][0] for x in range(0,len(selected_features))]
# with open("mrmr_features.txt", "w") as f:
#     for s in mrmr_features:
#         f.write(str(s) +"\n")
mrmr_features = []
with open("20news_mrmr_stopwords_features.txt", "r") as f:
    for line in f:
        mrmr_features.append(int(line.strip()))
del mrmr_features[1000:]
print(len(mrmr_features))
laying_chicken = Laying(Bfeatures=mrmr_features,TF_IDF_Vec=TF_IDF_vector, labels=label, k=12 , alpha=8,start="MRMR",dataset="20news")
final_best_mrmr = laying_chicken.algorithm()
del (laying_chicken)

bests = [final_best_mrmr,final_best_reli,final_best_CFS]
b = [0,0]
for best in bests:
    if best[1] > b[1] :
        final_best = best

df = pd.DataFrame(final_best) 
df = df.rename(columns={0:"best_sulotion"})
df.to_csv('best_multistart_TFIDF1000_@8_K12_5fold.csv',index=False)
