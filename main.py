{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "1-pQtk0kXplK",
    "outputId": "6298fed7-d08f-42fb-9768-2b978e34f5ac"
   },
   "outputs": [],
   "source": [
    "# from google.colab import drive \n",
    "# drive.mount('/content/drive/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "SpLaC7lTgWiF"
   },
   "outputs": [],
   "source": [
    "#imports\n",
    "# !pip install category_encoders\n",
    "from sklearn.feature_selection import f_regression as sklearn_f_regression\n",
    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
    "from sklearn.feature_selection import f_classif as sklearn_f_classif\n",
    "from sklearn.feature_extraction.text import TfidfTransformer\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_validate\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from multiprocessing import cpu_count\n",
    "from joblib import Parallel, delayed\n",
    "from sklearn import preprocessing\n",
    "from scipy.stats import ks_2samp\n",
    "import category_encoders as ce\n",
    "from skrebate import ReliefF\n",
    "from tqdm import tqdm\n",
    "import pandas as  pd\n",
    "import numpy as np\n",
    "import functools\n",
    "import itertools\n",
    "import warnings\n",
    "import random\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Here we should call feature selectors algorithms, fitness algorithm, laying chicken algorithm and Datasets to run \n",
    "# complete clear code will release after publication of the article"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "RvlSPCbJXOUX"
   },
   "outputs": [],
   "source": [
    "%%time\n",
    "#test with mrmr\n",
    "filter = MRMR(label, TF_IDF_vector, k =len(TF_IDF_vector[0]))\n",
    "selected_features = filter.feature_selection()\n",
    "mrmr_features = [selected_features[x][0] for x in range(0,len(selected_features))]\n",
    "with open(\"mrmr_features.txt\", \"w\") as f:\n",
    "    for s in mrmr_features:\n",
    "        f.write(str(s) +\"\\n\")\n",
    "mrmr_features = []\n",
    "with open(\"mrmr_features_aso.txt\", \"r\") as f:\n",
    "    for line in f:\n",
    "        mrmr_features.append(int(line.strip()))\n",
    "del mrmr_features[1000:]\n",
    "print(len(mrmr_features))\n",
    "laying_chicken = Laying(Bfeatures=mrmr_features,TF_IDF_Vec=TF_IDF_vector, labels=label, k=12 , alpha=8)\n",
    "final_best_mrmr = laying_chicken.algorithm()\n",
    "del (laying_chicken)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "# test with reliefF\n",
    "reli_selection = relieff (TF_IDF_vector=TF_IDF_vector, labels=label, k=1000)\n",
    "rf_feature_selection = reli_selection.feature_selection()\n",
    "with open(\"ASO_reliefF_features.txt\", \"w\") as f:\n",
    "    for s in rf_feature_selection:\n",
    "        f.write(str(s) +\"\\n\")\n",
    "reli_features = []\n",
    "with open(\"ASO_reliefF_features.txt\", \"r\") as f:\n",
    "    for line in f:\n",
    "        reli_features.append(int(line.strip()))\n",
    "del reli_features[1000:]\n",
    "print(len(reli_features))\n",
    "laying_chicken = Laying(Bfeatures=reli_features,TF_IDF_Vec=TF_IDF_vector, labels=label, k=12 , alpha=8)\n",
    "final_best_reli = laying_chicken.algorithm()\n",
    "del (laying_chicken)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "# test with CFS\n",
    "cfs_selection = CFS(TF_IDF_vector)\n",
    "CFS_feature_selection = cfs_selection.feature_selection()\n",
    "with open(\"ASO_CFS_features.txt\", \"w\") as f:\n",
    "    for s in CFS_feature_selection:\n",
    "        f.write(str(s) +\"\\n\")\n",
    "CFS_features = []\n",
    "with open(\"ASO_CFS_features.txt\", \"r\") as f:\n",
    "    for line in f:\n",
    "        CFS_features.append(int(line.strip()))\n",
    "del CFS_features[1000:]\n",
    "print(len(CFS_features))\n",
    "laying_chicken = Laying(Bfeatures=CFS_features,TF_IDF_Vec=TF_IDF_vector, labels=label, k=12 , alpha=8)\n",
    "final_best_CFS = laying_chicken.algorithm()\n",
    "del (laying_chicken)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bests = [final_best_mrmr,final_best_reli,final_best_CFS]\n",
    "b = [0,0]\n",
    "for best in bests:\n",
    "    if best[1] > b[1] :\n",
    "        final_best = best"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "gQyU-iCbdo-l"
   },
   "outputs": [],
   "source": [
    "df = pd.DataFrame(final_best) \n",
    "df = df.rename(columns={0:\"best_sulotion\"})\n",
    "df.to_csv('best_multistart_TFIDF1000_@8_K12_5fold.csv',index=False)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [
    "a2JSt0_GQuTs",
    "w8XsATrMBgj6",
    "bd8PCVYAJuHx",
    "rb3wesJAJ5fx"
   ],
   "machine_shape": "hm",
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
