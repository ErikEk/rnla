import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats
import sklearn.datasets
import sklearn.preprocessing


iris = sklearn.datasets.load_iris()

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

print('Iris dataset has {} rows and {} columns\n'.format(*df_iris.shape))

print('Here are the first 5 rows of the data:\n\n{}\n'.format(df_iris.head(5)))

print('Some simple statistics on the Iris dataset:\n\n{}\n'.format(df_iris.describe()))


U_iris, S_iris, Vt_iris = np.linalg.svd(df_iris, full_matrices=False)

print('matrix U has {} rows, {} columns\n'.format(*U_iris.shape))
print('here are the first 5 rows.')

print('{}'.format(pd.DataFrame(U_iris).head(5)))