# import numpy as np
# from sklearn.preprocessing import scale
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn import preprocessing
# import numpy as np

# import torch

def Input_from_opponent_char(str_):
    if str_ == 'a':
        return '1'
    elif str_ == 'b':
        return '2'
    elif str_ == 'c':
        return '3'
    elif str_ == 'd':
        return '4'
    elif str_ == 'e':
        return '5'

    elif str_ == 'f':
        return '6'
    elif str_ == 'g':
        return '7'
    elif str_ == 'h':
        return '8'
    elif str_ == 'i':
        return '9'
    elif str_ == 'j':
        return '10'

    elif str_ == 'k':
        return '11'
    elif str_ == 'l':
        return '12'
    elif str_ == 'm':
        return '13'
    elif str_ == 'n':
        return '14'
    elif str_ == 'o':
        return '15'

    elif str_ == 'p':
        return '16'
    elif str_ == 'q':
        return '17'
    elif str_ == 'r':
        return '18'
    elif str_ == 's':
        return '19'


def Input_from_opponent_num(str_):
    if str_ == '1':
        return 'S'
    elif str_ == '2':
        return 'R'
    elif str_ == '3':
        return 'Q'
    elif str_ == '4':
        return 'P'
    elif str_ == '5':
        return 'O'

    elif str_ == '6':
        return 'N'
    elif str_ == '7':
        return 'M'
    elif str_ == '8':
        return 'L'
    elif str_ == '9':
        return 'K'
    elif str_ == '10':
        return 'J'

    elif str_ == '11':
        return 'I'
    elif str_ == '12':
        return 'H'
    elif str_ == '13':
        return 'G'
    elif str_ == '14':
        return 'F'
    elif str_ == '15':
        return 'E'

    elif str_ == '16':
        return 'D'
    elif str_ == '17':
        return 'C'
    elif str_ == '18':
        return 'B'
    elif str_ == '19':
        return 'A'


def Output_from_num(str_):
    if str_ == '1':
        return 'A'
    elif str_ == '2':
        return 'B'
    elif str_ == '3':
        return 'C'
    elif str_ == '4':
        return 'D'
    elif str_ == '5':
        return 'E'

    elif str_ == '6':
        return 'F'
    elif str_ == '7':
        return 'G'
    elif str_ == '8':
        return 'H'
    elif str_ == '9':
        return 'I'
    elif str_ == '10':
        return 'J'

    elif str_ == '11':
        return 'K'
    elif str_ == '12':
        return 'L'
    elif str_ == '13':
        return 'M'
    elif str_ == '14':
        return 'N'
    elif str_ == '15':
        return 'O'

    elif str_ == '16':
        return 'P'
    elif str_ == '17':
        return 'Q'
    elif str_ == '18':
        return 'R'
    elif str_ == '19':
        return 'S'


def Output_from_char(str_):
    if str_ == 'a':
        return '19'
    elif str_ == 'b':
        return '18'
    elif str_ == 'c':
        return '17'
    elif str_ == 'd':
        return '16'
    elif str_ == 'e':
        return '15'

    elif str_ == 'f':
        return '14'
    elif str_ == 'g':
        return '13'
    elif str_ == 'h':
        return '12'
    elif str_ == 'i':
        return '11'

    elif str_ == 'j':
        return '10'
    elif str_ == 'k':
        return '9'
    elif str_ == 'l':
        return '8'
    elif str_ == 'm':
        return '7'

    elif str_ == 'n':
        return '6'
    elif str_ == 'o':
        return '5'
    elif str_ == 'p':
        return '4'
    elif str_ == 'q':
        return '3'
    elif str_ == 'r':
        return '2'
    elif str_ == 's':
        return '1'


while True:
    try:
        str_ = str(input()).split()
        print(str_)
        print("Input:", Input_from_opponent_num(str_[1]), Input_from_opponent_char(str_[0]))

        print("Output:", Output_from_num(str_[1]), Output_from_char(str_[0]))
    except:
        print('try again!!')

"""
x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
plt.scatter(x, y)
# plt.show()

df = pd.DataFrame({'x': x, 'y': y})

scaled_data = preprocessing.scale(df)

plt.scatter(scaled_data[:, 0], scaled_data[:, 1])

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

print(pca.get_covariance())

import sys
sys.exit()
path = r"C:\go_concat\new_player_table.pkl"
df = pd.read_pickle(path)

df = df.astype({"win%": "float", "win%_human": "float", "win%_bot": "float", "bot%": "float", "wb%": "float",
                "ww%": "float", "wb_human%": "float", "wb_bot%": "float", "ww_human%": "float", "ww_bot%": "float"})



plt.style.use("seaborn")
df = df.dropna()
df = df[["win%", "ww%", "wb%", "win%_human", "win%_bot", "wb_human%", "ww_human%", "ww_bot%", "wb_bot%"]]

# Scale the data so that each row the mean = 0 and std = 1
# The scale function expects samples to be rows so -> transpose it
scaled_data = preprocessing.scale(df.T)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

print(pca_data.shape)


# scree plot
# Displays how much variation each principal component captures from the data
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(i) for i in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("Variance Ratio%")
plt.xlabel("Principle components")
plt.title("Scree plot")
path = r"C:\go_concat\graph\pca_scree_plot.png"
plt.savefig(path)
plt.show()
index = ["win%", "ww%", "wb%", "win%_human", "win%_bot", "wb_human%", "ww_human%", "ww_bot%", "wb_bot%"]


# Draw PCA plot
# A PCA plot shows clusters of samples based on their similarity.
pca_df = pd.DataFrame(pca_data, index=index, columns=labels)

plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('PCA Graph')

plt.xlabel('PC1 -{0}%'.format(per_var[0]))
plt.ylabel('PC2 -{0}%'.format(per_var[1]))

for i in pca_df.index:
    plt.annotate(i, (pca_df['PC1'].loc[i],pca_df['PC2'].loc[i]))

from k_mean import cluster_df
path = r"C:\go_concat\graph\pca_graph.png"
cluster_df(df=pca_df[["PC1","PC2"]], x_label="PC1", y_label="PC2", n_cluster=3,save_path=path, color_centroid=False)

print(pca_df["PC1"])
# A loading plot shows how strongly each characteristic influences a principal component.
loading_scores = pd.Series(pca.components_[0], index=df.index)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes]
"""

"""
Principle Component Analysis (PCA)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# df = pd.read_pickle("./output/table/human_table.pkl")
# df = pd.DataFrame(df)
# df = df.dropna()
# df = df.astype({"win%": "float", "win%_human": "float", "win%_bot": "float", "bot%": "float", "wb%": "float",
#                 "ww%": "float", "wb_human%": "float", "wb_bot%": "float", "ww_human%": "float", "ww_bot%": "float"})

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)
x = dataset.drop('Class', 1)
y = dataset['Class']
# # Preprocessing
# print(df.to_string())
# x = df[["win%", "bot%"]]
# #
# y = df.sort_values(by=["n_game"])["n_game"]

# Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Perform standard scalar normalization to normalize the feature set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# PCA
pca = PCA(n_components=4)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# use random forest classification for making the predictions

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy', accuracy_score(y_test, y_pred))

# <END 1> --------------------------------- <END 1>


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('mushrooms.csv')

# Machine learning systems work with integers, we need to encode these
# string characters into ints

encoder = LabelEncoder()

# Now apply the transformation to all the columns:
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X_features = df.iloc[:,1:23]
y_label = df.iloc[:, 0]

# Scale the features
sc = StandardScaler()
X_features = sc.fit_transform(X_features)

# Visualize
pca = PCA()
pca.fit_transform(X_features)
pca_variance = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

pca2 = PCA(n_components=2)
pca2.fit(X_features)
x_3d = pca2.transform(X_features)

plt.figure(figsize=(8,6))
plt.scatter(x_3d[:,0], x_3d[:,1], c=df['class'])
plt.show()

# <END 2> --------------------------------- <END 2>

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()


pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


# # plot data
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal');
# plt.show()

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
print(X_new)
plt.axis('equal')
plt.show()

# <END 3> --------------------------------- <END 3>

"""

"""
K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.style.use("seaborn")
X = np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])
# plt.scatter(X[:, 0], X[:, 1], label="True Position")
# plt.show()
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()

"""



"""
Linear Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("seaborn")
df = pd.read_pickle("./output/table/human_table.pkl")
df = pd.DataFrame(df)


print(df.head().to_string())
df = df.sort_values(by=["n_game", "win%"])

# Change the data type of every column to float
df = df.astype({"win%": "float", "win%_human": "float", "win%_bot": "float", "bot%": "float", "wb%": "float",
                "ww%": "float", "wb_human%": "float", "wb_bot%": "float", "ww_human%": "float", "ww_bot%": "float"})

x = df["n_game"].tolist()
y = df["win%"].tolist()

x_bar = np.mean(x)
y_bar = np.mean(y)

sum_residual = 0
variance = 0
for i in range(df.shape[0]):
    x_diff = (x[i] - x_bar)
    sum_residual += x_diff * (y[i] - y_bar)
    variance += x_diff ** 2
m = sum_residual / variance

b = y_bar - m * x_bar
print(m, b)

x_n = np.arange(0, 20)
y_n = m*x_n + b

plt.plot(x_n, y_n)
plt.show()

"""
