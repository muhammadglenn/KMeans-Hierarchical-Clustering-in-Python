# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# Dataset
cust = pd.read_excel('D:\Datasets\Kaggle\Customer Clustering\customer_data.xlsx')
print("Customer Dataset")
print(cust.head(10))
print("")

# Explore Dataset
print("Info Hearth Dataset")
print(cust.info())
print("")

# Check Duplicates
print("Number of duplicates")
print(cust.duplicated().sum())
print("")
print(cust.loc[cust.duplicated()])
print("")

# Remove ID column
cust= cust.drop(['ID'],axis = 1)

# Numerical variables brief
print("Numerical variables brief")
print(cust.describe())
print("")

# Removing (statistical) outliers for Age
Q1 = cust.Age.quantile(0.05)
Q3 = cust.Age.quantile(0.95)
IQR = Q3 - Q1
cust = cust[(cust.Age >= Q1 - 1.5*IQR) & (cust.Age <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Income
Q1 = cust.Income.quantile(0.05)
Q3 = cust.Income.quantile(0.95)
IQR = Q3 - Q1
cust = cust[(cust.Income >= Q1 - 1.5*IQR) & (cust.Income <= Q3 + 1.5*IQR)]

# Numerical variables brief
print("Numerical variables brief after removing outliers")
print(cust.describe())
print("")

# Rescaling the attributes

cust_df = cust.copy()
print(cust_df)

scaler = StandardScaler()

cust_df_scaled = scaler.fit_transform(cust_df)
cust_df_scaled = pd.DataFrame(cust_df_scaled)
print('Variables values after rescaled')
print(cust_df_scaled.describe())
print('')

# Model
# k-means with some arbitrary k
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(cust_df_scaled)
print('Labels with 4 Clusters')
print(kmeans.labels_)
print('')

# Elbow-curve/SSD
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(cust_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)
plt.show();

#Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(cust_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(cust_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    print('');


# k-means with some arbitrary k
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(cust_df_scaled)
print('Labels with 3 Clusters')
print(kmeans.labels_)
print('')

# Assign the label
cust['Kmean_cluster'] = kmeans.labels_
print('Customer dataset with cluster variables from k mean algorithm')
print(cust)
print('')

# Visualize the numerical variables
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.boxplot(data=cust, x='Kmean_cluster', y='Income', ax=ax[0])
sns.boxplot(data=cust, x='Kmean_cluster', y='Age', ax=ax[1])
plt.show();

# Visualize the categorical variables
category = cust[['Sex','Marital status','Education','Occupation','Settlement size']]
for catplot in category:
    sns.countplot(data=cust, x=catplot, hue="Kmean_cluster")
    plt.show();

# Hierarchical Clustering (Single Linkage)
mergings1 = linkage(cust_df_scaled, method="single", metric='euclidean')
dendrogram(mergings1)
plt.show();

# Hierarchical Clustering (Complete Linkage)
mergings2 = linkage(cust_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings2)
plt.show();

# Hierarchical Clustering (Average Linkage)
mergings3= linkage(cust_df_scaled, method="average", metric='euclidean')
dendrogram(mergings3)
plt.show();

# Cutting the dendogram based on 3 clusters
cluster_labels1 = cut_tree(mergings1, n_clusters=3).reshape(-1, )
cluster_labels2 = cut_tree(mergings2, n_clusters=3).reshape(-1, )
cluster_labels3 = cut_tree(mergings3, n_clusters=3).reshape(-1, )

# Assign cluster labels
cust['Hierarchical_cluster'] = cluster_labels2
print('Customer dataset with cluster variables from k mean and hierarchical algorithm')
print(cust)
print('')

#Visualize the numerical variables
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.boxplot(data=cust, x='Hierarchical_cluster', y='Income', ax=ax[0])
sns.boxplot(data=cust, x='Hierarchical_cluster', y='Age', ax=ax[1])
plt.show();

# Visualize the categorical variables
category = cust[['Sex','Marital status','Education','Occupation','Settlement size']]
for catplot in category:
    sns.countplot(data=cust, x=catplot, hue="Hierarchical_cluster")
    plt.show();

