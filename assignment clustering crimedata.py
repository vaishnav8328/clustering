# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:43:55 2022

@author: vaishnav
"""
#Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.

#importing the data
import pandas as pd

df = pd.read_csv("C:\\anaconda\\New folder (2)\\crime_data.csv")
df

df.rename(columns={ df.columns[0]: "place" }, inplace = True)


#=================================================================================================
#EDA

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df)
plt.show()

plt.scatter(x=df["Murder"],y=df["UrbanPop"])

plt.scatter(x=df["Rape"],y=df["UrbanPop"])

plt.scatter(x=df["Assault"],y=df["Rape"])


df["Murder"].hist()

df["Rape"].hist()

df["UrbanPop"].hist()

df["Assault"].hist()


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))


# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
#==================================================================================================
# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


X = df.iloc[:,:]
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])

#=========================================================================================================
#clustering Agglomerative

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10,7))
plt.title('customer dendogram')
dend = shc.dendrogram(shc.linkage(df_norm,method='ward'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward")
Y=cluster.fit_predict(df_norm)

Clusters=pd.DataFrame(Y,columns=['Clusters'])
df_norm['h_clusterid'] = cluster.labels_
df_norm

df_norm["h_clusterid"].value_counts()

#=======================================================================================


plt.figure(figsize=(10,7))
plt.title('customer dendogram')
dend = shc.dendrogram(shc.linkage(df_norm,method='complete'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="complete")
Y=cluster.fit_predict(df_norm)

Clusters=pd.DataFrame(Y,columns=['Clusters'])
df_norm['h_clusterid'] = cluster.labels_
df_norm

df_norm["h_clusterid"].value_counts()


#============================================================================================

plt.figure(figsize=(10,7))
plt.title('customer dendogram')
dend = shc.dendrogram(shc.linkage(df_norm,method='single'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="single")
Y=cluster.fit_predict(df_norm)

Clusters=pd.DataFrame(Y,columns=['Clusters'])
df_norm['h_clusterid'] = cluster.labels_
df_norm

df_norm["h_clusterid"].value_counts()

#=============================================================================================


#clusteringKMeans

from sklearn.cluster import KMeans
kmeans=KMeans().fit(df_norm)
# let's find the optimum number of clusters;
score=[]
K=range(1,15)

for i in K:
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=3)
    kmeans.fit(df_norm)
    score.append(kmeans.inertia_)



#visualize;

plt.plot(K,score,color="red")
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()



from yellowbrick.cluster import KElbowVisualizer
# for K-elbow;

kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,15))
visualizer.fit(df_norm)
visualizer.poof()
plt.show()



# It makes more sense to divide into 4 clusters
#final model;
kmeans=KMeans(n_clusters=4,init="k-means++").fit(df_norm)
#add tag values;

cluster=kmeans.labels_
cluster

df["cluster_no"]=cluster
df.head()


df.cluster_no.value_counts()

#======================================================================================================
#DBScan

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss_x=ss.fit_transform(df.iloc[:,1:])

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.75,min_samples=3)
db.fit(ss_x)
y=db.labels_
y=pd.DataFrame(y,columns=['cluster'])
y["cluster"].value_counts()

newdata =pd.concat([df,y],axis=1)

noisedata = newdata[newdata['cluster'] == -1]
print(noisedata)
finaldata = newdata[newdata['cluster'] == 0]
print(finaldata)

#==========================================================================================================


































