# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:20:30 2022

@author: vaishnav
"""
#======================================================================================================
#importing the data
import pandas as pd

df = pd.read_csv("C:\\anaconda\\New folder (2)\\EastWestAirlines.csv")
df

df.dtypes
df.drop("ID",axis=1,inplace=True)
#=======================================================================================================
#EDA
import matplotlib.pyplot as plt

df["Balance"].hist()

df["Qual_miles"].hist()

df["cc1_miles"].hist()

df["cc2_miles"].hist()

df["cc3_miles"].hist()

df["Bonus_miles"].hist()

df["Bonus_trans"].hist()

df["Flight_miles_12mo"].hist()

df["Flight_trans_12"].hist()

df["Days_since_enroll"].hist()

import seaborn as sns

sns.kdeplot(df["Qual_miles"],shade=True)

sns.kdeplot(df["Balance"],shade=True)

sns.kdeplot(df["cc1_miles"],shade=True)

sns.kdeplot(df["cc2_miles"],shade=True)

sns.kdeplot(df["cc3_miles"],shade=True)

sns.kdeplot(df["Bonus_miles"],shade=True)

sns.kdeplot(df["Bonus_trans"],shade=True)

sns.kdeplot(df["Flight_miles_12mo"],shade=True)

sns.kdeplot(df["Flight_trans_12"],shade=True)

sns.kdeplot(df["Days_since_enroll"],shade=True)

sns.kdeplot(df["Award"],shade=True)

sns.countplot(x=df["Award"])



# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()



sns.pairplot(df)
plt.show()




#==========================================================================================================

cat_list=[]
num_list=[]


for i in df.columns:
    unique_values = len(df[i].unique())
    if unique_values<10:
        cat_list.append(i)
    else:
        num_list.append(i)
        
cat_list
num_list


#Let's look at the distributions of #num_list
import matplotlib.pyplot as plt

k=1
plt.figure(figsize=(12,12))
plt.suptitle("distribution of numerical values")
for i in df.loc[:,num_list]:
    plt.subplot(4,2,k)
    sns.distplot(df[i])
    plt.title(i)
    plt.tight_layout()
    k+=1

#foroutliers==============================================================
#to know if there are outliers or not

for i in df.loc[:,num_list]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    up = Q3 + 1.5*IQR
    low = Q1 - 1.5*IQR

    if df[(df[i] > up) | (df[i] < low)].any(axis=None):
        print(i,"yes")
    else:
        print(i, "no")

#==========================================================================================
#boxplots

k=1
plt.figure(figsize=(13,13))
plt.suptitle("Distribution of Outliers")

for i in df.loc[:,num_list]:
    plt.subplot(4,2,k)
    sns.boxplot(x = i, data = df.loc[:,num_list])
    plt.title(i)
    plt.tight_layout()
    k+=1


# we have examined all the stops, let's remove outliers in 4 variables (other cases are possible)
out_list=["Bonus_trans","Flight_miles_12mo","Flight_trans_12"]

# remove outliers;

for i in df.loc[:,out_list]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    up_lim = Q3 + 1.5 * IQR
    low_lim = Q1 - 1.5 * IQR
    df.loc[df[i] > up_lim,i] = up_lim
    df.loc[df[i] < low_lim,i] = low_lim

# we fixed outliers

#====================================================================================
#Cat_list

for i in cat_list:
    plt.figure(figsize=(6,6))
    sns.countplot(x = i, data =df.loc[:,cat_list])
    plt.title(i)
#=======================================================================================
#clusteringKMeans

from sklearn.cluster import KMeans
kmeans=KMeans().fit(df)
# let's find the optimum number of clusters;
score=[]
K=range(1,20)

for i in K:
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=3)
    kmeans.fit(df)
    score.append(kmeans.inertia_)

#visualize;

plt.plot(K,score,color="red")
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()

# number of clusters 3-4 selectable
# K-elbow 


#pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer
# for K-elbow;

kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,20))
visualizer.fit(df)
visualizer.poof()
plt.show()

# gave the optimum number of clusters 4, let's try by standardizing it;
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_std=sc.fit_transform(df)
X_std

kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,20))
visualizer.fit(X_std)
visualizer.poof()
plt.show()


# It makes more sense to divide into 4 clusters
#final model;
kmeans=KMeans(n_clusters=4,init="k-means++").fit(df)
#add tag values;

cluster=kmeans.labels_
cluster

df["cluster_no"]=cluster
df.head()

df.cluster_no.value_counts()

#======================================================================================================
#Hierarcihal Cluster
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(df,method="ward"))
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="ward")
Y=cluster.fit_predict(X_std)

Y=pd.DataFrame(Y)
Y[0].value_counts()

#======================================================================================================
dendogram=sch.dendrogram(sch.linkage(df,method="complete"))
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="complete")
Y=cluster.fit_predict(X_std)

Y=pd.DataFrame(Y)
Y[0].value_counts()

#=============================================================================================================
#DBScan

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.75,min_samples=3)
db.fit(X_std)
y=db.labels_
y=pd.DataFrame(y,columns=['cluster'])
y["cluster"].value_counts()

newdata =pd.concat([df,y],axis=1)

noisedata = newdata[newdata['cluster'] == -1]
print(noisedata)
finaldata = newdata[newdata['cluster'] == 0]
print(finaldata)

