import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import confusion_matrix

#importing the datafile
dataset=pd.read_csv('cancer.csv')
dataset.head()

#allocating the values to i/p feature variable(x) and target label(y)
x= dataset.iloc[:,2:32].values
y= dataset.iloc[:,1].values
print(dataset.shape)

#Next steps are optional for better visualization of data

#It is a plot of radiusVstexture 

'''
M=dataset[dataset.diagnosis == "M"]
B=dataset[dataset.diagnosis == "B"]
plt.title("Malignant vs Benign")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color='red', label="Malignant",alpha=0.3) #1st i/p is x axis and 2 is y axis alpha is just opacity
plt.scatter(B.radius_mean, B.texture_mean, color='blue', label="Benign", alpha=0.1)
plt.legend()
plt.show()
'''

#It is for histogram and to see if there are no null values(if any make sure there is none before scaling or normalization)
#you can generate histograms of diff column also you can use group by for a better visualization

'''
dataset.info()
dataset['radius_mean'].hist()
plt.show()
dataset['texture_mean'].hist()
plt.show()
dataset['fractal_dimension_worst'].hist()
plt.show()

#dataset.boxplot(column='')
#plt.show()

dataset.isnull().sum()
'''

#Transforming char values in y to numerical values
e_y = LabelEncoder()
y = e_y.fit_transform(y)

#splitting dataset into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) #0.25 shows that one-fourth is testing set(you can change this) 

#feature scaling is done here
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#performing logistic regression on training set
classification = LogisticRegression(random_state = 0)
classification.fit(x_train, y_train)

#performing logistic regression on testing set
y_pred = classification.predict(x_test)

print(y_pred)
print(y_test)

#making confusion matrix to know accuracy
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression Accuracy ~{:.2f}".format((cm[0,0]+cm[1,1])/cm.sum()*100))

classification = KNeighborsClassifier(n_neighbors = 5)
classification.fit(x_train, y_train)
y_pred = classification.predict(x_test)
print(y_pred)
print(y_test)
cm = confusion_matrix(y_test, y_pred)
print("Fitting KNN algo Accuracy ~{:.2f}".format((cm[0,0]+cm[1,1])/cm.sum()*100))

classification = GaussianNB()
classification.fit(X_train, Y_train)
y_pred = classification.predict(x_test)
print(y_pred)
print(y_test)
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes Accuracy ~{:.2f}".format((cm[0,0]+cm[1,1])/cm.sum()*100))




