from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore', module='sklearn')

### IMPORT DATASET
data = pd.read_csv('C:/Users/Student/Desktop/PROJECT/DataSet.CSV',header='infer')



# ### VISUALIZATION
# print("\n\nHERE IS THE HEAD")
# print(data.head())

# print("\n\nHERE ARE THE COLUMNS")
# print(data.columns)

# print("\n\nHERE IS THE DESCRIPTION")
# print(data.describe())


### PLOTTING GRAPH
# ax = plt.axes()
# ax.scatter(data.Age, data.Snoring)
# ax.set(xlabel='Age',
#        ylabel='Snoring',
#        title='Table');
# print(plt.show())


# ## CONVERTING TO FLOAT
# lb = LabelBinarizer()
# for col in ['Patient Id']:
#     data[col] = lb.fit_transform(data[col])

# msc = MinMaxScaler()
# data = pd.DataFrame(msc.fit_transform(data),
#                     columns=data.columns)



### DATAFRAME STRUCTURING/SPLITTING
x_cols = [x for x in data.columns if x != 'Level']
X_data = data[x_cols] #FEATURES
y_data = data['Level'] #LABELS/TARGETS

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size = .3)

print("\n\nTRAINING DATA: ")
print(X_train,y_train) #FOR TRAINING THE CLASSIFIER

print("\n\nTESTING DATA: ")
print(X_test, y_test) #FOR TESTING ACCURACY



### KNN PREDICTION
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_train,y_train) #TRAINING
knn_y_pred = knn.predict(X_test)

print("\n\nKNN Prediciton: ")
print(knn_y_pred)

### ACCURACY KNN
def accuracy(real, predict):
    return sum(y_test == knn_y_pred) / float(real.shape[0])
print("KNN Predicts with",accuracy(y_test, knn_y_pred),"accuracy")

### DECISION TREE PREDICTION
dtt = tree.DecisionTreeClassifier()
dtt = dtt.fit(X_train,y_train) #TRAINING
dtt_y_pred = dtt.predict(X_test)


print("\n\nDTT Prediciton: ")
print(dtt_y_pred)

### ACCURACY DTT
def accuracy(real, predict):
    return sum(y_test == dtt_y_pred) / float(real.shape[0])
print("DTT predicts with",accuracy(y_test, dtt_y_pred),"accuracy")


#CONVERTING FLOAT OUTPUT BACK TO STRING
output = list()

for x in dtt_y_pred:
	if x == 0:
		output.append("Low")
	elif x == 1:
		output.append("Medium")
	elif x == 2:
		output.append("High")
print(output)







