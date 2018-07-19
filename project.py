from __future__ import print_function
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore', module='sklearn')

### IMPORT DATASET
data = pd.read_csv('C:/Users/Student/Desktop/PROJECT/DataSet.CSV',header='infer')

### VISUALIZATION
print("\n\nHERE IS THE HEAD")
print(data.head())

print("\n\nHERE ARE THE COLUMNS")
print(data.columns)

print("\n\nHERE IS THE DESCRIPTION")
print(data.describe())

### PLOTTING GRAPH
# ax = plt.axes()
# ax.scatter(data.Age, data.Snoring)
# ax.set(xlabel='Age',
#        ylabel='Snoring',
#        title='Table');
# print(plt.show())


### CONVERTING TO FLOAT
lb = LabelBinarizer()
for col in ['Patient Id' ,'Level']:
    data[col] = lb.fit_transform(data[col])

msc = MinMaxScaler()
data = pd.DataFrame(msc.fit_transform(data),
                    columns=data.columns)

### DATAFRAMES
x_cols = [x for x in data.columns if x != 'Level']
X_data = data[x_cols]
y_data = data['Level']


### KNN PREDICTION
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_data,y_data)
y_pred = knn.predict(X_data)
print(y_pred)



### ACCURACY
def accuracy(real, predict):
    return sum(y_data == y_pred) / float(real.shape[0])
print(accuracy(y_data, y_pred))

