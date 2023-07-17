'''

12.07.2023 END
TODO | Figure out how to add categorized column to train_set with a pipeline.
TODO | Understand OneHotEncoder's working principle. | DONE
TODO | Understand Pipelines better.
TODO | Understand how to do the process' the pipelines do without needing a pipeline. | DONE
Good Night, Good Luck.

'''


import os
import numpy as np

PROJECT_DIR_ROOT = '.'
DATA_PATH = 'kaggle_titanic_data'
DOWNLOAD_PATH = 'c:\\Users\\ALKAN\\Downloads'
IMAGE_PATH = 'images\\titanic'

from zipfile import ZipFile

# Fetch the data in zip file.
# Create 'kaggle_titanic_data' folder in the folder where our code is.
# Create zip_path by combining my PC's download address and the name of zip.
# Locate zip address, extract it at data_path address.
def fetch_data(download_path = DOWNLOAD_PATH, data_path = DATA_PATH):
    os.makedirs(data_path, exist_ok = True)
    zip_path = os.path.join(download_path, 'titanic.zip')
    with ZipFile(zip_path, 'r') as titanic_zip:
        titanic_zip.extractall(path = data_path)
    
fetch_data()

import pandas as pd

# Options for Pandas output look.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Convert .csv file to DataFrame
def load_train_data(data_path = DATA_PATH):
    train_set_path = os.path.join(data_path, 'train.csv')
    return pd.read_csv(train_set_path)

def load_test_data(data_path = DATA_PATH): 
    test_set_path = os.path.join(data_path, 'test.csv')
    return pd.read_csv(test_set_path)

train_set = load_train_data()
test_set = load_test_data()
print('\nTrain Set\n')
print(train_set.head(), '\n\n\n')
print('More Information\n')
print(train_set.info(), '\n\n\n')
print('Description\n')
print(train_set.describe(), '\n\n\n')
train_set_original = train_set.copy()

# We fill null values in Age column with mean of age.
# Cabin and Embarked are not numerical, we will deal with them later.
mean_age = train_set['Age'].mean()
train_set['Age'].fillna(mean_age, inplace = True)
test_set['Age'].fillna(mean_age, inplace = True)

# Will assign 0 to male, 1 to female so that Sex column will be numerical.
train_set['Sex'] = train_set['Sex'].replace({'male': 0, 'female': 1})
test_set['Sex'] = test_set['Sex'].replace({'male': 0, 'female': 1})


# Will assign 0 == (C)herbourg, 1 == (Q)ueenstown, 2 == (S)outhhampton
# Will get mean and fillna()
train_set['Embarked'] = train_set['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})
test_set['Embarked'] = test_set['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})
train_mean_embarked = train_set['Embarked'].mean()
test_mean_embarked = test_set['Embarked'].mean()
train_set['Embarked'].fillna(train_mean_embarked, inplace = True)
test_set['Embarked'].fillna(test_mean_embarked, inplace = True)

# Every deck is converted to a numerical value, null values were assigned numerical values according to the 
# weights of every category.
train_weights = train_set['Cabin'].value_counts(normalize = True)
test_weights = test_set['Cabin'].value_counts(normalize = True)

pattern_replacement = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
train_set['Cabin'] = train_set['Cabin'].replace(pattern_replacement, regex = True)
test_set['Cabin'] = test_set['Cabin'].replace(pattern_replacement, regex = True)

train_weights = train_set['Cabin'].value_counts(normalize = True)
test_weights = test_set['Cabin'].value_counts(normalize = True)

replacement_values = np.random.choice(train_weights.index, size = train_set['Cabin'].isnull().sum(), p = train_weights.values)
train_set.loc[train_set['Cabin'].isnull(), 'Cabin'] = replacement_values

replacement_values = np.random.choice(test_weights.index, size = test_set['Cabin'].isnull().sum(), p = test_weights.values)
test_set.loc[test_set['Cabin'].isnull(), 'Cabin'] = replacement_values

# Remaining non-numerical values are Name, Ticket and Cabin.
# Obviously Name has no correlation with survivability, I would assume so does Ticket so maybe we'll simply drop them.
# Just incase we will run correlation.
train_set_numeric = train_set.select_dtypes(include=[np.number])
corr_matrix = train_set_numeric.corr()
print('Correlation Matrix\n')
print(corr_matrix, '\n\n\n')

# We had to go back and do the same data editing on the test set aswell. This is hell.
# And ofcourse they had to put a null value in test set that isn't in train set.. 152nd row has a null 'Fare'.
test_mean_fare = test_set['Fare'].mean()
test_set['Fare'].fillna(test_mean_fare, inplace = True)

'''

Sex effects Survived. As Sex increases (Female), so does the survivability.
Class effects Survived. As Class increases in value (not in real class as in 3rd class 2nd class etc.), so does the survivability.
Fare effects Survived. As Fare increases (numerical), so does the survivability.
Embarked has minor effect.

PassengerID, Age, SibSp, Parch, has very little effect on survivability.
Name, Ticket can't be categorized as numerical and most likely has no corelation with survivability.
Even if it did, it would be a pattern and useless.

'''

train_set.drop('Name', axis = 1, inplace = True)
test_set.drop('Name', axis = 1, inplace = True)
train_set.drop('Ticket', axis = 1, inplace = True)
test_set.drop('Ticket', axis = 1, inplace = True)
train_set.drop('PassengerId', axis = 1, inplace = True)
test_set.drop('PassengerId', axis = 1, inplace = True)

# All values are numerical. Now we will check features with a graph to see any anomalies and remove them.
import matplotlib.pyplot as plt

def save_figure(figure_name, tight_layout = False, figure_extension = 'png', resolution = 300):
    os.makedirs(IMAGE_PATH, exist_ok = True)
    path = os.path.join(IMAGE_PATH, figure_name + '.' + figure_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = figure_extension, dpi = resolution)

train_set.hist(bins = 50, figsize = (10, 10), color = 'red',
               edgecolor = 'black', linewidth = 1)
save_figure('passenger_values')
##plt.show()
plt.close()

train_set['Age'].hist(bins = 50, figsize = (10, 8), color = 'red',
               edgecolor = 'black', linewidth = 1)
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Number', fontsize = 16)
save_figure('age_plot')
##plt.show()
plt.close()

train_set['Fare'].hist(bins = 50, figsize = (10, 8), color = 'red',
               edgecolor = 'black', linewidth = 1)
plt.xlabel('Fare', fontsize = 16)
plt.ylabel('Number', fontsize = 16)
save_figure('fare_plot')
##plt.show()
plt.close()

train_set['Survived'].hist(bins = 2, figsize = (5, 8), color = 'red',
               edgecolor = 'black', linewidth = 2)
plt.xlabel('Survived', fontsize = 16)
plt.ylabel('Number', fontsize = 16)
save_figure('survived_plot')
##plt.show()
plt.close()

train_set['Pclass'].hist(bins = 3, figsize = (5, 8), color = 'red',
               edgecolor = 'black', linewidth = 2)
plt.xlabel('Class', fontsize = 16)
plt.ylabel('Number', fontsize = 16)
save_figure('class_plot')
##plt.show()
plt.close()

train_set['SibSp'].hist(bins = 8, figsize = (5, 8), color = 'red',
               edgecolor = 'black', linewidth = 2)
plt.xlabel('Sibling/Spouses', fontsize = 16)
plt.ylabel('Number', fontsize = 16)
save_figure('sibsp_plot')
##plt.show()
plt.close()

# Create scatter matrix just cause.
from pandas.plotting import scatter_matrix

scatter_matrix(train_set, figsize = (12, 8), edgecolor = 'black', linewidth = 0.2, color = 'navy')
save_figure('scatter_matrix')
##plt.show()
plt.close()

# Had our fun. Time to pick a model and see its accuracy.
# I'll start with Random Forest
from sklearn.ensemble import RandomForestClassifier

y = train_set['Survived']
X = train_set.drop('Survived', axis = 1)

forest_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 42)
forest_classifier.fit(X, y)
prediction = forest_classifier.predict(X)


# Will try accuracy with confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(forest_classifier, X, y, cv = 5)
print('Confusion Matrix Train Set\n')
print(confusion_matrix(y, y_pred), '\n\n\n')

X_test = test_set
X = train_set.drop('Survived', axis = 1)

prediction_test = forest_classifier.predict(X_test)
print('Prediction Test\n\n',prediction_test, '\n\n\n')

y_test_pred = cross_val_predict(forest_classifier, X_test, prediction_test, cv = 5)
print('Confusion Matrix Test Set\n')
print(confusion_matrix(prediction_test, y_test_pred), '\n\n\n')


'''

%99 Done, all I need to do is print 'Survived' values of test set into a text file.

'''


survivor_values = prediction_test
passenger_id = []
i = 892
while i < 1310:
    passenger_id.append(i)
    i += 1

final_array = np.column_stack((passenger_id, survivor_values))
final_array_csv = pd.DataFrame(final_array, columns = ['PassengerId', 'Survived'])
filename = 'prediction_titanic'
file_extension = 'csv'
prediction_path = DATA_PATH + '\\' + filename + '.' + file_extension
final_array_csv.to_csv(prediction_path, index = False)


'''

17.07.2023
Uploaded to Kaggle. %78.229 accuracy. Not bad for first project.

'''