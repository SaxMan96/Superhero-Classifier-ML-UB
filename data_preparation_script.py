# Imports

import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, f1_score, confusion_matrix
# Load Data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, , f1_score, accuracy_score, roc_curve, auc


def load_data() -> pd.Series:
    csv_train = pd.read_csv('superhero-or-supervillain/train.csv').assign(train = 1) 
    csv_test = pd.read_csv('superhero-or-supervillain/test.csv').assign(train = 0) 
    csv = pd.concat([csv_train,csv_test], sort=True)
    return csv

# Analyze Data

def nans_ctr(csv) -> pd.Series:
    return csv.isna().sum()

def unique_ctr(csv) -> pd.Series():
    unique = pd.Series()
    for col in list(csv):
        if(csv.columns.contains(col)):
            unique.at[col] = len(csv[col].unique())
    return unique

def val_types(csv) -> pd.Series():
    val_type = pd.Series()
    for col in list(csv):
        if not csv.columns.contains(col):
            continue
        if csv[col].dtype == np.float64:
            val_type.at[col] = np.float64
        elif csv[col].dtype == np.int64:
            val_type.at[col] = np.int64
        elif csv[col].dtype == np.int32:
            val_type.at[col] = np.int32
        elif csv[col].dtype == np.uint8:
            val_type.at[col] = np.uint8
        elif csv[col].dtype == object:
            val_type.at[col] = object
        elif csv[col].dtype == bool:
            val_type.at[col] = bool
        else:
            print(f"No common value type found in val_types() - {csv[col].dtype}")
    return val_type

def min_max_val(csv) -> pd.Series():
    min_val = pd.Series()
    max_val = pd.Series()
    val_type = val_types(csv)
    for col in list(csv):
        if val_type[col] != object:
            min_val.at[col] = csv[col].min()
            max_val.at[col] = csv[col].max()
        else:    
            min_val.at[col] = None
            max_val.at[col] = None
    return min_val, max_val        
            
def get_stats(csv):
    nans = nans_ctr(csv)
    unique = unique_ctr(csv)
    val_type = val_types(csv)
    min_val, max_val = min_max_val(csv)
    result = pd.DataFrame({ 'nans': nans, 'unique': unique, 'val_type': val_type, 'min_val': min_val, 'max_val': max_val}) 
    return result
    
def bool_to_integer(csv) -> pd.DataFrame():
    for col in csv.columns:
        if csv[col].dtype == bool:
            csv[col] = csv[col].astype(int)
    return csv
    
def standarize_numerical_values(csv):
    for col in csv.columns:
        if col == 'train':
            continue
        if csv[col].dtype == np.float64:
            data = csv[col]
            std = data.std()
            data = data[(data < data.quantile(0.99)) & (data > data.quantile(0.01))]
            mean = data.mean()
            csv[col] = (csv[col] - mean)/std
#             _ = plt.hist(csv[col], bins='auto', alpha = 0.5)
#             plt.yscale('log')
#             plt.title(f"Distr in {col} column")
#             plt.show()
    return csv

def check_rows(csv):
    for row in range(len(csv)):
        print(row, csv.iloc[row].isna().sum())
    return csv

def distribution_in_columns(csv):
    for col in list(csv):
        print(csv[col].value_counts())
    return csv
        
def plot_dist_y(csv):
    plt.pie([len(csv[csv['Alignment'] == 'good']), len(csv[csv['Alignment'] == 'bad']), 
             len(csv[csv['Alignment'] == 'neutral'])], labels = ['good', 'bad', 'neutral'])
    plt.show()
    return csv
    
def factorize(csv) -> pd.DataFrame():
    for col in csv.select_dtypes(include=['object']).columns:
        if col == "Alignment":
            continue
        dummy = pd.get_dummies(csv[col])
        dummy.columns = [col+ " "+x for x in dummy.columns]
        dummy = dummy.drop([dummy.columns[-1]], axis=1)
        csv = csv.drop(col, axis=1)
        csv = pd.concat([csv, dummy], axis=1)
    return csv
    
def prepare_models(use_scaler = False, use_grid_search = False, verbose=False):
    gs_dict = {}
    clfs_dict = {
        'DTC': DecisionTreeClassifier(random_state=0),
        'RFC': RandomForestClassifier(random_state=0),
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression(random_state=35)
    }
    param_grids = {
        'DTC': {
            '__max_depth': [None,1,2,3,4,5,7,10,15],
            '__criterion': ['gini', 'entropy'],
            '__class_weight': [None],
            '__presort': [False,True],
#             '__min_samples_split': np.linspace(0.001,1.0,7,endpoint=True),
#             '__min_weight_fraction_leaf': np.linspace(0.05,0.4,5) 
        },
        'RFC': { 
            '__n_estimators': [15,50, 100],
            '__max_depth' : [1,2,3,5,10,20],
            '__criterion' : ['gini', 'entropy'],
            '__class_weight': [None,'balanced'],
            '__max_features': ['auto']
        },
        'KNN': {
            '__n_neighbors': range(1, 12),
            '__weights': ['uniform', 'distance'],
        },
        'LR': {
            "__C": np.logspace(-3,3,7), 
            "__penalty":["l1","l2"]
        }# l1 lasso l2 ridge
    }
    scorer = make_scorer(f1_score, average='micro')            
    for clf_name in clfs_dict:
        clf_name = 'LR'
        if use_scaler:
            pipe_line = Pipeline([('scaler', StandardScaler()), ('', clfs_dict[clf_name])])
        else:
            pipe_line = clfs_dict[clf_name]
            
        if use_grid_search:
            gs = GridSearchCV(pipe_line, param_grid=param_grids[clf_name], scoring=scorer, verbose=verbose)
        else:
            gs = pipe_line
        gs_dict[clf_name] = gs
        break
    return gs_dict
        
def model_selection(x_train, x_valid, y_train, y_valid):
    gs_dict = prepare_models(use_scaler = True, use_grid_search = True, verbose=False)
#     gs_dict = prepare_models(verbose=True)
    for gs_name in gs_dict:        
        gs = gs_dict[gs_name].fit(x_train,y_train)
        print("\n==================================================================\n", gs_name, end="\t")
#         print("\tF1-Score: ", np.round(100*gs.score(x_valid, y_valid),1), "%", sep="", end="\t")
        print("\tAccuracy Valid: ", np.round(100*accuracy_score(y_valid,gs.predict(x_valid)),1),"%",sep="", end="")
        print("\tAccuracy Train: ", np.round(100*accuracy_score(y_train,gs.predict(x_train)),1),"%",sep="")
        print("\tBest params: ", gs.best_params_)
        print(pd.DataFrame(confusion_matrix(y_valid, gs.predict(x_valid)),
                     columns=['Predicted 0', 'Predicted 1', 'Predicted 2'],
                     index=['Actual 0', 'Actual 1', 'Actual 2']))
        gs_dict[gs_name] = gs
    return gs_dict

def predict(clf, x_test):
    x_test = pd.DataFrame(x_test).fillna(0)
    y_predict = clf.predict(x_test)
    y_predict = pd.DataFrame(y_predict)
    y_predict[y_predict == 0] = "bad"
    y_predict[y_predict == 1] = "good"
    y_predict[y_predict == 2] = "neutral"
    y_predict.columns = ["Prediction"]
    y_predict.to_csv('result.csv',index_label ="Id")

if __name__ == "__main__":
    csv = load_data()
    # Just ID not need it
    csv = csv.drop(columns=['Id'])
    csv = standarize_numerical_values(csv)
    csv = bool_to_integer(csv)
    
    csv = factorize(csv)
    stats = get_stats(csv)
    # with pd.option_context('display.max_rows', 500, 'display.max_columns', 500, 'display.width', 1000):
        # print(stats)
        
    Y = csv["Alignment"]
    csv = csv.drop("Alignment",axis=1)
    X = csv

    is_train = csv["train"] == 1
    y_train, uniques = pd.factorize(Y[is_train])

    x_train = X[is_train].values
    x_train = pd.DataFrame(x_train).fillna(0)
    
    x_test = X[is_train == False].values

    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train, test_size=0.3,random_state=35)

    gs_dict = model_selection(x_tr,x_valid, y_tr, y_valid)
    clf = gs_dict["LR"]
    predict(clf, x_test)