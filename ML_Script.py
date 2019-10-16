# Imports

import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm_notebook as tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, make_scorer, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# Load Data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import XGBClassifier
from sklearn.svm import SVC
import category_encoders as ce


def load_data():
    csv_train = pd.read_csv('superhero-or-supervillain/train.csv').assign(train = 1) 
    csv_test = pd.read_csv('superhero-or-supervillain/test.csv').assign(train = 0) 
    csv = pd.concat([csv_train,csv_test], sort=True)    
    csv = csv.drop(columns=['Id'])
#     with pd.option_context('display.max_rows', 500, 'display.max_columns', 500, 'display.width', 1000):
#         print(get_stats(csv))
    csv = bool_to_integer(csv)
    is_train = csv["train"] == 1
    csv = csv.drop("train",axis=1)
    Y = csv["Alignment"]
    csv = csv.drop("Alignment",axis=1)
    X = csv
    y_train, uniques = pd.factorize(Y[is_train])
    X_train = X[is_train]
    X_test = X[is_train == False]
    return X_train, y_train, X_test

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
        # 'DTC': DecisionTreeClassifier(random_state=0),
        # 'RFC': RandomForestClassifier(random_state=0),
        # 'KNN': KNeighborsClassifier(),
        # 'LR': LogisticRegression(random_state=35),
        'XGB': XGBClassifier(random_state=0),
        # 'SVC': SVC(random_state=0)
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
#             '__max_features' : list(range(6,32,5))+['auto']
        },
        'KNN': {
            '__n_neighbors': range(1, 12),
            '__weights': ['uniform', 'distance'],
        },
        'LR': {
            "__C": np.logspace(-3,3,10), 
            "__penalty":["l1","l2"]
        },
        'XGB': {
            '__colsample_bytree': [0.6], 
            '__gamma': [0.5], 
            '__max_depth': [4], 
            '__min_child_weight': [1], 
            '__n_estimators': [100], 
            '__subsample': [0.6]
        },
# '__n_estimators': [80, 100,200,300],
# '__min_child_weight': [1, 5, 10],
# '__gamma': [0.5, 1, 1.5, 2, 5],
# '__subsample': [0.6, 0.8, 1.0],
# '__colsample_bytree': [0.6, 0.8, 1.0],
# '__max_depth': [3, 4, 5]        
        'SVC': {
            '__kernel':('linear', 'rbf'), 
            '__C': [1],
            '__gamma': (1,2,'auto'),
            '__decision_function_shape':('ovo','ovr'),
            '__shrinking':(True,False)
        }
# '__C': 1, 
# '__decision_function_shape': 'ovo', 
# '__gamma': 1, 
# '__kernel': 'rbf', 
# '__shrinking': True
    }
    scorer = make_scorer(f1_score, average='micro')            
    for clf_name in clfs_dict:
        if use_scaler:
            pipe_line = Pipeline([('scaler', StandardScaler()), ('', clfs_dict[clf_name])])
        else:
            pipe_line = clfs_dict[clf_name]
            
        if use_grid_search:
            gs = GridSearchCV(pipe_line, param_grid=param_grids[clf_name], scoring=scorer, verbose=verbose, iid=True)
        else:
            gs = pipe_line
        gs_dict[clf_name] = gs
    return gs_dict
        
def model_selection(x_train, x_valid, y_train, y_valid):
    gs_dict = prepare_models(use_scaler = True, use_grid_search = True, verbose=False)
    for gs_name in gs_dict:        
        print("\n==================================================================\n", gs_name, end="\t")
        gs = gs_dict[gs_name].fit(x_train,y_train)
        print("\tAccuracy Train: ", np.round(100*accuracy_score(y_train,gs.predict(x_train)),1),"%",sep="", end="")
        print("\tAccuracy Valid: ", np.round(100*accuracy_score(y_valid,gs.predict(x_valid)),1),"%",sep="")
        print("\tBest params: ", gs.best_params_)
        print(pd.DataFrame(confusion_matrix(y_valid, gs.predict(x_valid)),
                     columns=['Predicted 0', 'Predicted 1', 'Predicted 2'],
                     index=['Actual 0', 'Actual 1', 'Actual 2']))
        gs_dict[gs_name] = gs
    return gs_dict

def feature_encoding(X_train, y_train, X_test, method='ordinal'):
    columns = list(X_train.select_dtypes(include=['object']).columns)    
    if method == "ordinal":
        ce_binary = ce.OrdinalEncoder(cols = columns)
    elif method == "binary":
        ce_binary = ce.BinaryEncoder(cols = columns)
    elif method == "onehot":
        ce_binary = ce.OneHotEncoder(cols = columns)
    elif method == "basen":
        ce_binary = ce.BaseNEncoder(cols = columns)  
    elif method == "hashing":
        ce_binary = ce.HashingEncoder(cols = columns)           
    else:
        raise Exception("Wrong Method Choosen!")
    
    X_train= ce_binary.fit_transform(X_train, y_train)
    X_test = ce_binary.transform(X_test)
    return X_train.values, y_train, X_test.values

def filling_nans(data, method = "method"):
    data_filled = []
    for df_org in data:
        df = df_org.copy()
        for label in df.columns:
            if method == "randomdistrubuted":
                df[label].fillna(lambda x: random.choice(df[df[label] != np.nan][label]), inplace =True)
            elif method=="new_value":
                df[label].fillna(0, inplace =True)
            elif method=="mostfrequent":
                df[label].fillna(df[label].value_counts().idxmax(), inplace =True)
        data_filled.append(df)
    return data_filled

def perform_final_training(clf, x_tr, y_tr):
    return clf.fit(x_tr, y_tr)
    
def upsampling(x_tr, y_tr, method="smote"):
    sm = SMOTE(random_state=0)
    return sm.fit_resample(x_tr, y_tr)
    
def dim_reduce(x_tr, x_valid, x_test, method="pca"):
    if method=="pca":
        x_tr = StandardScaler().fit_transform(x_tr)
        pca = PCA(n_components=0.9, svd_solver='full', random_state=0).fit(x_tr)
        x_tr = pca.transform(x_tr)
        x_valid = pca.transform(x_valid)
        x_test = pca.transform(x_test)
        # print(len(pca.components_))
    return x_tr, x_valid, x_test
    
def pretrain_predict(clf, x_test, x_tr, y_tr, f_name):
    clf = perform_final_training(clf, x_tr, y_tr)
    predict(clf, x_test, f_name)
    
def predict(clf, x_test, f_name):
#     x_test = pd.DataFrame(x_test)
    y_predict = clf.predict(x_test)
    y_predict = pd.DataFrame(y_predict)
    y_predict[y_predict == 0] = "bad"
    y_predict[y_predict == 1] = "good"
    y_predict[y_predict == 2] = "neutral"
    y_predict.columns = ["Prediction"]
    y_predict.to_csv(f_name,index_label ="Id")
        
if __name__ == "__main__":
    X_train, y_train, X_test = load_data() #X_train and X_test are Dataframes
    X_train, X_test = filling_nans((X_train, X_test), method = "new_value")
    x_train, y_train, x_test = feature_encoding(X_train, y_train, X_test, method="onehot") #x_train and x_test are nd.arrays
    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train, test_size=0.3,random_state=0)
    x_tr, x_valid, x_test = dim_reduce(x_tr, x_valid, x_test, method="pca")
    x_tr, y_tr = upsampling(x_tr, y_tr, method="smote")
    gs_dict = model_selection(x_tr,x_valid, y_tr, y_valid)
    
    clf = gs_dict["XGB"]
    pretrain_predict(clf, x_test, x_tr, y_tr, "result_only_train.csv")


# if __name__ == "__main__":
    # print("SMOTE")
    # , "new_value", "mostfrequent"
    # for fill_nans_method in ["new_value", "mostfrequent"]:
        # print(fill_nans_method)
        # ,"ordinal", "binary",  "basen", "hashing"
        # for encoding_method in ["onehot"]:
            # print("\t", encoding_method)
            # X_train, y_train, X_test = load_data() #X_train and X_test are Dataframes
            # X_train = filling_nans(X_train, method = fill_nans_method)
            # X_test = filling_nans(X_test, method = fill_nans_method)
            # x_train, y_train, x_test = feature_encoding(X_train, y_train, X_test, method=encoding_method) #x_train and x_test are nd.arrays
            # x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train, test_size=0.3,random_state=0)

            # x_tr, x_valid, y_tr, y_valid = pca_decomp(x_tr, x_valid, y_tr, y_valid)     
            # for i in [x_tr, y_tr, x_valid, y_valid]:
                # print(i.shape)
            # sm = SMOTE(random_state=0)
            # x_tr, y_tr = sm.fit_resample(x_tr, y_tr)
            

            # gs_dict = model_selection(x_tr,x_valid, y_tr, y_valid)
            # print("\t\t", gs_dict, np.mean(list(gs_dict.values())))
    # clf = gs_dict["LR"]
    # predict(clf, x_test)