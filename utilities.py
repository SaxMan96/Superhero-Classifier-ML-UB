import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from stats import *


def load_data():
    csv_train = pd.read_csv('superhero-or-supervillain/train.csv').assign(train=1)
    csv_test = pd.read_csv('superhero-or-supervillain/test.csv').assign(train=0)
    csv = pd.concat([csv_train, csv_test], sort=True)
    csv = csv.drop(columns=['Id'])
    #     with pd.option_context('display.max_rows', 500, 'display.max_columns', 500, 'display.width', 1000):
    #         print(get_stats(csv))
    csv = bool_to_integer(csv)
    is_train = csv["train"] == 1
    csv = csv.drop("train", axis=1)
    Y = csv["Alignment"]
    csv = csv.drop("Alignment", axis=1)
    X = csv
    y_train, uniques = pd.factorize(Y[is_train])
    X_train = X[is_train]
    X_test = X[is_train == False]
    return X_train, y_train, X_test


def bool_to_integer(csv) -> pd.DataFrame():
    for col in csv.columns:
        if csv[col].dtype == bool:
            csv[col] = csv[col].astype(int)
    return csv


# def standarize_numerical_values(csv):
#     for col in csv.columns:
#         if col == 'train':
#             continue
#         if csv[col].dtype == np.float64:
#             data = csv[col]
#             std = data.std()
#             data = data[(data < data.quantile(0.99)) & (data > data.quantile(0.01))]
#             mean = data.mean()
#             csv[col] = (csv[col] - mean) / std
#     #             _ = plt.hist(csv[col], bins='auto', alpha = 0.5)
#     #             plt.yscale('log')
#     #             plt.title(f"Distr in {col} column")
#     #             plt.show()
#     return csv


# def check_rows(csv):
#     for row in range(len(csv)):
#         print(row, csv.iloc[row].isna().sum())
#     return csv


# def distribution_in_columns(csv):
#     for col in list(csv):
#         print(csv[col].value_counts())
#     return csv


# def plot_dist_y(csv):
#     plt.pie([len(csv[csv['Alignment'] == 'good']), len(csv[csv['Alignment'] == 'bad']),
#              len(csv[csv['Alignment'] == 'neutral'])], labels=['good', 'bad', 'neutral'])
#     plt.show()
#     return csv


# def factorize(csv) -> pd.DataFrame():
#     for col in csv.select_dtypes(include=['object']).columns:
#         if col == "Alignment":
#             continue
#         dummy = pd.get_dummies(csv[col])
#         dummy.columns = [col + " " + x for x in dummy.columns]
#         dummy = dummy.drop([dummy.columns[-1]], axis=1)
#         csv = csv.drop(col, axis=1)
#         csv = pd.concat([csv, dummy], axis=1)
#     return csv


def prepare_models(clfs_to_test, use_scaler=False, use_grid_search=False, verbose=False):
    gs_dict = {}
    clfs_dict = {
        'DTC': DecisionTreeClassifier(random_state=0),
        'RFC': RandomForestClassifier(random_state=0),
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression(random_state=35, solver='liblinear', multi_class='auto'),
        'XGB': XGBClassifier(random_state=0),
        'SVC': SVC(random_state=0)
    }

    param_grids = {
        'DTC': {
            '__class_weight': [None],
            '__presort': [False],
            '__max_depth': [None],
            '__criterion': ['gini', 'entropy'],
            '__min_samples_split': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            '__min_weight_fraction_leaf': np.linspace(0.05, 0.4, 5)
        },
        'RFC': {
            # '__n_estimators': [50],
            # '__max_depth': [20],
            # '__criterion': ['entropy'],
            '__class_weight': [None],
            '__n_estimators': [15, 50, 100],
            '__max_depth': [5, 10, 20],
            '__class_weight': [None, 'balanced'],
            '__max_features': list(range(16, 32, 3)) + ['auto']
        },
        'KNN': {
            # '__n_neighbors': [7],
            # '__weights': ['distance'],
            '__n_neighbors': range(7, 13),
            '__weights': ['uniform', 'distance'],
        },
        'LR': {
            # "__C": [0.1],
            # "__penalty": ["l1"]
            "__C": np.logspace(-3, 3, 10),
            "__penalty": ["l1", "l2"]
        },
        'XGB': {
            '__colsample_bytree': [0.6],
            '__gamma': [0.5],
            '__max_depth': [4],
            '__min_child_weight': [1],
            '__n_estimators': [100],
            '__subsample': [0.6]
            # '__n_estimators': [80, 100,200,300],
            # '__min_child_weight': [1, 5, 10],
            # '__gamma': [0.5, 1, 1.5, 2, 5],
            # '__subsample': [0.6, 0.8, 1.0],
            # '__colsample_bytree': [0.6, 0.8, 1.0],
            # '__max_depth': [3, 4, 5]
        },

        'SVC': {
            '__C': [1],
            '__decision_function_shape': ['ovo'],
            '__gamma': [1],
            '__kernel': ['rbf'],
            '__shrinking': [True]
            # '__kernel': ('linear', 'rbf'),
            # '__C': [1],
            # '__gamma': (1, 2, 'auto'),
            # '__decision_function_shape': ('ovo', 'ovr'),
            # '__shrinking': (True, False)
        }

    }
    scorer = make_scorer(f1_score, average='micro')
    for clf_name in {your_key: clfs_dict[your_key] for your_key in clfs_to_test}:
        if use_scaler:
            pipe_line = Pipeline([('scaler', StandardScaler()), ('', clfs_dict[clf_name])])
        else:
            pipe_line = clfs_dict[clf_name]

        if use_grid_search:
            gs = GridSearchCV(pipe_line, param_grid=param_grids[clf_name], cv=8, scoring=scorer, verbose=verbose,
                              iid=True)
        else:
            gs = pipe_line
        gs_dict[clf_name] = gs
    return gs_dict


def model_selection(clfs_to_test, x_train, x_valid, y_train, y_valid):
    gs_dict = prepare_models(clfs_to_test, use_scaler=True, use_grid_search=True, verbose=False)
    for gs_name in gs_dict:
        print("\n==================================================================\n", gs_name, end="\t")
        gs = gs_dict[gs_name].fit(x_train, y_train)
        print("\tAccuracy Train: ", np.round(100 * accuracy_score(y_train, gs.predict(x_train)), 1), "%", sep="",
              end="")
        print("\tAccuracy Valid: ", np.round(100 * accuracy_score(y_valid, gs.predict(x_valid)), 1), "%", sep="")
        print("\tBest params: ")
        for param in gs.best_params_:
            print("\t\t\'", param, "\': ", "[", gs.best_params_[param], "]", sep="")

        print(pd.DataFrame(confusion_matrix(y_valid, gs.predict(x_valid))))
        # columns=['Predicted 0', 'Predicted 1', 'Predicted 2'],
        # index=['Actual 0', 'Actual 1', 'Actual 2']))
        gs_dict[gs_name] = gs
    return [(k, v) for k, v in gs_dict.items()]


def feature_encoding(X_train, y_train, X_test, method='ordinal'):
    columns = list(X_train.select_dtypes(include=['object']).columns)
    if method == "ordinal":
        ce_binary = ce.OrdinalEncoder(cols=columns)
    elif method == "binary":
        ce_binary = ce.BinaryEncoder(cols=columns)
    elif method == "onehot":
        ce_binary = ce.OneHotEncoder(cols=columns)
    elif method == "basen":
        ce_binary = ce.BaseNEncoder(cols=columns)
    elif method == "hashing":
        ce_binary = ce.HashingEncoder(cols=columns)
    else:
        raise Exception("Wrong Method Choosen!")

    X_train = ce_binary.fit_transform(X_train, y_train)
    X_test = ce_binary.transform(X_test)
    return X_train.values, y_train, X_test.values


def filling_nans(data, method="method"):
    data_filled = []
    for df_org in data:
        df = df_org.copy()
        for label in df.columns:
            # if method == "randomdistrubuted":
            #     df[label].fillna(lambda x: random.choice(df[df[label] != np.nan][label]), inplace=True)
            if method == "new_value":
                df[label].fillna(0, inplace=True)
            elif method == "mostfrequent":
                df[label].fillna(df[label].value_counts().idxmax(), inplace=True)
        data_filled.append(df)
    return data_filled


def perform_final_training(clf, x_tr, y_tr):
    return clf.fit(x_tr, y_tr)


def upsampling(x_tr, y_tr, method="smote"):
    if method == "smote":
        sm = SMOTE(random_state=0)
    return sm.fit_resample(x_tr, y_tr)


def dim_reduce(x_tr, x_valid, x_test, method="pca"):
    if method == "pca":
        x_tr = StandardScaler().fit_transform(x_tr)
        pca = PCA(n_components=0.9, svd_solver='full', random_state=0).fit(x_tr)
        x_tr = pca.transform(x_tr)
        x_valid = pca.transform(x_valid)
        x_test = pca.transform(x_test)
        print("pca.components: ", len(pca.components_))
    return x_tr, x_valid, x_test


def pretrain_predict(clf, x_test, x_tr, y_tr, f_name):
    clf = perform_final_training(clf, x_tr, y_tr)
    return predict(clf, x_test, f_name)


def predict(clf, x_test, f_name):
    y_predict = clf.predict(x_test)
    y_predict_df = pd.DataFrame(y_predict)
    y_predict_df[y_predict_df == 0] = "bad"
    y_predict_df[y_predict_df == 1] = "good"
    y_predict_df[y_predict_df == 2] = "neutral"
    y_predict_df.columns = ["Prediction"]
    y_predict_df.to_csv(f_name, index_label="Id")
    return y_predict
