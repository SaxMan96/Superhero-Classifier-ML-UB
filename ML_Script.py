import pickle
import warnings

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

from utilities import *

warnings.filterwarnings("ignore")

clfs_to_test = ['DTC', 'RFC', 'KNN', 'LR', 'XGB']

if __name__ == "__main__":
    X, y_train = load_data()
    y_train[y_train == 2] = 1
    X_train, X_test = manual_preprocessing(X)
    X_train, X_test = filling_nans((X_train, X_test), method="new_value")
    x_train, y_train, x_test = feature_encoding(X_train, y_train, X_test, method="hashing")
    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    x_tr, y_tr = upsampling(x_tr, y_tr, method="smote")
    x_tr, x_valid, x_test = dim_reduce(x_tr, x_valid, x_test, method="pca")
    gs_list = model_selection(clfs_to_test, x_tr, x_valid, y_tr, y_valid)

    with open('gs_list.pickle', 'wb') as handle:
        pickle.dump(gs_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('gs_list.pickle', 'rb') as handle:
        gs_list = pickle.load(handle)
    print("========ensamble================")
    weights_test = [[0.0, 1, 0.2, 0.6, 0.8],
                    [0.0, 1, 0.2, 0.6, 0.6],
                    [0.0, 1, 0.4, 0.8, 0.6],
                    [0.0, 1, 0.4, 0.8, 0.8]]

    for weights in weights_test:
        print(weights)
        ensemble = VotingClassifier(gs_list, voting='soft', n_jobs=-1, weights=weights)
        ensemble.fit(x_tr, y_tr)
        score = ensemble.score(x_valid, y_valid)
        print(ensemble.score(x_valid, y_valid))
        a = predict(ensemble, x_test, "result_only_train" + str(score) + ".csv")
        b = pretrain_predict(ensemble, x_test, x_train, y_train, "result_train_valid" + str(score) + ".csv")
        print(np.sum((a - b) ** 2))
