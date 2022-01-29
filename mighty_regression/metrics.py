import copy

from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split


def get_test_metric(model, X, y, cv_type="fold", num_folds=5, random=True):
    if cv_type in ["fold", "loo"]:
        X_array = X.values
        y_array = y.values

        if cv_type == "fold":
            splitter = KFold(num_folds)
        else:
            splitter = LeaveOneOut()

        all_preds = []
        all_true = []
        for train_idx, test_idx in splitter.split(X_array):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            regression_obj = copy.copy(model)
            fit_model = regression_obj.fit(X_train, y_train)
            pred = fit_model.predict(X_test)

            all_preds += list(pred)
            all_true += list(y_test)

        return r2_score(all_true, all_preds)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=random, random_state=42, test_size=cv_type)
        regression_obj = copy.copy(model)
        fit_model = regression_obj.fit(X_train, y_train)
        pred = fit_model.predict(X_test)

        return r2_score(y_test, pred)