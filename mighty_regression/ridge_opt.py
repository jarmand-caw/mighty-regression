import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import MinMaxScaler

from mighty_regression.features import FeatureSelection


class RidgeOpt(object):
    def __init__(
            self,
            df,
            target,
            features=None
    ):
        self.scaler = MinMaxScaler()
        scaled_array = self.scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_array, columns=list(df.columns))
        self.df = scaled_df

        if type(target) == list:
            self.target = target
        else:
            self.target = [target]

        if features is None:
            self.features = [x for x in self.df.columns if x not in self.target]
        else:
            self.features = features

        self.best_features = None
        self.best_alpha = None

    def test_metric(self, X, y, cv_type="fold", num_folds=5, random=True):
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

                regression_obj = copy.copy(self.model)
                model = regression_obj.fit(X_train, y_train)
                pred = model.predict(X_test)

                all_preds += list(pred)
                all_true += list(y_test)

            return r2_score(all_true, all_preds)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=random, random_state=42, test_size=cv_type)
            regression_obj = copy.copy(self.model)
            model = regression_obj.fit(X_train, y_train)
            pred = model.predict(X_test)
            return r2_score(y_test, pred)

    def brute_force_with_selection(
            self,
            selection_type="backward",
            cv_type="fold",
            num_folds=5,
            random=True,
    ):
        best_r2 = -1e5
        best_features = None
        best_alpha = None

        for x in np.arange(-3, 1, 0.1):
            alpha = 10**x
            selector = FeatureSelection(
                copy.copy(self.df),
                self.target,
                self.features,
                model=Ridge(alpha)
            )
            if selection_type == "backward":
                features, r2 = selector.backward_selection(cv_type, num_folds, random)
            elif selection_type == "forward":
                features, r2 = selector.forward_selection(cv_type, num_folds, random)
            elif selection_type == "lasso":
                features, r2 = selector.lasso_selection(None, cv_type, num_folds, random)
            else:
                raise NotImplementedError("selection type not implemented. Options are forward, backward, lasso")

            if r2 > best_r2:
                best_r2 = r2
                best_features = features
                best_alpha = alpha

        self.best_alpha = best_alpha
        self.best_features = best_features

        return best_alpha, best_features, best_r2


