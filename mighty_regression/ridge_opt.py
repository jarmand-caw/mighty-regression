import copy
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from mighty_regression.features import FeatureSelection
from mighty_regression.metrics import get_test_metric

logger = logging.getLogger(__file__)

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

    def brute_force_opt(
            self,
            features=None,
            cv_type="fold",
            num_folds=5,
            random=True
    ):
        best_r2 = -1e5
        best_alpha = None

        for x in np.arange(-3, 1, 0.1):
            alpha = 10 ** x
            ridge_reg = Ridge(alpha=alpha)

            if features is None:
                X = self.df[self.features]
            else:
                assert type(features) == list, "Features must be a list. Currently is {}".format(type(features))
                X = self.df[features]

            y = self.df[self.target]
            r2 = get_test_metric(ridge_reg, X, y, cv_type, num_folds, random)

            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        logger.info("Best alpha: {} with an R^2 of {}".format(best_alpha, best_r2))
        return best_alpha, best_r2

    def feature_select_given_alpha(
            self,
            alpha,
            selection_type="backward",
            cv_type="fold",
            num_folds=5,
            random=True
    ):
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

        return features, r2


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
            features, r2 = self.feature_select_given_alpha(alpha, selection_type, cv_type, num_folds, random)

            if r2 > best_r2:
                best_r2 = r2
                best_features = features
                best_alpha = alpha

        self.best_alpha = best_alpha
        self.best_features = best_features

        logger.info("Best feature/alpha combo is {}/{} with an R^2 of {}".format(best_features, best_alpha, best_r2))
        return best_alpha, best_features, best_r2

    def optuna_opt(
            self,
            features=None,
            num_trials=100,
            cv_type="fold",
            num_folds=5,
            random=True
    ):
        def objective(trial):
            alpha = trial.suggest_float('alpha', -1e-4, 10, log=True)
            ridge_reg = Ridge(alpha=alpha)
            if features is None:
                X = self.df[self.features]
            else:
                assert type(features) == list, "Features must be a list. Currently is {}".format(type(features))
                X = self.df[features]
            y = self.df[self.target]
            r2 = get_test_metric(ridge_reg, X, y, cv_type, num_folds, random)
            return r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_trials)
        best_alpha = study.best_params["alpha"]
        best_r2 = study.best_trial.value

        logger.info("Best alpha: {} with an R^2 of {}".format(best_alpha, best_r2))

        return best_alpha, best_r2

    def optuna_with_selection(
            self,
            num_trials=100,
            selection_type="backward",
            cv_type="fold",
            num_folds=5,
            random=True
    ):
        def objective(trial):
            alpha = trial.suggest_float('alpha', -1e-4, 10, log=True)
            _, r2 = self.feature_select_given_alpha(alpha, selection_type, cv_type, num_folds, random)
            return r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_trials)
        best_alpha = study.best_params["alpha"]
        best_r2 = study.best_trial.value
        best_features, _ = self.feature_select_given_alpha(best_alpha, selection_type, cv_type, num_folds, random)

        logger.info("Best feature/alpha combo is {}/{} with an R^2 of {}".format(best_features, best_alpha, best_r2))

        return best_alpha, best_features, best_r2


