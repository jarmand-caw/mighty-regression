from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
import copy
import logging

logger = logging.getLogger(__file__)


class FeatureSelection(object):
    def __init__(
            self,
            df,
            target,
            features=None,
            model=LinearRegression()
    ):
        self.df = df
        if type(target) == list:
            self.target = target
        else:
            self.target = list(target)

        if features is None:
            self.features = [x for x in self.df.columns if x not in self.target]
        else:
            self.features = features

        self.model = model

        self.forward_selection_features = None
        self.backward_selection_features = None

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

    def forward_selection(
            self,
            cv_type="fold",
            num_folds=5,
            random=True,
            starting_features=None,
            best_r2=-1e5,
    ):
        """

        Args:
            cv_type: one of "fold", "loo", float. If float, uses that portion of the data as a test set
            num_folds: int. only used if cv_type is fold. Number of folds in kfold cv
            random: bool. only used of cv_type is float. Determines if test set is generated randomly, or if last portion is used
            starting_features: list of features you want to start with. defaults to none
            best_r2: r2 of the starting features. Defaults to -1e5
        Returns:
            list. The best features via. forward selection
        """
        if starting_features is None:
            starting_features = []

        features_to_examine = [x for x in self.features if x not in starting_features]

        #If there are no new features left, return the starting features
        if len(features_to_examine) == 0:
            logger.info("All of your features improve performance. Likely makes other feature selection methods moot.")
            self.forward_selection_features = starting_features
            return starting_features

        curr_best_feature = None
        for feature in features_to_examine:
            this_list = copy.copy(starting_features)
            this_list.append(feature)
            X = self.df[this_list]
            y = self.df[self.target]
            r2 = self.test_metric(X, y, cv_type, num_folds, random)
            if r2 > best_r2:
                best_r2 = r2
                curr_best_feature = feature

        #If none of the features made an improvement, return the starting features
        if curr_best_feature is None:
            self.forward_selection_features = starting_features
            logger.info("Your forward selection features are {}.".format(starting_features))
            return self.forward_selection_features

        starting_features.append(curr_best_feature)
        logger.info(curr_best_feature)
        return self.forward_selection(cv_type, num_folds, random, starting_features, best_r2)

    def backward_selection(
            self,
            cv_type="fold",
            num_folds=5,
            random=True,
            already_removed=None,
            best_r2=-1e5,
    ):
        """

        Args:
            cv_type: one of "fold", "loo", float. If float, uses that portion of the data as a test set
            num_folds: int. only used if cv_type is fold. Number of folds in kfold cv
            random: bool. only used of cv_type is float. Determines if test set is generated randomly, or if last portion is used
            starting_features: list of features you want to start with. defaults to none
            best_r2: r2 of the starting features. Defaults to -1e5
        Returns:
            list. The best features via. backward selection
        """

        if already_removed is None:
            already_removed = []

        features_to_examine_removal = [x for x in self.features if x not in already_removed]

        if len(features_to_examine_removal) == 0:
            self.backward_selection_features = features_to_examine_removal
            logger.info("Backward selection found that none of your features are any good. That sucks.")
            return []

        curr_best_removed = None
        for feature in features_to_examine_removal:
            this_list = copy.copy(features_to_examine_removal)
            this_list.remove(feature)
            X = self.df[this_list]
            y = self.df[self.target]
            r2 = self.test_metric(X, y, cv_type, num_folds, random)
            if r2 > best_r2:
                best_r2 = r2
                curr_best_removed = feature

        # If none of the features made an improvement, return the starting features
        if curr_best_removed is None:
            self.backward_selection_features = features_to_examine_removal
            logger.info("Your backward selection features are {}.".format(features_to_examine_removal))
            return features_to_examine_removal
        already_removed.append(curr_best_removed)
        return self.backward_selection(cv_type, num_folds, random, already_removed, best_r2)

