import copy
import itertools
import logging

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import MinMaxScaler

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
            self.target = [target]

        if features is None:
            self.features = [x for x in self.df.columns if x not in self.target]
        else:
            self.features = features

        self.model = model

        self.forward_selection_features = None
        self.backward_selection_features = None
        self.lasso_selection_features = None

        self.interaction_features = None

    def create_interaction_features(self, handle_div_zero="fill0"):
        interaction_features = []

        combinations_object = itertools.combinations(self.features, 2)
        combinations_list = [list(x) for x in combinations_object]
        for feature_set in combinations_list:
            feature_name = "*".join(feature_set)
            interaction_features.append(feature_name)
            self.df[feature_name] = self.df[feature_set[0]]*self.df[feature_set[1]]

        permutations_object = itertools.permutations(self.features, 2)
        permutations_list = [list(x) for x in permutations_object]
        for feature_set in permutations_list:
            feature_name = "/".join(feature_set)
            interaction_features.append(feature_name)
            self.df[feature_name] = self.df[feature_set[0]]/self.df[feature_set[1]]
            if handle_div_zero == "fill0":
                self.df[feature_name] = self.df[feature_name].fillna(0)
                self.df[feature_name] = self.df[feature_name].replace(np.inf, 0)
            # Need to implement more options here...

        self.interaction_features = interaction_features
        return interaction_features


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
            use_interaction=False,
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

        if use_interaction:
            if self.interaction_features is None:
                self.create_interaction_features()
            features_to_examine = [x for x in self.features+self.interaction_features if x not in starting_features]
        else:
            features_to_examine = [x for x in self.features if x not in starting_features]

        #If there are no new features left, return the starting features
        if len(features_to_examine) == 0:
            logger.info("All of your features improve performance. Likely makes other feature selection methods moot.")
            self.forward_selection_features = starting_features
            return starting_features, best_r2

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
            return self.forward_selection_features, best_r2

        starting_features.append(curr_best_feature)
        return self.forward_selection(cv_type, num_folds, random, use_interaction, starting_features, best_r2)

    def backward_selection(
            self,
            cv_type="fold",
            num_folds=5,
            random=True,
            use_interaction=False,
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

        if use_interaction:
            if self.interaction_features is None:
                self.create_interaction_features()
            features_to_examine_removal = [x for x in self.features+self.interaction_features if x not in already_removed]
        else:
            features_to_examine_removal = [x for x in self.features if x not in already_removed]

        if len(features_to_examine_removal) == 0:
            self.backward_selection_features = features_to_examine_removal
            logger.info("Backward selection found that none of your features are any good. That sucks.")
            return [], best_r2

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
            return features_to_examine_removal, best_r2
        already_removed.append(curr_best_removed)
        return self.backward_selection(cv_type, num_folds, random, use_interaction, already_removed, best_r2)

    def combinatorics_selection(
            self,
            num_feature_limit=None,
            cv_type="fold",
            num_folds=5,
            random=True,
            use_interaction=False
    ):
        if use_interaction:
            if self.interaction_features is None:
                self.create_interaction_features()
            features = self.interaction_features+self.features
        else:
            features = self.features

        all_combinations = []
        if num_feature_limit is None:
            num_feature_limit = len(features)+1
        else:
            num_feature_limit = num_feature_limit + 1

        for r in range(1, num_feature_limit):
            combinations_object = itertools.combinations(features, r)
            combinations_list = [list(x) for x in combinations_object]
            all_combinations += combinations_list

        best_r2 = -1e5
        best_combination = None
        for feature_combination in all_combinations:
            X = self.df[feature_combination]
            y = self.df[self.target]
            r2 = self.test_metric(X, y, cv_type, num_folds, random)
            if r2 > best_r2:
                best_r2 = r2
                best_combination = feature_combination

        return best_combination, best_r2

    def lasso_selection(
            self,
            num_features=None,
            cv_type="fold",
            num_folds=5,
            random=True,
            use_interaction=False
    ):
        if use_interaction:
            if self.interaction_features is None:
                self.create_interaction_features()
            features = self.interaction_features+self.features
        else:
            features = self.features

        scaler = MinMaxScaler()
        X = self.df[features]
        X_scaled = scaler.fit_transform(X)
        y = self.df[self.target]
        y_scaled = scaler.fit_transform(y)

        feature_dict = dict(zip([x for x in range(len(features)+1)],
                                [None for x in range(len(features)+1)]))
        for x in np.arange(-5, 3, 0.01):
            xx = 10**x
            lasso = Lasso(alpha=xx)
            model = lasso.fit(X_scaled, y_scaled)

            array = np.ones((len(model.coef_), 2))
            array = array.astype(object)
            array[:, 0] = model.coef_
            array[:, 1] = np.array(list(X.columns))
            still_present_features = list(array[:, 1][np.where(np.isclose(array[:, 0].astype(float), 0))])

            n_features = len(still_present_features)
            if feature_dict[n_features] is None:
                feature_dict[n_features] = still_present_features

        # If you want a specified number of features, then just return
        if num_features is not None:
            self.lasso_selection_features = feature_dict[num_features]
            X = self.df[self.lasso_selection_features]
            y = self.df[self.target]
            r2 = self.test_metric(X, y, cv_type, num_folds, random)
            return feature_dict[num_features], r2

        # Else, lets find the combination with the best testing r2
        best_feature_set = None
        best_r2_score = -1e5
        for feature_combination in feature_dict.values():
            if feature_combination is not None:
                X = self.df[feature_combination]
                y = self.df[self.target]
                r2 = self.test_metric(X, y, cv_type, num_folds, random)
                if r2>best_r2_score:
                    best_r2_score = r2
                    best_feature_set = feature_combination

        self.lasso_selection_features = best_feature_set
        return best_feature_set, best_r2_score

    def bagged_selection(
            self,
            occurence_cutoff=2,
            cv_type="fold",
            num_folds=5,
            random=True,
            use_interaction=False
    ):
        self.forward_selection(cv_type, num_folds, random, use_interaction)
        self.backward_selection(cv_type, num_folds, random, use_interaction)
        self.lasso_selection(None, cv_type, num_folds, random, use_interaction)


        total_feature_list = self.forward_selection_features+self.backward_selection_features+self.lasso_selection_features
        total_feature_set = list(set(total_feature_list))
        feature_count_dict = dict(zip(total_feature_set, [0 for x in total_feature_set]))
        for feature in total_feature_set:
            feature_count_dict[feature] = total_feature_list.count(feature)

        all_features = [k for k,v in feature_count_dict.items() if v >= occurence_cutoff]
        return all_features, feature_count_dict




