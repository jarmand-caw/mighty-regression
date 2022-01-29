import logging

from mighty_regression.features import FeatureSelection

logger = logging.getLogger(__file__)

def test_feature_creation(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )
    prev_len = len(selector.df.columns)
    selector.create_interaction_features()
    assert prev_len<len(selector.df.columns)

def test_forward_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )
    best_features, best_r2 = selector.forward_selection(cv_type="fold", num_folds=3, use_interaction=True)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0

    best_features, best_r2 = selector.forward_selection(cv_type="loo")
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0

    best_features, best_r2 = selector.forward_selection(cv_type=0.25, random=True)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0

def test_backward_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )

    best_features, best_r2 = selector.backward_selection(cv_type="fold", num_folds=3)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0

def test_combinatorics_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )

    best_features, best_r2 = selector.combinatorics_selection(3, .25, random=True)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0


def test_lasso_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )

    best_features, best_r2 = selector.lasso_selection(cv_type="fold", num_folds=3)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0

def test_bagged_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )

    best_features, feature_count_dict = selector.bagged_selection(cv_type="fold", num_folds=3)
    logger.info(best_features)
    logger.info(feature_count_dict)
    assert len(best_features) > 0