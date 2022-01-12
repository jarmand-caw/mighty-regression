import logging
from mighty_regression.features import FeatureSelection

logger = logging.getLogger(__file__)

def test_forward_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )
    best_features = selector.forward_selection(cv_type="fold", num_folds=3)
    logger.info(selector.forward_selection_features)
    logger.info(best_features)
    assert len(best_features) > 0

    best_features = selector.forward_selection(cv_type="loo")
    logger.info(best_features)
    assert len(best_features) > 0

    best_features = selector.forward_selection(cv_type=0.25, random=True)
    logger.info(best_features)
    assert len(best_features) > 0

def test_backward_selection(data):
    selector = FeatureSelection(
        data,
        ["SalePrice"],
    )

    best_features = selector.backward_selection(cv_type="fold", num_folds=3)
    logger.info(best_features)
    assert len(best_features) > 0