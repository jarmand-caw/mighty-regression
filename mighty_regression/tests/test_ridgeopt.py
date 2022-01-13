import logging

from mighty_regression.ridge_opt import RidgeOpt

logger = logging.getLogger(__file__)

def test_bfws(data):
    ridger = RidgeOpt(
        data,
        ["SalePrice"]
    )

    best_alpha, best_features, best_r2 = ridger.brute_force_with_selection(
        "lasso",
        0.25,
    )
    logger.info(best_alpha)
    logger.info(best_features)
    logger.info(best_r2)
    assert len(best_features) > 0