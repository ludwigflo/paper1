from .classification_metrics import ClassificationMetrics
from typing import Tuple


def init_metrics_object(metric_params: dict) -> Tuple[ClassificationMetrics, dict]:
    """
    Initializes an object, which is used to compute the validation metric_object.

    Parameters
    ----------
    metric_params: Parameters for the metric_object object are defined.

    Returns
    -------
    metric_object: Classification metric_object object, which can be used for computing various classification metric_object.
    """

    # create the object
    metric_object = ClassificationMetrics(metric_params['inverse_class_mapping'])
    return metric_object
