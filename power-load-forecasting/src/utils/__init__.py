"""
utils package initialization

This package contains utility modules for metrics calculation and visualization.
"""

from .metrics import (
    mean_absolute_percentage_error,
    normalized_mean_absolute_error,
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error,
    r_squared
)

try:
    from .visualization import (
        setup_visualization_style,
        plot_time_series,
        plot_multiple_time_series,
        plot_correlation_matrix,
        plot_feature_importance,
        plot_training_history,
        plot_prediction_intervals
    )
    HAS_VISUALIZATION = True
except ImportError:
    # 可视化依赖可能不可用
    HAS_VISUALIZATION = False

__all__ = [
    "mean_absolute_percentage_error",
    "normalized_mean_absolute_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_squared_error",
    "r_squared"
]

if HAS_VISUALIZATION:
    __all__.extend([
        "setup_visualization_style",
        "plot_time_series",
        "plot_multiple_time_series",
        "plot_correlation_matrix",
        "plot_feature_importance",
        "plot_training_history",
        "plot_prediction_intervals"
    ])