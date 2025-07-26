"""
experiments.code package initialization

This package contains the core experiment code modules including training, 
evaluation, visualization, and version control functionality.
"""

# 检查模块是否存在再导入
try:
    from .train import train_model_task
    HAS_TRAIN = True
except ImportError:
    HAS_TRAIN = False

try:
    from .evaluate import evaluate_model_task
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False

try:
    from .visualize import visualize_results_task
    HAS_VISUALIZE = True
except ImportError:
    HAS_VISUALIZE = False

try:
    from .version_control import ExperimentVersionControl
    HAS_VERSION_CONTROL = True
except ImportError:
    HAS_VERSION_CONTROL = False

__all__ = []

if HAS_TRAIN:
    __all__.append("train_model_task")

if HAS_EVALUATE:
    __all__.append("evaluate_model_task")

if HAS_VISUALIZE:
    __all__.append("visualize_results_task")

if HAS_VERSION_CONTROL:
    __all__.append("ExperimentVersionControl")