from .filters import (
    method_variance, method_anova, method_mi, method_mrmr, method_relieff
)
from .embedded import (
    method_l1_logistic, method_elasticnet, method_boruta,
    method_linearsvc_l1, method_extratrees
)
from .wrappers import method_rfecv, method_ga, method_bpso
from .published_ea import method_sfe, method_mel

MAIN_METHODS = {
    "variance":    method_variance,
    "anova":       method_anova,
    "mi":          method_mi,
    "mrmr":        method_mrmr,
    "relieff":     method_relieff,
    "l1_logistic": method_l1_logistic,
    "elasticnet":  method_elasticnet,
    "boruta":      method_boruta,
    "linearsvc_l1": method_linearsvc_l1,
    "extratrees":  method_extratrees,
    "rfecv":       method_rfecv,
    "ga":          method_ga,
    "bpso":        method_bpso,
}

SUPPLEMENTARY_METHODS = {
    "sfe": method_sfe,
    "mel": method_mel,
}

ALL_METHODS = {**MAIN_METHODS, **SUPPLEMENTARY_METHODS}

__all__ = ["MAIN_METHODS", "SUPPLEMENTARY_METHODS", "ALL_METHODS"]
