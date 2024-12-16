from .losses import ce, fl, mae, gce, phuber_ce, taylor_ce, \
                    js_loss, sce, nce_rce, lc_ce, nce_nnce
from .losses_ogc import ogc_ce, ogc_fl, ogc_gce

__all__ = [
    'ce',
    'fl',
    'mae',
    'gce',
    'phuber_ce',
    'taylor_ce',
    'js_loss',
    'sce',
    'nce_rce',
    'nce_agce',
    'lc_ce',
    'nce_nnce',

    # ogc
    'ogc_ce',
    'ogc_fl',
    'ogc_gce'
]