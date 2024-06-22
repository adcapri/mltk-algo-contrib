#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor

from base import RegressorMixin, BaseAlgo
from util.param_util import convert_params
from util.algo_util import handle_max_features
from codec import codecs_manager


class ExtraTreesRegressor(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)
        params = options.get('params', {})
        out_params = convert_params(
            params,
            floats=['max_samples', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'min_impurity_split'],
            bools=['bootstrap', 'oob_score', 'warm_start'],
            ints=['n_estimators', 'max_depth', 'max_leaf_nodes', 'min_impurity_decrease'],
            strs=['criterion'],
        )

        self.estimator = _ExtraTreesRegressor(**out_params)

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError('"%s" models do not take options for summarization' % self.__class__.__name__)
        df = DataFrame({
            'feature': self.columns,
            'importance': self.estimator.feature_importances_.ravel()
        })
        return df
        
    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec('algos_contrib.ExtraTreesRegressor',
                                 'ExtraTreesRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble._forest',
                                 'ExtraTreesRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._classes', 'ExtraTreeRegressor',
                                 SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
