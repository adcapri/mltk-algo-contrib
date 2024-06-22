#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor

from base import RegressorMixin, BaseAlgo
from util.param_util import convert_params
from util.algo_util import handle_max_features
from codec import codecs_manager


class AdaBoostRegressor(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)
        params = options.get('params', {})
        out_params = convert_params(
            params,
            strs=['loss'],
            floats=['learning_rate'],
            ints=['random_state','n_estimators'],
        )

        self.estimator = _AdaBoostRegressor(**out_params)

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError(
                '"%s" models do not take options for summarization' % self.__class__.__name__
            )
        df = DataFrame(
            {'feature': self.columns, 'importance': self.estimator.feature_importances_.ravel()}
        )
        return df
        
    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec(
            'algos_contrib.AdaBoostRegressor', 'AdaBoostRegressor', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.ensemble._weight_boosting', 'AdaBoostRegressor', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.tree._classes', 'DecisionTreeRegressor', SimpleObjectCodec
        )
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
