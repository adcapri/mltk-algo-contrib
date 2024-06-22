#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor

from base import RegressorMixin, BaseAlgo
from codec import codecs_manager
from util.param_util import convert_params

class AdaBoostRegressor(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=[
                'random_state',
                'n_estimators',
            ],
            strs=['loss'],
            floats=['learning_rate'],
        )

        self.estimator = _AdaBoostRegressor(**out_params)

    def apply(self, df, options):
        # needed for backward compatibility with sklearn 0.17
        # since n_features_ was added in version 0.18
        self.estimator.n_features_ = len(self.columns)
        return super(AdaBoostRegressor, self).apply(df, options)
        
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
