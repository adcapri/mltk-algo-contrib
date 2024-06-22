#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier

from base import ClassifierMixin, BaseAlgo
from codec import codecs_manager
from util.param_util import convert_params

class BaggingClassifier(ClassifierMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=[
                'random_state',
                'n_estimators',
            ],
            strs=['max_features'],
        )


        if 'max_features' in out_params:
            # Handle None case
            if out_params['max_features'].lower() == "none":
                out_params['max_features'] = None
            else:
                # EAFP... convert max_features to int if it is a number.
                try:
                    out_params['max_features'] = float(out_params['max_features'])
                    max_features_int = int(out_params['max_features'])
                    if out_params['max_features'] == max_features_int:
                        out_params['max_features'] = max_features_int
                except:
                    pass


        self.estimator = _BaggingClassifier(**out_params)

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
            'algos_contrib.BaggingClassifier', 'BaggingClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.ensemble._bagging', 'BaggingClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.tree._classes', 'DecisionTreeClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
        