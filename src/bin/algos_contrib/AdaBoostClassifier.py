#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier

from base import ClassifierMixin, BaseAlgo
from codec import codecs_manager
from util.param_util import convert_params
from util.algo_util import tree_summary

class AdaBoostClassifier(ClassifierMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['n_estimators'],
            floats=['learning_rate'],
        )

        self.estimator = _AdaBoostClassifier(algorithm='SAMME',**out_params)
    
    def summary(self, options):
        if 'args' in options:
            raise RuntimeError('Summarization does not take values other than parameters')
        return tree_summary(self, options)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec(
            'algos_contrib.AdaBoostClassifier', 'AdaBoostClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.ensemble._weight_boosting', 'AdaBoostClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.tree._classes', 'DecisionTreeClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
         
