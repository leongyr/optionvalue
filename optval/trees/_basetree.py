# https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/base.py
# class BaseEstimator

import inspect

class _BaseTree():
	
	@classmethod
	def _get_param_names(cls):
		class_init = cls.__init__
		class_init_signature = inspect.signature(class_init)
		param_names = [
			param.name
			for param in class_init_signature.parameters.values()
			if param.name != "self"
			]
		return param_names

	def get_params(self):
		result = dict()
		for param in self._get_param_names():
			value = getattr(self, param)
			result[param] = value
		return result