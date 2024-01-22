# https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/base.py
# class BaseEstimator

from numbers import Integral, Real
import inspect
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np

from .utils._params_validation import ValidateInterval, ValidateChoices, validate_params

class _Base():
	
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


class PayoffType(StrEnum):
	Call = "C"
	Put = "P"
	BinaryCall = "BC"
	BinaryPut = "BP"


class _BaseOption(ABC, _Base):

	_parameter_constraints: dict = {
		"strike": [
			ValidateInterval(Real, 0, None, include="neither")
		],
		"maturity":[
			ValidateInterval(Real, 0, None, include="lower")
		],
		"payoffType": [
			ValidateChoices(str, choices={"C", "P"})
		]
	}

	def __init__(self,
				 strike: float,
				 maturity: float,
				 payoffType: str):

		self.strike = strike
		self.maturity = maturity
		self.payoffType = payoffType

		validate_params(self._parameter_constraints,
						self.get_params(),
						self.__class__.__name__)

	def payoff(self, nodePrice):
		if self.payoffType == "C":
			return np.maximum(nodePrice - self.strike, 0)
		elif self.payoffType == "P":
			return np.maximum(self.strike - nodePrice, 0)

	@abstractmethod
	def node_values(self, nodePrice, discountedValues):
		"""
		Option payoffs at each time step for various option types

		Parameters
		----------
		spotPrice : array of float
			Underlying spot prices at each time step

		discountedValues : array of float
			Discounted option payoffs from the  time step

		Returns
		-------
		node_values : array of float
			Option payoffs for a particular option type
		"""

