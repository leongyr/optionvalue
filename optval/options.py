import numpy as np
from enum import StrEnum

from ._base import _BaseOption

class OptionName(StrEnum):
	European = "EU"
	American = "AM"
	Bermudan = "BM"


class European(_BaseOption):

	def __init__(self,
				 strike: float,
				 maturity: float,
				 payoffType: str):
		super().__init__(strike, maturity, payoffType)
		self.name = "EU"

	def node_values(self, nodePrice, discountedValues):
		return discountedValues


class American(_BaseOption):

	def __init__(self,
				 strike: float,
				 maturity: float,
				 payoffType: str):
		super().__init__(strike, maturity, payoffType)
		self.name = "AM"

	def node_values(self, nodePrice, discountedValues):
		return np.maximum(self.payoff(nodePrice), discountedValues)


class Bermudan(_BaseOption):

	def __init__(self,
				 strike: float,
				 maturity: float,
				 payoffType: str):
		super().__init__(strike, maturity, payoffType)
		self.name = "BM"

	def node_values(self, nodePrice, discountedValues):
		pass