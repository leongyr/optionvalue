from numbers import Integral, Real
import math
import inspect

import numpy as np

from .._base import _Base, _BaseOption
from ..utils._params_validation import (
	ValidateInterval,
	ValidateChoices,
	validate_params
)


class BinomialPricer(_Base):
	"""
	Binomial tree for options pricing.

	Currently support options include:
	1. European call and put options
	2. American call and put options
	3. Bermudan call and put options

	Parameters
	----------
	spot_price: float or int
		Spot price of the underlying asset.

	option: BaseOption
		Instance of a BaseOption class.

	volatility: float, int or None, default=None
		Implied volatility of the underlying asset.
		Will take precedence over using up_state to calculate up state probability if set.

	state: float, int or {"crr", "tian", "jr", "jrrn", "symjrrn", "adjtree"}, default=2
		Multiple by which underlying asset is expected to increase annually.
		If value is between 0 and 1:
			up_state will be set as reciprocal of input value.
		If crr:
			Cox-Ross-Rubinstein tree implementation.
		If tian:
			Tian tree implementation.
			Uses an extra D.O.F to match the first three moments.
		If jr:
			Non risk-neutral Jarrow-Rudd tree.
			Probability of up state is fixed at 0.5.
		If jrrn:
			Jarrow-Rudd risk-neutral (JRRN) tree.
		If symjrrn:
			Symmetrized version of JRRN tree.
		If adjtree:
			text
		If flextree:
			text
		If cptree:
			text
		If lr:
			text
		If j4:
			text
		If splittree:
			text

	nom_discount_rate: float or int, default=0
		Nominal annual rate in percentage for discounting future values.

	discount_freq: {"D","C"}, default='C'
		If D:
			Discrete compounding of discount rate at each step.
		If C:
			Continuous compounding of discount rate.

	steps: int, default=3
		Number of discrete intervals till maturity.

	Attributes
	----------
	price_tree: ndarray of shape (steps, steps)
		Tree representation of underlying asset price tree for each node at
		every time interval.

	p: float
		Value of the risk-neutral probability (does not apply if state is 'jr').
	
	Examples
	--------
	>>> from optval.trees import BinomialTree as BT
	>>> option_1 = BT(spot_price=4.0, maturity=3.0, nom_discount_rate=25, discount_freq="D")
	>>> print(option_1.price_tree)
	[[ 4. ,  8. , 16. , 32. ],
	 [ 0. ,  2. ,  4. ,  8. ],
	 [ 0. ,  0. ,  1. ,  2. ],
	 [ 0. ,  0. ,  0. ,  0.5]]
	>>>
	>>> option_1.summary()
	Summary
	--------------------------------
	Spot Price ($)         :    4.00
	Time to maturity (Yrs) :    3.00
	Volatility             :    None
	Discount Rate (%)      :   25.00
	Discount Frequency     :       D
	Steps                  :       3
	>>>
	>>> option_1.get_european_price(strike=10.0, sel="call", disp_tree=True, get_tree=False)
	Call:
	[[ 1.408  3.52   8.8   22.   ]
	 [ 0.     0.     0.     0.   ]
	 [ 0.     0.     0.     0.   ]
	 [ 0.     0.     0.     0.   ]]
	1.408
	>>>
	>>> option_1.get_american_price(strike=10.0, sel="put", disp_tree=True, get_tree=False)
	Put:
	[[6.   2.72 0.8  0.  ]
	 [0.   8.   6.   2.  ]
	 [0.   0.   9.   8.  ]
	 [0.   0.   0.   9.5 ]]
	6.0
	"""

	_parameter_constraints: dict = {
		"spot_price": [
			ValidateInterval(Real, 0, None, include="neither")
		],
		"option":[
			_BaseOption
		],
		"volatility": [
			ValidateInterval(Real, 0, None, include="lower"),
			None
		],
		"state":[
			ValidateInterval(Real, 0, None, include="neither"),
			ValidateChoices(str, choices={"crr", "tian", "jr", "jrrn", "symjrrn", "adjtree",
										  "flextree", "cptree", "lr", "j4", "splittree"})
		],
		"nom_discount_rate":[
			ValidateInterval(Real, 0, None, include="lower"),
		],
		"discount_freq":[
			ValidateChoices(str, choices={"C", "D"})
		],
		"steps":[
			ValidateInterval(Integral, 0, None, include="lower")
		]
	}


	def __init__(self,
				 spot_price: float | int,
				 option: _BaseOption,
				 *,
				 volatility: float | int | None = None,
				 state: float | int | str = 2.,
				 nom_discount_rate: float | int = 0.,
				 discount_freq: str = "C",
				 steps: int = 3):

		self.spot_price = spot_price
		self.option = option
		self.volatility = volatility
		self.state = state
		self.nom_discount_rate = nom_discount_rate
		self.discount_freq = discount_freq
		self.steps = steps

		_calibration: dict = {
			"crr": self._crrcalib,
			"tian": self._tiancalib,
			"jr": self._jrcalib,
			"jrrn": self._jrrncalib,
			"symjrrn": self._symjrrncalib,
			"adjtree": self._adjtreecalib,
			"flextree": self._flextreecalib,
			"cptree": self._cptreecalib,
			"lr": self._lrcalib,
			"j4": self._j4calib,
			"splittree": self._splittreecalib,
		}

		# Validate inputs against constraints
		validate_params(self._parameter_constraints,
						self.get_params(),
						self.__class__.__name__)

		self._dT = self.option.maturity / self.steps
		self._step_disc = np.exp(self.nom_discount_rate/100*self._dT) if self.discount_freq == "C" else (1+self.nom_discount_rate/100*self._dT)


		if self.volatility is not None:
			# Check if condition dT < volatility^2 / (step_disc - div_yield)^2
			# is fulfilled for p to be in the interval (0, 1)
			# Current implementation assumes no dividend, i.e div_yield = 0
			if self._dT > (self.volatility / self._step_disc)**2:
				raise ValueError("Constraints not satisfied for risk-neutral p to fall within interval (0, 1).")
			if not isinstance(self.state, str):
				raise TypeError(f"Invalid calibration model selection. Got {self.state}.")
			self._u, self._d = _calibration[state]()
		elif self.state < 1:
			warnings.warn("state has value less than 1. Reciprocal value will be taken.")
			self._d = self.state
			self._u = 1 / self._d
		else:
			self._u = self.state
			self._d = 1 / self._u

		self.p = 0.5 if state == "jr" else (self._step_disc - self._d) / (self._u - self._d)
		self.price_tree = self._get_underlying_prices()


	def _crrcalib(self):
		"""
		Cox, Ross and Rubinstein (CRR) implementation, 1979.
		"""
		u = np.exp(self.volatility * np.sqrt(self._dT))
		d = 1 / u
		return u, d


	def _tiancalib(self):
		r = np.exp(self._step_disc * self._dT)
		v = np.exp(self.volatility**2 * self._dT)
		u = 0.5 * r * v * (v + 1 + np.sqrt(v**2 + 2*v - 3))
		d = 0.5 * r * v * (v + 1 - np.sqrt(v**2 + 2*v -3))
		return u, d


	def _jrcalib(self):
		mu = self._step_disc - 0.5 * self.volatility**2
		u = np.exp(mu*self._dT + self.volatility*np.sqrt(self._dT))
		d = np.exp(mu*self._dT - self.volatility*np.sqrt(self._dT))
		return u, d


	def _jrrncalib(self):
		u, d = self._jrcalib()
		return u, d


	def _symjrrncalib(self):
		u, d = self._jrcalib()
		X = 2 * np.exp(self._step_disc * self._dT) / (u + d)
		return X*u, X*d


	def _adjtreecalib(self):
		mu = 1/self.option.maturity * (np.log(self.option.strike) - np.log(self.spot_price))
		u = np.exp(mu*self._dT + self.volatility*np.sqrt(self._dT))
		d = np.exp(mu*self._dT - self.volatility*np.sqrt(self._dT))
		return u, d


	def _flextreecalib(self):
		j = np.ceil(
				(np.log(self.option.strike/self.spot_price) + self.steps*self.volatility*np.sqrt(self._dT)) 
				/ 
				(2*self.volatility*np.sqrt(self._dT))
			)
		lumda = (
			(np.log(self.option.strike/self.spot_price) - (2*j - self.steps)*self.volatility*np.sqrt(self._dT)) 
			/ 
			(self.steps * self.volatility**2 * self._dT)
		)
		u = np.exp(self.volatility*np.sqrt(self._dT) + lumda * self.volatility**2 * self._dT)
		d = np.exp(-self.volatility*np.sqrt(self._dT) + lumda * self.volatility**2 * self._dT)
		return u, d


	def _cptreecalib(self):
		j = np.ceil(
				(np.log(self.option.strike/self.spot_price) + self.steps*self.volatility*np.sqrt(self._dT)) 
				/ 
				(2*self.volatility*np.sqrt(self._dT))
			)
		lumda = (
			(np.log(self.option.strike/self.spot_price) - (2*j - 1 - self.steps)*self.volatility*np.sqrt(self._dT)) 
			/ 
			(self.steps * self.volatility**2 * self._dT)
		)
		u = np.exp(self.volatility*np.sqrt(self._dT) + lumda * self.volatility**2 * self._dT)
		d = np.exp(-self.volatility*np.sqrt(self._dT) + lumda * self.volatility**2 * self._dT)
		return u, d


	def _lrcalib(self):
		pass


	def _j4calib(self):
		pass


	def _splittreecalib(self):
		pass


	def _get_underlying_prices(self):
		"""
		Derive the underlying asset price for each time step.

		Returns
		-------
		price_tree: array of shape (steps, steps)
			Underlying asset price tree.
		"""
		#time_steps = np.linspace(0, self.option.maturity, self.steps+1)
		time_steps = np.arange(0, self.steps+1)

		# Calculate difference between number of up and down states at each step interval
		state_diff = np.full((self.steps+1, self.steps+1), time_steps, dtype=float) - time_steps.reshape(-1, 1)*2

		# Determine underlying asset prices at each step interval
		# Only retain elements in upper diagonal which reflect the underlying prices
		price_tree = self.spot_price * self._u**state_diff
		price_tree = np.where(np.triu(price_tree), price_tree, 0)

		return price_tree


	def _pascal_triangle(self, n):
		line = np.ones(n+1)
		for k in range(1, n+1):
			line[k] = line[k-1] * (n-k+1) / (k)
		return line


	def _calculate_probas(self, i):
		val_range = np.arange(i, -1, -1)
		return self.p**(val_range) * (1-self.p)**(i - val_range)


	def _get_probability_tree(self):
		# Currently not in use
		proba_tree = np.zeros((self.steps+1, self.steps+1))
		for step in range(self.steps+1):
			proba_tree[:step+1, step] = self._calculate_probas(step)
			proba_tree[:step+1, step] *=  self._pascal_triangle(step)
		return proba_tree


	def summary(self):
		"""
		Prints a summary of the input parameters.
		"""
		print(f"Summary\n" +
			  f"-" * 32 + f"\n"
			  f"{'Spot Price ($)':<23}: {self.spot_price:>7.2f}\n"
			  f"{'Strike Price ($)':<23}: {self.option.strike:>7.2f}\n"
			  f"{'Payoff Type':<23}: {self.option.payoffType:>7}\n"
			  f"{'Time to maturity (Yrs)':<23}: {self.option.maturity:>7.2f}\n"
			  f"{'Volatility':<23}: {str(self.volatility) if self.volatility is None else self.volatility:{'>7' if self.volatility is None else '>7.3f'}}\n"
			  f"{'Discount Rate (%)':<23}: {self.nom_discount_rate:>7.2f}\n"
			  f"{'Discount Frequency':<23}: {self.discount_freq:>7}\n"
			  f"{'Steps':<23}: {self.steps:>7}\n"
			 )


	def price_option(self, tree:bool=False):
		"""
		Calculates the value for a given option.
		
		Parameters
		----------
		tree: bool, default=False
			When True (False by default), returns the option prices at
			each step in an array.

		Returns
		-------
		price: float
			Value of the option.

		option_tree: array of shape (steps, steps)
			Array of option prices at each step.
		"""

		option_tree = np.zeros((self.steps+1, self.steps+1))
		option_tree[ : , -1] = self.option.payoff(self.price_tree[ : , -1])
		for step in np.arange(self.steps-1, -1, -1):
			discounted_values = (1/(self._step_disc) * (self.p*option_tree[ :step+1, step+1] + 
													   (1-self.p)*option_tree[1:step+2, step+1])
								)
			option_tree[ :step+1, step] = self.option.node_values(self.price_tree[ :step+1, step], discounted_values)
		if tree:
			return option_tree[0, 0], option_tree
		return option_tree[0, 0]


	def plot_tree(self):
		pass

