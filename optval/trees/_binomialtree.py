from numbers import Integral, Real
import math
import inspect

import numpy as np

from ..utils._params_validation import ValidateInterval, ValidateChoices, RealNotInt, validate_params
from ._basetree import _BaseTree


class BinomialTree(_BaseTree):
	"""
	Binomial tree for options pricing.
	It uses the implementation of Cox, Ross and Rubinstein (CRR), 1979.

	Calculates and displays binomial trees for:
	1. European call and put options
	2. American call and put options
	3. Bermudan call and put options

	Parameters
	----------
	spot_price: float or int
		Spot price of the underlying asset.

	maturity: float
		Time to option maturity in years.

	volatility: float or int, default=None
		Implied volatility of the underlying asset.
		Will take precedence over using up_state to calculate up state probability if set.

	up_state: float or int, default=2
		Multiple by which underlying asset is expected to increase annually.
		If value is between 0 and 1:
			up_state will be set as reciprocal of input value.

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

	Methods
	-------
	
	
	Examples
	--------
	>>> from optval.trees import BinomialTree as BT
	>>> option_1 = BT(spot_price=4.0, maturity=3.0, nom_discount_rate="25", discount_freq="D")
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
	"""

	_parameter_constraints: dict = {
		"spot_price": [
			ValidateInterval(Real, 0, None, include="neither")
		],
		"maturity":[
			ValidateInterval(Real, 0, None, include="lower")
		],
		"volatility": [
			ValidateInterval(Real, 0, None, include="lower"),
			None
		],
		"up_state":[
			ValidateInterval(Real, 1, None, include="lower"),
			ValidateInterval(RealNotInt, 0, 1, warn="up_state has value less than 1. Reciprocal value will be taken.", include="neither")
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
				 maturity: float | int,
				 *,
				 volatility: float | int = None,
				 up_state: float | int = 2.,
				 nom_discount_rate: float | int = 0.,
				 discount_freq: str = "C",
				 steps: int = 3):

		self.spot_price = spot_price
		self.maturity = maturity
		self.volatility = volatility
		self.up_state = up_state
		self.nom_discount_rate = nom_discount_rate
		self.discount_freq = discount_freq
		self.steps = steps

		# Validate inputs against constraints
		validate_params(self._parameter_constraints,
						self.get_params(),
						self.__class__.__name__)

		self._dT = self.maturity / self.steps
		if self.volatility is not None:
			self._u = np.exp(self.volatility * np.sqrt(self._dT))
		elif self.up_state < 1:
			#warnings.warn("up_state has value less than 1. Reciprocal value will be taken.")
			self._u = 1 / (self.up_state**self._dT)
		else:
			self._u = self.up_state**self._dT
		self._d = 1 / self._u
		self._step_disc = np.exp(self.nom_discount_rate/100*self._dT) if self.discount_freq == "C" else (1+self.nom_discount_rate/100*self._dT)

		self.risk_neutral_p = (self._step_disc - self._d) / (self._u - self._d)
		self.price_tree = self._get_underlying_prices()

	def _get_underlying_prices(self):
		"""
		Derive the underlying asset price for each time step.

		Returns
		-------
		price_tree: array of shape (steps, steps)
			Underlying asset price tree.
		"""
		time_steps = np.linspace(0, self.maturity, self.steps+1)

		# Calculate difference between number of up and down states at each step interval
		state_diff = np.full((self.steps+1, self.steps+1), time_steps, dtype=float) - time_steps.reshape(-1, 1)*2

		# Determine underlying asset prices at each step interval
		# Only retain elements in upper diagonal which reflect the underlying prices
		price_tree = self.spot_price * self._u**state_diff
		price_tree = np.where(np.triu(price_tree), price_tree, 0)

		return price_tree

	def _pascal_triangle(self, n):

		validate_params({"n": [ValidateInterval(Integral, 0, None, include="lower")]},
						locals(),
						f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

		line = np.ones(n+1)
		for k in range(1, n+1):
			line[k] = line[k-1] * (n-k+1) / (k)
		return line

	def _calculate_probas(self, i):
		val_range = np.arange(i, -1, -1)
		return self.risk_neutral_p**(val_range) * (1-self.risk_neutral_p)**(i - val_range)

	def _get_probability_tree(self):
		proba_tree = np.zeros((self.steps+1, self.steps+1))
		for step in range(self.steps+1):
			proba_tree[:step+1, step] = self._calculate_probas(step)
			proba_tree[:step+1, step] *=  self._pascal_triangle(step)
		return proba_tree

	def summary(self):
		print(f"Summary\n" +
			  f"-" * 32 + f"\n"
			  f"{'Spot Price ($)':<23}: {self.spot_price:>7.2f}\n"
			  f"{'Time to maturity (Yrs)':<23}: {self.maturity:>7.2f}\n"
			  f"{'Volatility':<23}: {str(self.volatility) if self.volatility is None else self.volatility:{'>7' if self.volatility is None else '>7.3f'}}\n"
			  f"{'Discount Rate (%)':<23}: {self.nom_discount_rate:>7.2f}\n"
			  f"{'Discount Frequency':<23}: {self.discount_freq:>7}\n"
			  f"{'Steps':<23}: {self.steps:>7}\n"
			 )

	def _print_tree(self, payoff):
		option_tree = np.zeros((self.steps+1, self.steps+1))
		option_tree[ : , -1] = payoff
		for step in np.arange(self.steps-1, -1, -1):
			option_tree[ :step+1, step] = 1/(self._step_disc) * (self.risk_neutral_p*option_tree[ :step+1, step+1] + (1-self.risk_neutral_p)*option_tree[1:step+2, step+1])
		return option_tree

	def get_european_price(self, strike, sel="both", disp_tree=False, get_tree=False):
		"""
		Calculates the present value of a European call/put option for a given strike.
		
		Parameters
		----------
		strike: float
			Option strike price.

		sel: {"call","put","both"}, default="both"
			Option type to be priced.

		disp_tree: bool, default=False
			When True (False by default), displays the discounted option prices 
			at each time step.

		get_tree: bool, default=False
			When True (False by default), will return the discounted option prices 
			at each time step.

		Returns
		-------
		payoff_tree: array of shape (steps, steps)
			Discounted option prices at each time step.

		parity_payoff_tree: array of shape (steps, steps)
			Discounted put option prices at each time step.

		option_price: float
			Present value of selected option for the given strike.

		parity_price: float
			Present value of put option for the given strike.
		"""
		
		validate_params({"strike": [ValidateInterval(Real, 0, None, include="neither")],
						 "sel": [ValidateChoices(str, choices={"call", "put", "both"})],
						 "disp_tree": ["boolean"],
						 "get_tree": ["boolean"]},
						 locals(),
						 f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

		# Initialize final payoffs for the selected option type
		if sel in {"call", "both"}:
			payoff = np.maximum(self.price_tree[ : , -1] - strike, 0)
			sign = 1
		else:
			payoff = np.maximum(strike - self.price_tree[ : , -1], 0)
			sign = -1

		# Calculate present value of option by discounting against payoffs and probabilities at maturity
		#end_state_proba = self._get_probability_tree()[ : , -1]
		end_state_proba = self._pascal_triangle(self.steps) * self._calculate_probas(self.steps)
		option_price = 1/(self._step_disc**self.steps) * np.dot(payoff, end_state_proba)
		
		# Present value calculated from _print_tree method should tally
		if (disp_tree or get_tree):
			payoff_tree = self._print_tree(payoff)
			if disp_tree: print(f"Call:\n{payoff_tree}") 

		if sel not in {"both"}:
			if not get_tree:
				return option_price
			else:
				return payoff_tree, option_price

		# Calculate present value of call/put option using call-put parity
		# Call-put parity is given by the equation: C + PV(Strike) = P + Spot
		parity_price = option_price - sign*self.spot_price + sign*strike*1/(self._step_disc**self.steps)
		
		if (disp_tree or get_tree):
			parity_payoff = np.maximum(strike - self.price_tree[ : , -1], 0)
			parity_payoff_tree = self._print_tree(parity_payoff)
			if disp_tree: print(f"Put:\n{parity_payoff_tree}")

		if not get_tree:
			return option_price, parity_price
		else:
			return payoff_tree, parity_payoff_tree, option_price, parity_price


	def get_american_price(self, strike, sel="both", disp_tree=False, get_tree=False):
		pass

	def get_bermudan_call(self, strike, sel="both", disp_tree=False, get_tree=False):
		pass

	def plot_tree(self):
		pass
