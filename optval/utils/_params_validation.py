from numbers import Integral, Real
from abc import ABC, abstractmethod
import operator

import numpy as np


class InvalidParameterError(ValueError, TypeError):
	"""
	Custom exception to be raised when the parameter of a class/method/function
    does not have a valid value or type.
    """

    # Inherits from ValueError and TypeError for backward compatibility.


class RealNotInt(Real):
	"""
	A type that represents real numbers that are not instances of int.

    Behaves like float, but also works with values extracted from numpy arrays.
    isinstance(1, RealNotInt) -> False
    isinstance(1.0, RealNotInt) -> True
    """

RealNotInt.register(float)


def _type_name(typing):
	module = typing.__module__
	qual_name = typing.__qualname__
	if module == "builtins":
		return qual_name
	elif typing == Real:
		return "float/int"
	elif typing == Integral:
		return "int"
	return f"{module}.{qual_name}"


class _Validation(ABC):
	"""
	Base class for Validation objects.
	"""
	def __init__(self):
		self.hidden = False

	@abstractmethod
	def is_satisfied_by(self, val):
		"""
		Whether a value satisfies the parameter constraints.

		Parameters
		----------
		val : object
			The value to check.

		Returns
		-------
		is_satisfied : bool
			Whether the constraint satisfy the expected type and range of values.
		"""
	@abstractmethod
	def __str__(self):
		"""
		A human readable representational string of the constraint.
		"""


class _ValidateNone(_Validation):
	
	def is_satisfied_by(self, val):
		return val is None

	def __str__(self):
		return "None"


class _ValidateBoolean(_Validation):

	def is_satisfied_by(self, val):
		return isinstance(val, bool) or isinstance(val, np.bool_)

	def __str__(self):
		return "boolean"


class _InstanceOf(_Validation):

	def __init__(self, instance):
		super().__init__()
		self.instance = instance

	def is_satisfied_by(self, val):
		return isinstance(val, self.instance)

	def __str__(self):
		return f"an instance of {_type_name(self.instance)}"


class ValidateInterval(_Validation):

	def __init__(self,
				 typing,
				 lower,
				 upper,
				 *,
				 include,
				 ):

		super().__init__()
		self.typing = typing
		self.lower = lower
		self.upper = upper
		self.include = include

		self._check_constraints()

	def _check_constraints(self):
		if self.typing not in (Integral, Real, RealNotInt):
			raise ValueError("type must be either numbers.Integral, numbers.Real or RealNotInt."
				             f"Got {self.typing} instead.")

		if self.include not in ("lower", "upper", "both", "neither"):
			raise ValueError("include must be either 'lower', 'upper', 'both' or 'neither'."
				             f"Got `{self.include}` instead.")

		suffix = "for the given interval."
		# Check for correct input typing
		if self.typing is Integral:
			if self.lower is not None and not isinstance(self.lower, Integral):
				raise TypeError("Expecting lower bound to be an int {suffix}.")
			if self.upper is not None and not isinstance(self.upper, Integral):
				raise TypeError("Expecting upper bound to be an int {suffix}.")
		else:
			if self.lower is not None and not isinstance(self.lower, Real):
				raise TypeError("Expecting lower bound to be a real number.")
			if self.upper is not None and not isinstance(self.upper, Real):
				raise TypeError("Expecting upper bound to be a real number.")

		# Check for correct boundary constraint conditions
		if self.lower is None and self.include in ("lower", "both"):
			raise ValueError(f"Lower bound cannot be None when include is {self.include} {suffix}.")
		if self.upper is None and self.include in ("upper", "both"):
			raise ValueError(f"Upper bound cannot be None when include is {self.include} {suffix}.")

		# Check for correct constraints interval
		if self.lower is not None and self.upper is not None and self.upper <= self.lower:
			raise ValueError("Upper bound cannot be less than the lower bound."
				        	 f"Got lower={self.lower} and upper={self.upper}.")


	def is_satisfied_by(self, val):
		return (isinstance(val, self.typing)) and (val in self)

	def __contains__(self, val):
		if np.isnan(val):
			return False

		left = -np.inf if self.lower is None else self.lower
		right = np.inf if self.upper is None else self.upper

		left_cmp = operator.lt if self.include in ("lower", "both") else operator.le
		right_cmp = operator.gt if self.include in ("upper", "both") else operator.ge

		if left_cmp(val, left):
			return False 
		if right_cmp(val, right):
			return False 
		return True

	def __str__(self):
		type_str = "an int" if self.typing is Integral else "a float/int"
		left_bracket = "[" if self.include in ("lower", "both") else "("
		left_bound = "-inf" if self.lower is None else self.lower
		right_bound = "inf" if self.upper is None else self.upper
		right_bracket = "]" if self.include in ("upper", "both") else ")"

		if not self.typing == Integral and isinstance(self.lower, Real):
			left_bound = float(left_bound)
		if not self.typing == Integral and isinstance(self.upper, Real):
			right_bound = float(right_bound)

		return f"{type_str} in the range {left_bracket}{left_bound}, {right_bound}{right_bracket}"


class ValidateChoices(_Validation):

	def __init__(self, typing, *, choices):
		super().__init__()
		self.typing = typing
		self.choices = choices

	def is_satisfied_by(self, val):
		return isinstance(val, self.typing) and val in self.choices

	def __str__(self):
		return f"a {_type_name(self.typing)} among the choices {list(self.choices)}"


def _define_validation(criteria):
	if isinstance(criteria, (ValidateInterval, ValidateChoices)):
		return criteria
	if criteria is None:
		return _ValidateNone()
	if isinstance(criteria, str) and criteria == "boolean":
		return _ValidateBoolean()
	if isinstance(criteria, type):
		return _InstanceOf(criteria)
	raise ValueError(f"Unknown validation criteria: {criteria}")


def validate_params(parameter_constraints, params, caller_name):
	for param_name, param_val in params.items():
		if param_name not in parameter_constraints:
			continue
		constraints = parameter_constraints[param_name]
		validations = [_define_validation(constraint) for constraint in constraints]
		for validate in validations:
			if validate.is_satisfied_by(param_val):
				break
		else:
			if len(validations) == 1:
				validate_str = f"{constraints[0]}"
			else:
				validate_str = (f"{', '.join([str(c) for c in constraints[:-1]])} or"
							    f" {constraints[-1]}.")
			raise InvalidParameterError(f"The {param_name} parameter of {caller_name} must be"
										f" {validate_str}. Got the value `{param_val}` instead.")













