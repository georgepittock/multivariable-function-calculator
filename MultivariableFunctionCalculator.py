from itertools import product, chain
from typing import Generator

from sympy import Symbol, integrate, Derivative
from pandas import DataFrame


def get_values(n: int) -> Generator:
    if n < 0:
        return
    numbers = chain(*zip(range(n + 1), range(-1, -n - 1, -1)), [n])
    yield from product(numbers, repeat=2)


class FunctionCalculator:
    def __init__(
        self, maximum: tuple, minimum: tuple, saddle: tuple, print_steps: bool = False
    ):
        self.max_x, self.max_y = maximum
        self.min_x, self.min_y = minimum
        self.saddle_x, self.saddle_y = saddle
        self.print_steps = print_steps

        self.x = Symbol("x")
        self.y = Symbol("y")

        self.A = Symbol("A")
        self.B = Symbol("B")

        self.f_x, self.f_y = self.get_first_order_partial_derivatives()
        self.general_form = self.get_general_form()
        self.f_xx, self.d_xy = self.get_second_partial_derivatives()

    def get_first_order_partial_derivatives(self):
        """
        Defines the general form of the function

        :return: array of the first order derivatives with respect to X and to Y
        """
        # automatically try this where B is a constant
        f_x = self.A * self.x * self.x - self.A * self.x * self.max_x
        f_y = self.B * self.y * self.y - self.B * self.y * self.min_y
        return [f_x, f_y]

    def get_general_form(self):
        """
        Get full function in terms of x and y

        :return:  general form of Ax^3 + Bx^2 + Cy^3 + Dy^2
        """
        func_x = integrate(self.f_x, self.x)
        func_y = integrate(self.f_y, self.y)
        return func_y + func_x  # this is the generic form of the final function

    def get_second_partial_derivatives(self):
        """
        Find the equation for the second partial derivatives and also the D function

        :return: array of second partial derivative with respect to x, and D function
        """
        f_xx = self.f_x.diff(self.x)  # finding the second derivative w.r.t x
        f_yy = self.f_y.diff(self.y)  # and again w.r.t y

        f_xy = self.f_x.diff(self.y)
        d_xy = f_xx * f_yy - f_xy ** 2
        return [f_xx, d_xy]

    def evaluate_derivative_at_stationary_points(self):
        """
        Evaluates the second derivative with respect to x, and the D function at the maxima, minima and saddle points
        If steps are to be printed it will print a Pandas DataFrame of the expected information

        :return: array of all the values at these points
        """
        f_xx_at_max = self.f_xx.evalf(subs={self.x: self.max_x, self.y: self.max_y})
        f_xx_at_min = self.f_xx.evalf(subs={self.x: self.min_x, self.y: self.min_y})
        f_xx_at_saddle = self.f_xx.evalf(
            subs={self.x: self.saddle_x, self.y: self.saddle_y}
        )

        d_at_max = self.d_xy.evalf(subs={self.x: self.max_x, self.y: self.max_y})
        d_at_min = self.d_xy.evalf(subs={self.x: self.min_x, self.y: self.min_y})
        d_at_saddle = self.d_xy.evalf(
            subs={self.x: self.saddle_x, self.y: self.saddle_y}
        )

        if self.print_steps:
            data = [
                [
                    (self.max_x, self.max_y),
                    "Maximum",
                    f"{f_xx_at_max} < 0",
                    f"{d_at_max} > 0",
                ],
                [
                    (self.min_x, self.min_y),
                    "Minimum",
                    f"{f_xx_at_min} > 0",
                    f"{d_at_min} > 0",
                ],
                [
                    (self.saddle_x, self.saddle_y),
                    "Saddle",
                    f"{f_xx_at_saddle} != 0",
                    f"{d_at_saddle} < 0",
                ],
            ]
            headers = ["Point", "Type", "f_xx", "f_xx*f_yy-f_xy^2"]
            print(DataFrame(data=data, columns=headers))

        return [
            f_xx_at_max,
            f_xx_at_min,
            f_xx_at_saddle,
            d_at_max,
            d_at_min,
            d_at_saddle,
        ]

    def solve(self, n=1000):
        """
        :param n: the maximum number to test to, i.e. A and B will both be less than this number
        :return: the final function with the only variables x and y
        """
        (
            f_xx_at_max,
            f_xx_at_min,
            f_xx_at_saddle,
            d_at_max,
            d_at_min,
            d_at_saddle,
        ) = self.evaluate_derivative_at_stationary_points()

        for item in get_values(n=n):
            a, b = item
            subs_at_max = {self.x: self.max_x, self.y: self.max_y, self.A: a, self.B: b}
            subs_at_min = {self.x: self.min_x, self.y: self.min_y, self.A: a, self.B: b}
            subs_at_saddle = {self.x: self.saddle_x, self.y: self.saddle_y, self.A: a, self.B: b}
            evaluated_funcs = [
                f_xx_at_max.evalf(subs=subs_at_max) < 0,
                f_xx_at_min.evalf(subs=subs_at_min) > 0,
                f_xx_at_saddle.evalf(subs=subs_at_saddle) != 0,
                d_at_max.evalf(subs=subs_at_max) > 0,
                d_at_min.evalf(subs=subs_at_min) > 0,
                d_at_saddle.evalf(subs=subs_at_saddle) < 0,
            ]
            if all(evaluated_funcs):
                full_func = self.general_form.evalf(subs={self.A: a, self.B: b})
                return full_func
        else:
            return "Solution could not be found"
