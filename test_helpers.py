import numpy as np
import random as rd
from helpers import *

def test_get_polynomial_function():
    fix = [1, 0, 1]
    exp = lambda x: x**2 + 1
    res = get_polynomial_function(fix)
    testing_field = [rd.gauss(rd.randint(-5, 5), 1) for _ in range(200)]
    for i in range(len(testing_field)):
        assert res(testing_field[i]) == exp(testing_field[i])

def test_get_polynomial_function_null():
    fix = [0]
    res = get_polynomial_function(fix)
    testing_field = [rd.gauss(rd.randint(-5, 5), 1) for _ in range(200)]
    for i in range(len(testing_field)):
        assert res(testing_field[i]) == 0


def test_sample_data():
    fix_num = 100
    fix_gen = get_polynomial_function([0])
    res = np.array(list(sample_data(num=fix_num, gen=fix_gen)))
    assert not any(res[:, 1])