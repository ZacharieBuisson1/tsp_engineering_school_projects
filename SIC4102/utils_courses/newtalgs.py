#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2020/01/24

Defines functions:
    - approx_lb

@author: castella
"""

from numpy import log, sqrt, diag
from numpy.linalg import solve
from .lpalgs import approx_linf


def approx_lb(A, b, xlbini):
    """ approx_lb(A, b, xlbini)

    Returns the solution of:
        min. logb(b-Ax)
        (wrt x)
    where logb is the log-barrier function.
    Newton algorithm
    Requires feasible initialization point xlbini such that
    ||b-A.dot(xlbini)||_infty < 1
    Can be obtained as follows:

    xlbini, linf = tp2a.lpalgs.approx_linf(A, b)
    A = A/(1.1*linf)
    b = b/(1.1*linf)
    """

    xlb = xlbini

    # Newton algorithm
    alpha, beta = 0.01, 0.5  # param for Newton method and line search
    for niter in range(100):
        yp, ym = 1 - (A.dot(xlb) - b), 1 + (A.dot(xlb) - b)
        f = - log(yp).sum() - log(ym).sum()
        g = A.transpose().dot(1/yp) - A.transpose().dot(1/ym)
        H = A.transpose().dot(diag(1/yp**2 + 1/ym**2)).dot(A)
        v = - solve(H, g)
        fprime = g.transpose().dot(v)
        ntdecr = sqrt(-fprime)
        if ntdecr < 1e-5:
            break
        # backtracking line search
        t = 1
        newx = xlb + t*v
        while ((1 - (A.dot(newx) - b)).min() < 0 or
               (1 + (A.dot(newx) - b)).min() < 0):
            t = beta*t
            newx = xlb + t*v
        newf = - log(1 - (A.dot(newx) - b)).sum() \
               - log(1 + (A.dot(newx) - b)).sum()
        while newf > f + alpha*t*fprime:
            t = beta*t
            newx = xlb + t*v
        xlb = xlb + t*v
    return xlb
