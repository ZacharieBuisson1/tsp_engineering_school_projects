#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 23:07:42 2018

Fonctions crees pour le TP de signal 2A LinearModels.
- approx_linf : min ||b-Ax||_infty (solveur LP)
- approx_dzl : min dzl(b-Ax) (solveur LP)
- approx_l1 : min ||b-Ax||_1 (solveur LP)
- least_l1_pen : basis pursuit = min ||x||_1 st Ax=b (solveur LP)

@author: castella
"""

from numpy import concatenate, expand_dims, ones, zeros, eye
from scipy.optimize import linprog


def approx_linf(A, b):
    """ x, minlinfval = approx_linf(A, b)

    Returns the solution of
        min ||b-Ax||_infty
        (wrt x)
    (LP solver from scipy.optimize.linprog)
    """
    m, n = A.shape
    bub = concatenate((b, -b))
    A2 = concatenate((A, -A))
    Aub = concatenate((expand_dims(-ones(2*m), 1), A2), axis=1)
    c = zeros(n+1)
    c[0] = 1
    lpSol = linprog(c, Aub, bub, method='interior-point')
    x = lpSol.x[1:]
    minlinfval = lpSol.fun
    return x, minlinfval


def approx_dzl(A, b, alpha):
    """ x = approx_dzl(A, b)

    Returns the solution of
        min dzl(b-Ax)
        (wrt x)
        where dzl is the dead-zone linear function
        dzl(x) = max(0, abs(x) - alpha)
    (LP solver from scipy.optimize.linprog)
    """
    raise UserWarning('Reprogrammer cette fonction')


def approx_l1(A, b):
    """ x = approx_l1(A, b)

    Returns the solution of
        min ||b-Ax||_1
        (wrt x)
    (LP solver from scipy.optimize.linprog)
    """
    raise UserWarning('Reprogrammer cette fonction')


def least_l1_pen(A, b):
    """ least_l1_pen(A, b)  (a.k.a. BasisPursuit)

    Returns the solution of
        min. ||x||_1 s.t. Ax=b
        (wrt x)
    (LP solver from scipy.optimize.linprog)
    """
    raise UserWarning('Reprogrammer cette fonction')
