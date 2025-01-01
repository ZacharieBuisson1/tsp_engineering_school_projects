#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 2018

Defines functions:
    - bad_cond_matrix : creation matrice mal conditionnee n x n
    - inv_sol : resolution de systeme lineaire (calcul A^{-1}*b but pedagogique)
    - approx_l2 : moindres carres min ||b-Ax||_2^2 (forme explicite et solve)
    - least_l2_pen: min 1/2||x||_2 st Ax=b (forme explicite et solve)
    - ridge (forme explicite + solve)

@author: castella
"""

from numpy.random import randn
from numpy.linalg import svd, inv, solve
from numpy import diag, eye


def bad_cond_matrix(matSize, condNumb):
    """ A = bad_cond_matrix(matSize, condNumb)

    Returns a matrix of size matSize with condition number condNumb.
    Designed for educational purposes to test badly conditionned matrices.
    """
    m, n = matSize
    # ### une matrice aleatoire dont je triture les valeurs singulieres ###
    A = randn(m, n)
    u, s, v = svd(A, full_matrices=False)
    newsmin = s.max()/condNumb
    news = (s - s.max()) / (s.max() - s.min())*(s.max() - newsmin) + s.max()
    A = u.dot(diag(news)).dot(v)
    # ### check condition number ### #ne pas oublier importer cond#
    # print('cond(A) = {0}'.format(cond(A)))
    return A


from numpy.random import randn
from numpy.linalg import svd, inv, solve
from numpy import diag, eye

def inv_sol(A, b):
    return inv(A).dot(b)
    #raise UserWarning('Reprogrammer cette fonction')


def approx_l2(A, b):
    B=np.dot(np.transpose(A),A)
    Ay=np.transpose(A).dot(b)
    return inv(B).dot(Ay)
    #raise UserWarning('Reprogrammer cette fonction')


def least_l2_pen(A, b):
    return A.transpose().dot(inv(A.dot(A.transpose()))).dot(b)
    #raise UserWarning('Reprogrammer cette fonction')


def ridge(A, b, lamb):
    B=np.dot(np.transpose(A),A)+lamb*np.eye(A.shape[0])
    Ay=np.transpose(A).dot(b)
    return inv(B).dot(Ay)
    #raise UserWarning('Reprogrammer cette fonction')