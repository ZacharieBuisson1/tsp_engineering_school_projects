#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 23:07:42 2018

Divers algo proximaux. Utilisés pour le TP Signal 2A TSP LinearModels
(estimation dans les modèles linéaires).
    Lasso -> algorithme calcul Lasso par Forward-Backward (gradient proximal)
    least_l1_pen -> BasisPursuit min ||x||_1 s.t. Ax=b
    approx_l1 -> min ||b-Ax||_1
    approx_dzl -> min dzl(b-Ax)
    prox_dzl -> le prox de la fonction dzl
    soft_thresh -> soft-thresholding (prox de | |)

@author: castella
"""

from numpy import sign, array, maximum, zeros, sqrt, eye, where
from numpy.linalg import solve, norm


def huber(x, w):
    """ huber(x, w)

    Huber function defined by:
    1/2*x^2    if |x| <= w
    w|x|-w^2/2 otherwise.
    """
    return where(abs(x) < w, 1/2*x**2, w*abs(x)-w**2/2)


def prox_huber(v, w, lamb):
    """ prox_huber(v, w, lamb)

    Returns the prox of lamb*huber.
    """
    return where(abs(v) < (1+lamb)*w, v/(1+lamb), v-lamb*w*sign(v))


def prox_dzl(v, alpha, lamb):
    """ x = prox_dzl(v, alpha, lamb)

    Returns the prox of lamb*dzl(u) where dzl is the dead-zone linear penalty
    dzl(u) = max(O, abs(u)-alpha)
    """
    x = [i*(abs(i) < alpha)
         + alpha*sign(i)*((abs(i) >= alpha) & (abs(i) <= alpha+lamb))
         + (i - lamb*sign(i)) * (abs(i) > alpha + lamb) for i in v]
    return array(x)


def soft_thresh(u, s):
    """ v = soft_thresh(u, s)

    Returns a soft-thresholding (elementwise) of u with threshold s
    v = sign(v)*maximum(0, abs(u) - s)
    """
    v = sign(u)*maximum(0, abs(u) - s)
    return v


def approx_huber(A, b, w):
    """
    x = approx_huber(A, b, w)

    Returns the solution of
        min huber(b-Ax, w)
        (wrt x)
        where huber(u, w) = 1/2*x^2 if |x| <= w, w|x|-w^2/2 otherwise.
    (ADMM algorithm)
    """
    m, n = A.shape
    AA = A.transpose().dot(A)

    rho = 10
    lambd = 1/rho

    history = {'crit':[], 'r_norm':[], 's_norm':[], 'eps_prim':[], 'eps_dual':[]}
    MAXITER = 1000
    ABSTOL, RELTOL = 1e-4, 1e-2
    x, z, u = zeros(n), zeros(m), zeros(m)
    for iter in range(MAXITER):
        x = solve(AA, A.transpose().dot(b + z - u))
        Ax = A.dot(x)
        zold = z
        z = prox_huber(Ax - b + u, w, lambd)
        u = u + Ax - z - b

        s_norm = norm(-1/lambd*A.transpose().dot(z - zold))
        r_norm = norm(Ax - z - b)

        eps_prim = sqrt(m)*ABSTOL + RELTOL*max([norm(Ax), norm(z), norm(b)])
        eps_dual = sqrt(n)*ABSTOL + RELTOL*(1/lambd)*norm(A.transpose().dot(u))

        history['s_norm'].append(s_norm)
        history['r_norm'].append(r_norm)
        history['crit'].append(norm(x, 1))
        history['eps_prim'].append(eps_prim)
        history['eps_dual'].append(eps_dual)

        if (s_norm < eps_prim) & (r_norm < eps_dual):
            break

    return x


def approx_dzl(A, b, alpha):
    """
    x = approx_dzl(A, b)

    Returns the solution of
        min dzl(b-Ax)
        (wrt x)
        where dzl(u) = max(0,abs(u)-alpha)
    (ADMM algorithm)
    """
    m, n = A.shape
    AA = A.transpose().dot(A)

    rho = 10
    lambd = 1/rho

    history = {'crit':[], 'r_norm':[], 's_norm':[], 'eps_prim':[], 'eps_dual':[]}
    MAXITER = 1000
    ABSTOL, RELTOL = 1e-4, 1e-2
    x, z, u = zeros(n), zeros(m), zeros(m)
    for iter in range(MAXITER):
        x = solve(AA, A.transpose().dot(b + z - u))
        Ax = A.dot(x)
        zold = z
        z = prox_dzl(Ax - b + u, alpha, lambd)
        u = u + Ax - z - b

        s_norm = norm(-1/lambd*A.transpose().dot(z - zold))
        r_norm = norm(Ax - z - b)

        eps_prim = sqrt(m)*ABSTOL + RELTOL*max([norm(Ax), norm(z), norm(b)])
        eps_dual = sqrt(n)*ABSTOL + RELTOL*(1/lambd)*norm(A.transpose().dot(u))

        history['s_norm'].append(s_norm)
        history['r_norm'].append(r_norm)
        history['crit'].append(norm(x, 1))
        history['eps_prim'].append(eps_prim)
        history['eps_dual'].append(eps_dual)

        if (s_norm < eps_prim) & (r_norm < eps_dual):
            break

    return x


def approx_l1(A, b):
    """ x = approx_l1(A, b)

    Returns the solution of
        min ||b-Ax||_1
        (wrt x)
    (ADMM algorithm)
    """
    m, n = A.shape
    AA = A.transpose().dot(A)

    rho = 10
    lambd = 1/rho

    history = {'crit':[], 'r_norm':[], 's_norm':[], 'eps_prim':[], 'eps_dual':[]}
    MAXITER = 1000
    ABSTOL, RELTOL = 1e-4, 1e-2
    x, z, u = zeros(n), zeros(m), zeros(m)
    for iter in range(MAXITER):
        x = solve(AA, A.transpose().dot(b + z - u))
        Ax = A.dot(x)
        zold = z
        z = soft_thresh(Ax - b + u, lambd)
        u = u + Ax - z - b

        s_norm = norm(-1/lambd*A.transpose().dot(z - zold))
        r_norm = norm(Ax - z - b)

        eps_prim = sqrt(m)*ABSTOL + RELTOL*max([norm(Ax), norm(z), norm(b)])
        eps_dual = sqrt(n)*ABSTOL + RELTOL*(1/lambd)*norm(A.transpose().dot(u))

        history['s_norm'].append(s_norm)
        history['r_norm'].append(r_norm)
        history['crit'].append(norm(x, 1))
        history['eps_prim'].append(eps_prim)
        history['eps_dual'].append(eps_dual)

        if (s_norm < eps_prim) & (r_norm < eps_dual):
            break

    return x


def least_l1_pen(A, b):
    """ least_l1_pen(A, b)  (a.k.a. BasisPursuit)

    Returns the solution of
        min. ||x||_1 s.t. Ax=b
        (wrt x)
    (ADMM algorithm)
    """
    n = A.shape[1]

    AA = A.dot(A.transpose())
    AAlA = solve(AA, A)
    Pproj = eye(n) - A.transpose().dot(AAlA)
    AAlb = solve(AA, b)
    bproj = A.transpose().dot(AAlb)

    lambd = 0.1

    history = {'crit': [], 'r_norm': [], 's_norm': [], 'eps_prim': [], 'eps_dual': []}
    MAXITER = 1000
    ABSTOL, RELTOL = 1e-4, 1e-2
    x, z, u = zeros(n), zeros(n), zeros(n)
    for iter in range(MAXITER):
        x = Pproj.dot(z-u) + bproj
        zold = z
        z = soft_thresh(x+u, lambd)
        u = u + x - z

        s_norm = norm(-1/lambd*(z-zold))
        r_norm = norm(x - z)

        eps_prim = sqrt(n)*ABSTOL + RELTOL*max([norm(x), norm(z)])
        eps_dual = sqrt(n)*ABSTOL + RELTOL*(1/lambd)*norm(u)

        history['s_norm'].append(s_norm)
        history['r_norm'].append(r_norm)
        history['crit'].append(norm(x, 1))
        history['eps_prim'].append(eps_prim)
        history['eps_dual'].append(eps_dual)

        if (s_norm < eps_prim) & (r_norm < eps_dual):
            break
    return x


def lasso(A, b, lamb, maxiter=5000):
    """ lasso(A, b, lamb, maxiter=500)

    min. 1/2*norm(b- Ax)^2+lambda*sum(abs(x))
    (wrt x)
    Forward-Backward algorithm
    """
    n = A.shape[1]
    gam = 1.9/norm(A, 2)**2

    x = zeros(n)
#    history = {'crit':[]}
    prec = 1e-6
    crit = 0
    for nit in range(maxiter):
        Ax = A.dot(x)
        critold = crit
        crit = norm(Ax-b)**2/2+lamb*sum(abs(x))
#        history['crit'].append(crit)
        if nit > 1 and critold-crit < prec*critold:
            break
        x = x-gam*A.transpose().dot(Ax-b)
        x = soft_thresh(x, lamb*gam)
#        print('{0} crit = {1}\n'.format(nit, crit))
    # print('   Exiting at iteration {0} (maxiter = {1}).\n'
    # .format(nit, maxiter))
    return x
