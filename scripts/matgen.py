#!/usr/bin/env python3

import numpy as np
import argparse


def is_symmetric(A):
    """ Checks whether the matrix A is symmetric along the main diagonal """
    return np.array_equal(A, A.T)


def is_positively_definite(A):
    """ Checks if all eigenvalues are positive """
    return np.all(np.linalg.eigvals(A) > 0)


def spectral_radius(A):
    """ Returns the largest absolute eigenvalue """
    return np.max(np.abs(np.linalg.eigvals(A)))


def make_lhs(size, y):
    """ (A|...) """
    result = np.zeros((size, size), dtype=np.double)
    np.fill_diagonal(result, y)
    result = np.roll(result, 1, axis=1)
    np.fill_diagonal(result, -1)
    result = np.roll(result, -2, axis=1)
    np.fill_diagonal(result, -1)
    result = np.roll(result, 1, axis=1)
    result[0, -1] = 0
    result[-1, 0] = 0
    return result


def make_lhs_rand(size):
    result = np.zeros((size, size), dtype=np.double)
    rand = np.random.rand(size)
    np.fill_diagonal(result, rand)
    result = np.roll(result, 1, axis=1)
    np.fill_diagonal(result, -1)
    result = np.roll(result, -2, axis=1)
    np.fill_diagonal(result, -1)
    result = np.roll(result, 1, axis=1)
    result[0, -1] = 0
    result[-1, 0] = 0
    return result


def make_rhs(size, y):
    """ (...|b) """
    result = np.ndarray(size, dtype=np.double)
    result[0] = y-1
    result[1:-1] = y-2
    result[-1] = y-1
    return result


def make_rhs_rand(size):
    return np.random.rand(size)


def make_rhs_zero(size):
    return np.zeros(size)


def main():
    parser = argparse.ArgumentParser(
        prog='matgen.py',
        description='Generates systems of linear equations with a symmetric RHS and LHS.',
        epilog='')

    parser.add_argument('-s', '--side', choices=['lhs', 'rhs'])
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-d', '--dimension', type=int)
    parser.add_argument('-y', '--diag_val', type=float)
    parser.add_argument('-r', '--random', type=bool)
    parser.add_argument('-z', '--zero', type=bool)

    args = parser.parse_args()

    if (args.zero):
        mat = make_rhs_zero(args.dimension)
        np.save(args.output, mat)
        return

    if args.random:
        if args.side == 'lhs':
            mat = make_lhs_rand(args.dimension)
        else:
            mat = make_rhs_rand(args.dimension)
    else:
        if args.side == 'lhs':
            mat = make_lhs(args.dimension, args.diag_val)
        else:
            mat = make_rhs(args.dimension, args.diag_val)

    np.save(args.output, mat)


if __name__ == "__main__":
    main()
