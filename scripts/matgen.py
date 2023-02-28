#!/usr/bin/env python3

import numpy as np
import argparse


def is_diagonally_dominant(A):
    """ Checks whether all numbers on the diagonal are larger than the sums of their respective rows """
    diag = np.diag(np.abs(A))
    row_sum = np.sum(np.abs(A), axis=1) - diag
    return np.all(diag > row_sum)


def is_symmetric(A):
    """ Checks whether the matrix A is symmetric along the main diagonal """
    return np.array_equal(A, A.T)


def is_positively_definite(A):
    """ Checks if all eigenvalues are positive """
    return np.all(np.linalg.eigvals(A) > 0)


def has_positive_diagonal(A):
    """ Checks whether all numbers on the diagonal are positive """
    return np.all(np.diag(A) > 0)


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


def make_rhs(size, y):
    """ (...|b) """
    result = np.ndarray(size, dtype=np.double)
    result[0] = y-1
    result[1:-1] = y-2
    result[-1] = y-1
    return result


def main():
    parser = argparse.ArgumentParser(
        prog='matgen.py',
        description='Generates systems of linear equations with a symmetric RHS and LHS.',
        epilog='')

    parser.add_argument('-s', '--side', choices=['lhs', 'rhs'])
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-d', '--dimension', type=int)
    parser.add_argument('-y', '--diag_val', type=float)

    args = parser.parse_args()

    if args.side == 'lhs':
        mat = make_lhs(args.dimension, args.diag_val)
    else:
        mat = make_rhs(args.dimension, args.diag_val)

    np.save(args.output, mat)


if __name__ == "__main__":
    main()
