import numpy as np


def is_symmetric(A):
    return np.array_equal(A, A.T)


def is_positively_definite(A):
    return np.all(np.linalg.eigvals(A) > 0)


def cgm(A: np.ndarray, b: np.ndarray, max_it=10_000, error=1e-6):
    if not is_positively_definite(A):
        print("Not positively definite")

    if not is_symmetric(A):
        print("Not symmetric")
        return None, None

    x = np.zeros_like(b)

    r = b - A @ x
    p = r
    rTr = r.T @ r

    k = 0

    while k < max_it and np.linalg.norm(r, 2) > error:
        Ap = A @ p

        a = rTr / (p.T @ Ap)

        x = x + a * p
        r = r - a * Ap

        rTr2 = r.T @ r
        scale = (rTr2) / rTr
        rTr = rTr2

        p = r + scale * p

        k += 1

    return x, k
