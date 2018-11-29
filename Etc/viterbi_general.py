import numpy as np


def backtrace(z_pred, back, T, labels):
    prev_cursor = z_pred[-1]

    for m in np.arange(T)[::-1]:
        prev_cursor = back[prev_cursor, m]
        z_pred[m] = prev_cursor

    if labels is None:
        return z_pred

    return [labels[z] for z in z_pred]

    # Q = set of states; n is number of states
    # A = transition probability matrix;  aijaij : probability of transitioning from state  qiqi  to state  qjqj
    # V = set of possible observations; o is number of observs
    # X = sequence of observations; m is number of timesteps
    # B = emission matrix;  bijbij : probability of state  qiqi  emitting observation  xjxj
    # q0q0 : start state or a vector  /pi/pi  over start states

def viterbi(q_size, A, X, B, q_init, labels=None):
    Q = np.arange(q_size)
    T = len(X)
    p = np.zeros(shape=(len(Q), T))
    back = np.zeros(shape=(len(Q), T), dtype=np.int)
    z_pred = np.zeros(shape=(T, ), dtype=np.int)

    for q in Q:
        p[q, 0] = A[q_init, q] * B[q, X[0]]
        back[q, 0] = q_init

    for t in np.arange(T)[1:]:
        for q in Q:
            p[q, t] = np.max([p[qp, t - 1] * A[qp, q] * B[q, X[t]] for qp in Q])
            back[q, t] = np.argmax([p[qp, t - 1] * A[qp, q] for qp in Q])

    z_pred[T - 1] = np.argmax([p[q, T - 1] for q in Q])

    return backtrace(z_pred, back, T, labels)
