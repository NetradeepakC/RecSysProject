import numpy as np

def calculate_qr(matrix):
    rows, columns = matrix.shape

    q = np.zeros((rows, columns))
    r = np.zeros((columns, columns))

    for i in range(columns):
        current_column = matrix[:, i]
        for j in range(i):
            r[j, i] = np.dot(q[:, j], current_column)
            current_column = current_column - r[j, i] * q[:, j]
        r[i, i] = np.linalg.norm(current_column)
        q[:, i] = current_column / r[i, i]

    return r, q


def calculate_rank(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    total_rank = min(rows, columns)

    for current_row in range(total_rank):
        if matrix[current_row][current_row] == 0:
            is_rank_reduced = True
            for i in range(current_row + 1, rows):
                if matrix[i][current_row] != 0:
                    matrix[i], matrix[current_row] = matrix[current_row], matrix[i]
                    is_rank_reduced = False
                    break
            if is_rank_reduced:
                total_rank -= 1
                for i in range(current_row, rows):
                    matrix[i][current_row] = matrix[i][total_rank]
        else:
            for current_column in range(current_row + 1, rows):
                factor = matrix[current_column][current_row] / matrix[current_row][current_row]
                for i in range(current_row, columns):
                    matrix[current_column][i] -= matrix[current_row][i] * factor

    return total_rank


def calculate_eigen(matrix):
    total_itterations = 1000
    tolerence = 1e-8

    rows, columns = matrix.shape
    eigen_vector = np.random.randn(columns, columns)

    for _ in range(total_itterations):
        eigen_vector_new = matrix.dot(eigen_vector)
        _, eigen_vector_new = calculate_qr(eigen_vector_new)

        if np.allclose(eigen_vector_new, eigen_vector, rtol=tolerence):
            break
        eigen_vector = eigen_vector_new

    # compute the eigenvalues
    eigen_value = np.diag(eigen_vector.T.dot(matrix.dot(eigen_vector)))

    return eigen_vector, eigen_value


def SVD(matrix):
    if matrix.shape[0] > matrix.shape[1]:
        S = matrix.T.dot(matrix)
        S_copy = S.copy()
        total_rank = calculate_rank(S_copy)
    else:
        S = matrix.dot(matrix.T)
        S_copy = S.copy()
        total_rank = calculate_rank(S_copy)

    eigen_vectors, eigen_value = calculate_eigen(S)
    indices_sort = np.argsort(eigen_value)[::-1]
    eigen_vectors = eigen_vectors[:, indices_sort]
    eigen_value = eigen_value[indices_sort]

    s = np.sqrt(eigen_value)
    s = s[:total_rank]
    s_inverse = np.zeros_like(matrix.T)
    np.fill_diagonal(s_inverse, 1.0 / s)

    if matrix.shape[0] <= matrix.shape[1]:
        u = eigen_vectors
        V_T = np.dot(s_inverse, np.dot(u.T, matrix))
        if (len(s) != u.shape[1]):
            u = u[:, :len(s) - u.shape[1]]

    else:
        u = np.dot(matrix, np.dot(eigen_vectors, s_inverse))
        V_T = eigen_vectors.T
        if (len(s) != V_T.shape[0]):
            V_T = V_T[:len(s) - V_T.shape[0], :]


    sigma = np.zeros([u.shape[1], V_T.shape[0]])
    for i in range(len(s)):
        sigma[i, i] = s[i]

    return u, s, sigma, V_T


def ReducedSVD(matrix, removal_threshold=0, removed=0):
    u, s, sigma, V_T = SVD(matrix)

    if (removed < 0):
        print('The number of eigen values to be removed cannot be less than 0!!')
        exit()

    elif (removed > 0 and removed < len(s)):
        s = s[:-removed]
        u = u[:, :-removed]
        v_t = v_t[:-removed, :]
        sigma = sigma[:-removed, :-removed]

    if (removal_threshold < 0):
        print("Removal threshold cannot be less than 0!!")
        exit()

    elif (removal_threshold > 0 and removal_threshold < s[0]):
        s = s[s >= removal_threshold]
        u = u[:, :len(s)]
        v_t = v_t[:len(s), :]
        sigma = sigma[:len(s), :len(s)]


    return u, s, sigma, v_t
