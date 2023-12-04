import numpy as np

##  Zig-zag
def zig_zag(input_matrix, block_size):
    z = np.empty([block_size * block_size], dtype=int)
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = int(input_matrix[j, i - j])
            else:
                z[index] = int(input_matrix[i - j, j])
    return z

def zig_zag_reverse(input_matrix):
    block_size = 8
    output_matrix = np.empty([block_size, block_size], dtype=int)
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                output_matrix[j, i - j] = int(input_matrix[index])
            else:
                output_matrix[i - j, j] = int(input_matrix[index])
    return output_matrix