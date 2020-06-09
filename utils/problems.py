import numpy as np
import scipy
from scipy import linalg
from PIL import Image


def inpaint(x_shape, drop_prob=0.5, rng=None):
    if rng is not None:
        mask = rng.rand(*x_shape).astype(np.float32) > drop_prob; # binary drop mask
    else:
        mask = np.random.rand(*x_shape).astype(np.float32) > drop_prob; # binary drop mask
    A = np.diag(mask.flatten())
    return A, mask.shape


def block(x_shape, box_size_ratio, total_box=1, rng=None):
    box_size = int(box_size_ratio * x_shape[2])
    spare = 0.25 * box_size
    mask = np.ones(x_shape, dtype=np.float32)
    for f in range(x_shape[1]):
        for i in range(total_box):
            start_row = spare
            end_row = x_shape[2] - spare - box_size - 1
            start_col = spare
            end_col = x_shape[3] - spare - box_size - 1
            if rng is not None:
                idx_row = int(rng.rand(1) * (end_row - start_row) + start_row)
                idx_col = int(rng.rand(1) * (end_col - start_col) + start_col)
            else:
                idx_row = int(np.random.rand(1) * (end_row - start_row) + start_row)
                idx_col = int(np.random.rand(1) * (end_col - start_col) + start_col)
            mask[0, f, idx_row:(idx_row + box_size), idx_col:(idx_col + box_size), :] = 0. # binary drop mask
    A = np.diag(mask.flatten())
    return A, mask.shape


def center(x_shape, box_size_ratio):
    box_size = int(box_size_ratio * x_shape[2])
    mask = np.ones(x_shape, dtype=np.float32)
    idx_row = np.round(x_shape[2] / 2.0 - box_size / 2.0).astype(int)
    idx_col = np.round(x_shape[3] / 2.0 - box_size / 2.0).astype(int)
    mask[0, :, idx_row:(idx_row + box_size), idx_col:(idx_col + box_size), :] = 0. # binary drop mask
    A = np.diag(mask.flatten())
    return A, mask.shape


def superres(x_shape, resize_ratio):
    box_size = 1.0 / resize_ratio
    out_row = np.floor(x_shape[2] / box_size).astype(int)
    out_col = np.floor(x_shape[3] / box_size).astype(int)
    sr = np.zeros((out_row * out_col, x_shape[2] * x_shape[3]), dtype=np.float32)
    for i in range(out_row):
        for j in range(out_col):
            for i2 in range(i * int(box_size), (i+1) * int(box_size)):
                for j2 in range(j * int(box_size), (j+1) * int(box_size)):
                    sr[i * out_col + j, i2 * x_shape[3] + j2] = (resize_ratio ** 2)
    A = np.kron(np.kron(np.eye(x_shape[1]), sr), np.eye(x_shape[4]))
    return A, x_shape[:2] + (out_row, out_col) + x_shape[4:]


def cs(x_shape, compress_ratio, rng=None):
    d = np.prod(x_shape).astype(int)
    m = np.round(compress_ratio * d).astype(int)
    if rng is not None:
        A = rng.randn(m, d).astype(np.float32) / np.sqrt(m) # A is overcomplete random gaussian
    else:
        A = np.random.randn(m, d).astype(np.float32) / np.sqrt(m) # A is overcomplete random gaussian
    return A, (1, m,)


def videocs(x_shape, compress_ratio, rng=None):
    d = np.prod(x_shape[2:4]).astype(int)
    m = np.round(compress_ratio * d).astype(int)
    if rng is not None:
        A_measurement = rng.randn(m, d).astype(np.float32) / np.sqrt(m) # A is overcomplete random gaussian
    else:
        A_measurement = np.random.randn(m, d).astype(np.float32) / np.sqrt(m) # A is overcomplete random gaussian
    A_diff = (np.eye(x_shape[1], k=1) - np.eye(x_shape[1]))[:-1]
    A_tdmeasurement = np.kron(np.kron(A_diff, A_measurement), np.eye(x_shape[4]))
    A_reference = np.concatenate([np.eye(1 * x_shape[2] * x_shape[3] * x_shape[4]),
                                  np.zeros((1 * x_shape[2] * x_shape[3] * x_shape[4], (x_shape[1] - 1) * x_shape[2] * x_shape[3] * x_shape[4]))], axis=1)
    A = np.concatenate([A_reference, A_tdmeasurement], axis=0)
    return A, (1, A.shape[0],)


def conv2d(x_shape, F):
    # number columns and rows of the input
    I_row_num, I_col_num = x_shape[2:4]
    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape
    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0), (0, output_col_num - F_col_num)), 'constant', constant_values=0)
    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0] - 1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num - 1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = linalg.toeplitz(c, r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0] + 1)
    r = np.r_[c[0], np.zeros(I_row_num - 1, dtype=int)]
    doubly_indices = linalg.toeplitz(c, r)
    ## create doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0] * doubly_indices.shape[0]
    w = toeplitz_shape[1] * doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)
    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[doubly_indices[i, j] - 1]
    return doubly_blocked, x_shape[:2] + (output_row_num, output_col_num) + x_shape[4:]


def blurdisk(x_shape, size=7, radius=2.3):
    sz = (size - 1) / 2
    [x, y] = np.meshgrid(np.arange(-sz, sz + 1), np.arange(-sz, sz + 1))
    rad = np.sqrt(np.square(x) + np.square(y))
    F = (rad <= radius)
    F = (F / np.sum(F))
    C, shp_out = conv2d(x_shape, F)
    A = np.kron(np.kron(np.eye(x_shape[1]), C), np.eye(x_shape[4]))
    return A, shp_out


def blurmotion(x_shape, size=7, angle=None, rng=None):
    if angle is None:
        if rng is not None:
            angle = 360. * np.random.rand()
        else:
            angle = 360. * np.random.rand()
    sz = (size - 1) // 2
    F = np.zeros((sz, sz), dtype=np.float32)
    F[int(np.round(sz / 2)), :] = 1
    img = Image.fromarray(F)
    img = img.convert('L')
    img = img.rotate(angle).resize(F.shape).getdata()
    F = np.reshape(np.array(img), F.shape).astype(np.float32)
    F = (F / np.sum(F))
    C, shp_out = conv2d(x_shape, F)
    A = np.kron(np.kron(np.eye(x_shape[1]), C), np.eye(x_shape[4]))
    return A, shp_out


def videoblurdisk(x_shape, size=7, radius=2.3):
    sz = (size - 1) / 2
    x = np.meshgrid(np.arange(-sz, sz + 1))
    rad = np.sqrt(np.square(x))
    F = (rad <= radius)
    F = (F / np.sum(F))
    C, shp_out = conv2d(x_shape[2:4] + x_shape[:2] + (x_shape[4],), F)
    A = np.kron(np.kron(C, np.eye(x_shape[2] * x_shape[3])), np.eye(x_shape[4]))
    return A, shp_out[2:4] + shp_out[:2] + shp_out[4:]


def frameinterp(x_shape, interp_ratio):
    interp_size = 1.0 / interp_ratio
    out_frame = np.floor(x_shape[1] / interp_size).astype(int)
    fi = np.zeros((out_frame, x_shape[1]), dtype=np.float32)
    for f in range(out_frame):
        f2 = f * int(interp_size)
        fi[f, f2] = 1
    A = np.kron(fi, np.eye(x_shape[2] * x_shape[3] * x_shape[4]))
    return A, x_shape[:1] + (out_frame,) + x_shape[2:]


def prediction(x_shape, predict_ratio):
    delta = np.floor(x_shape[1] * predict_ratio).astype(int)
    A = np.concatenate([np.eye((x_shape[1] - delta) * x_shape[2] * x_shape[3] * x_shape[4]),
                        np.zeros(((x_shape[1] - delta) * x_shape[2] * x_shape[3] * x_shape[4], delta * x_shape[2] * x_shape[3] * x_shape[4]))], axis=1)
    return A, x_shape[:1] + (x_shape[1] - delta,) + x_shape[2:]


def colorization(x_shape):
    if x_shape[4] == 1:
        A = np.eye(x_shape[1] * x_shape[2] * x_shape[3])
    elif x_shape[4] == 3:
        A = np.kron(np.eye(x_shape[1] * x_shape[2] * x_shape[3]), np.array([0.299, 0.587, 0.114]))
    return A, x_shape[:4] + (1,)

