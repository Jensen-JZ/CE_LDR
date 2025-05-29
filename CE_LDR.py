import re
import sys
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def config_parse():
    parser = argparse.ArgumentParser(description='Path Parser')
    parser.add_argument('--input_path', type=str, required=True, default='./ori.png', help='Specify the path of input image.')
    parser.add_argument('--alpha', type=float, default=2.5, help='Specify the value of alpha in equation 23. Default is 2.5.')
    parser.add_argument('--U', type=np.load, default=None, help='Specify the path to a .npy file for the matrix U in equation 31. Optional.')

    args = parser.parse_args()
    
    if args.U is not None:
        args.U = np.load(args.U)

    return args

def LDR(input_data, alpha, U_matrix, fname):
    print('Wait...')

    if U_matrix is None:
        U = np.zeros((255, 255))
        tmp_k = np.arange(255)
        for layer in range(255):
            U[:, layer] = np.minimum(tmp_k, 255 - layer) - np.maximum(tmp_k - layer, 0) + 1
    else:
        U = U_matrix

    [HEIGHT, WIDTH, DEEP] = input_data.shape
    if (HEIGHT == 256) & ((WIDTH * DEEP) == 256) and DEEP == 1: # Assuming if h2D_in is passed, it's 2D
        h2D_in = input_data.squeeze() # Ensure it's 2D
    else:
        in_Y = input_data.astype(np.int16) # Ensure integer type for indexing
        h2D_in = np.zeros((256, 256), dtype=np.int32)

        # Reshape in_Y to combine WIDTH and DEEP for easier pair finding, as per original logic
        # The original code iterates j from 0 to (WIDTH * DEEP) - 1
        # ref = in_Y[i, j % WIDTH, math.floor(j / WIDTH)]
        # This is equivalent to iterating a reshaped array:
        in_Y_flat = in_Y.reshape(HEIGHT, WIDTH * DEEP)

        # Horizontal pairs
        ref_h = in_Y_flat[:, :-1]
        trg_h = in_Y_flat[:, 1:]
        
        max_h = np.maximum(ref_h, trg_h)
        min_h = np.minimum(ref_h, trg_h)
        np.add.at(h2D_in, (max_h.ravel(), min_h.ravel()), 1)

        # Vertical pairs
        ref_v = in_Y_flat[:-1, :]
        trg_v = in_Y_flat[1:, :]

        max_v = np.maximum(ref_v, trg_v)
        min_v = np.minimum(ref_v, trg_v)
        np.add.at(h2D_in, (max_v.ravel(), min_v.ravel()), 1)

    D = np.zeros((255, 255))
    s = np.zeros((255, 1))

    for layer in range(255):
        # The original loop for h_l was:
        # for i in range(1 + layer, 256): (rows from 1+layer to 255)
        #     j = i - layer - 1          (columns from 0 to 254-layer)
        #     h_l[tmp_idx, 0] = math.log(h2D_in[i, j] + 1)
        # This corresponds to the diagonal k = -(layer + 1)
        # The elements are h2D_in[idx, idx - (layer+1)]
        # If h2D_in has shape (M, N), np.diag(v, k) considers elements v[i, i+k]
        # We want h2D_in[row, col] where col = row - (layer+1), so row - col = layer+1
        # So, if we consider h2D_in[i, j], then i-j = layer+1.
        # For np.diag(v, k), k = j - i. So, k = -(layer+1).
        # The diagonal starts at h2D_in[layer+1, 0]
        
        # Extract the specific diagonal from h2D_in
        # The diagonal has elements h2D_in[layer+1, 0], h2D_in[layer+2, 1], ..., h2D_in[255, 254-layer]
        # This is equivalent to np.diag(h2D_in[layer+1:, :255-layer], k=0) but simpler:
        diag_elements = np.diag(h2D_in, k=-(layer + 1))
        
        # We need to select the relevant part of this diagonal.
        # The original loop for i from 1+layer to 255 means it takes 255-(1+layer)+1 = 255-layer elements.
        # The first element is h2D_in[1+layer, 0].
        # np.diag(h2D_in, k=-(layer+1)) will give elements h2D_in[layer+1,0], h2D_in[layer+2,1], ...
        # The length of this diagonal is min(256-(layer+1), 256) = 255-layer
        h_l_values = diag_elements[:255-layer].copy() # Ensure correct length

        h_l = np.log1p(h_l_values).reshape(-1, 1)

        s[layer, 0] = np.sum(h_l)

        if s[layer, 0] == 0:
            continue

        tmp_m_l = np.polymul(h_l.reshape(-1), np.ones((layer + 1, 1)).reshape(-1)).reshape(-1, 1)

        m_l = tmp_m_l if len(tmp_m_l) == 255 else np.insert(tmp_m_l, 0, np.zeros((255 - len(tmp_m_l),))).reshape(-1, 1)

        d_l = (m_l - min(m_l)) / U[:, layer].reshape(-1, 1)

        if sum(d_l) == 0:
            continue

        D[:, layer] = (d_l / sum(d_l)).reshape(-1)

    W = (s / max(s)) ** alpha
    d = np.matmul(D, W)

    d = d / sum(d)
    tmp = np.zeros((256, 1))

    for k in range(255):
        tmp[k + 1] = tmp[k] + d[k]

    x = 255 * tmp

    # Ensure input_data is integer type for indexing
    # PIL images are usually uint8, but explicit conversion is safer.
    # x contains the mapping, input_data contains the original pixel values.
    output = x[input_data.astype(np.uint8)]


    CE_IMG = np.uint8(np.around(output))
    print('Finished!')

    plt.title('Enhancement_Image')
    plt.imshow(CE_IMG)
    plt.imsave('ce_' + fname + '.png', CE_IMG)
    plt.show()

def main():
    args = config_parse()
    input_data = np.array(Image.open(args.input_path))
    fname = re.split('[./\\\]', args.input_path)[-2] + '_' + re.split('[./\\\]', args.input_path)[-1]
    plt.title('Original_Image')
    plt.imshow(np.array(Image.open(args.input_path)))
    plt.show()
    LDR(input_data, args.alpha, args.U, fname) # Pass args.U which might be None or the loaded array

if __name__ == "__main__":
    main()