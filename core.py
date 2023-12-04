from dct import *
from huffman import *
from zigzag import *
import cv2
import collections
import numpy as np
from quantization import *


def compress_channel(img, qtable):
    
    # Kích thước ảnh
    iHeight, iWidth = img.shape[:2]

    zigZag = []

    # Nén ảnh theo từng khối 8x8
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            # Lấy khối 8x8
            block = img[startY:startY + 8, startX:startX + 8]

            # Tính DCT cho khối
            block_t = np.float32(block)
            dct = dct_block(block_t)

            # Lượng tử hóa các hệ số DCT
            block_q = quantize(dct,qtable)
            #print("block",block_q)
            # Zig Zag
            zigZag.append(zig_zag(block_q, 8))
            
    dc = []
    dc.append(zigZag[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag)):
        dc.append(zigZag[i][0] - zigZag[i - 1][0])

    # RLC cho giá trị AC
    rlc = []
    zeros = 0
    for i in range(0, len(zigZag)):
        zeros = 0
        for j in range(1, len(zigZag[i])):
            if (zigZag[i][j] == 0):
                zeros += 1
            else:
                rlc.append(zeros)
                rlc.append(zigZag[i][j])
                zeros = 0
        if (zeros != 0):
            rlc.append(zeros)
            rlc.append(0)
    #### Huffman ####

    # Huffman DPCM
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterDPCM = collections.Counter(dc)

    # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsDPCM = []
    for key, value in counterDPCM.items():
        probsDPCM.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsDPCM = makenodes(probsDPCM)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    rootDPCM = iterate(symbolsDPCM)

    # Mã hóa danh sách các ký hiệu nguồn.
    sDPMC = encode(dc, symbolsDPCM)

    # Huffman RLC
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterRLC = collections.Counter(rlc)

    # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsRLC = []
    for key, value in counterRLC.items():
        probsRLC.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsRLC = makenodes(probsRLC)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    root = iterate(symbolsRLC)

    # Mã hóa danh sách các ký hiệu nguồn.
    sRLC = encode(rlc, symbolsRLC)
    
    # return data
    dDPCM = decode(sDPMC, rootDPCM)
    decodeDPMC = []
    for i in range(0, len(dDPCM)):
        decodeDPMC.append(int(dDPCM[i]))

    # Huffman RLC
    # Giải mã một chuỗi nhị phân bằng cách sử dụng cây Huffman được truy cập thông qua root
    dRLC = decode(sRLC, root)
    decodeRLC = []
    for i in range(0, len(dRLC)):
        decodeRLC.append(int(dRLC[i]))

    # Inverse DPCM
    inverse_DPCM = []
    inverse_DPCM.append(decodeDPMC[0])
    for i in range(1, len(decodeDPMC)):
        inverse_DPCM.append(decodeDPMC[i] + inverse_DPCM[i - 1])

    # Inverse RLC
    inverse_RLC = []
    for i in range(0, len(decodeRLC)):
        if (i % 2 == 0):
            if (decodeRLC[i] != 0.0):
                if (i + 1 < len(decodeRLC) and decodeRLC[i + 1] == 0):
                    for j in range(1, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
                else:
                    for j in range(0, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
        else:
            inverse_RLC.append(decodeRLC[i])
    new_img = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(len(inverse_DPCM)):
        temp.append(inverse_DPCM[i])
        for j in range(0, 63):
            temp.append((inverse_RLC[j + i * 63]))
        temp2.append(temp)
        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = zig_zag_reverse(temp)
        #print("zigzag", inverse_blockq)
        # inverse DCT
        inverse_dct = idct_block(iQuantize(inverse_blockq,qtable))
        #print("quanti",iQuantize(inverse_blockq,qtable))
        #print("idct", inverse_dct)
        # Update new_img
        new_img[height:height + 8, width:width + 8] = inverse_dct

        # Update indices
        width += 8
        if width >= iWidth:
            width = 0
            height += 8

        temp = []
        temp2 = []

    np.place(new_img, new_img > 255, 255)
    np.place(new_img, new_img < 0, 0)
    return new_img