import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import os
import shutil
import random
import h5py

def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    # print(np.min(X), np.max(X))
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.float32(X)


def getBlurMatrix(kernal_size, sigma):
    '''
    get Blur matrix B
    :param kernal_size:
    :param sigma:
    :return:
    '''
    side = cv2.getGaussianKernel(kernal_size, sigma)
    Blur = np.multiply(side, side.T)
    return Blur


def get_kernal(kernal_size, sigma, rows, cols):
    '''
    Generate a Gaussian kernel and make a fast Fourier transform
    :param kernal_size:
    :param sigma:
    :return:
    '''
    # Generate 2D Gaussian filter
    blur = cv2.getGaussianKernel(kernal_size, sigma) * cv2.getGaussianKernel(kernal_size, sigma).T
    psf = np.zeros([rows, cols])
    psf[:kernal_size, :kernal_size] = blur
    # Cyclic shift, so that the Gaussian core is located at the four corners
    B1 = np.roll(np.roll(psf, -kernal_size // 2, axis=0), -kernal_size // 2, axis=1)
    # Fast Fourier Transform
    fft_b = np.fft.fft2(B1)
    # return fft_b
    return fft_b


def spectralDegrade(X, R, addNoise=True, SNR=40):
    '''
    spectral downsample
    :param X:
    :param R:
    :return:
    '''
    height, width, bands = X.shape
    X = np.reshape(X, [-1, bands], order='F')
    Z = np.dot(X, R.T)
    Z = np.reshape(Z, [height, width, -1], order='F')

    if addNoise:
        h, w, c = Z.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Z)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Z += sigmah * np.random.randn(h, w, c)

    return Z


def Blurs(X, B, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :return:
    '''
    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)
    return Y


def downSample(X, B, ratio, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''

    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)

    # downsample
    Y = Y[::ratio, ::ratio, :]
    return Y


def readCAVEData(path, mat_path):
    '''
    Read initial CAVE data
    since the original data is standardized we do not repeat it
    :return:
    '''
    hsi = np.zeros([512, 512, 31], dtype=np.float32)
    count = 0
    for dir in os.listdir(path):
        concrete_path = path + '/' + dir + '/' + dir
        for i in range(31):
            fix = str(i + 1)
            if i + 1 < 10:
                fix = '0' + str(i + 1)
            png_path = concrete_path + '/' + dir + '_' + fix + '.png'
            try:
                hsi[:, :, i] = plt.imread(png_path)
            except:
                img = plt.imread(png_path)
                img = img[:, :, :3]
                img = np.mean(img, axis=2)
                hsi[:, :, i] = img

        count += 1
        print('%d has finished' % count)
        sio.savemat(mat_path + str(count) + '.mat', {'HS': hsi})
 
 
def readChkData(path, mat_path):  
    
    #Read initial Chikusei data
    path = path + 'Hyperspec_Chikusei_MATLAB/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
    C_data = h5py.File(path)
    data = C_data['chikusei']
    print(data.shape)
    data = np.transpose(data)
    print(data.shape)
    # crop an image of size 2048×2048 along the top left corner
    data1 = data[:2048, :2048, :]
    # [0,1]
    data1 = standard(data1)
    print('done')
    print('-----start dividing data-----')

    # divide it into 64 images of size 256 × 256 by non-overlapping partitions
    rows, cols, _ = data1.shape
    piece_size = 256
    stride = piece_size
    count = 0
    for x in range(0, rows - piece_size + stride, stride):
        for y in range(0, cols - piece_size + stride, stride):
            data_piece = data1[x:x + piece_size, y:y + piece_size, :]
            count += 1
            sio.savemat(mat_path + '%d.mat' % count, {'HS': data_piece})
            print('piece num %d has saved' % count)
    print(count)
    print('done')


def readBotswanaData(path, mat_path):

    #Read initial Bostwana data
    path = path + 'Botswana.mat'
    data = sio.loadmat(path)['Botswana']
    data = standard(data)
    print(data)
    # crop an image of size 1280 × 256 along the top left corner
    data1 = data[:1280, :, :]
    print('done')
    print('-----start dividing data-----')
    # divide it into 20 images of size 128 × 128 by non-overlapping partitions
    rows, cols, _ = data1.shape
    piece_size = 128
    stride = piece_size
    count = 0
    for x in range(0, rows - piece_size + stride, stride):
        for y in range(0, cols - piece_size + stride, stride):
            data_piece = data1[x:x + piece_size, y:y + piece_size, :]
            count += 1
            sio.savemat(mat_path + '%d.mat' % count,{'HS': data_piece})
            print('piece num %d has saved' % count)
        print(count)
    print('done')

def readLAData(path, mat_path):
    
    # Read initial Los Angeles data
    path = path + 'dataLA.mat'

    data = h5py.File(path)

    pan = data['HRpan_LA']
    pan_data = np.transpose(pan)
    pan_data = np.expand_dims(pan_data, axis=2)
    standard(pan_data)
    lrhs = data['LRhsi_LA145']
    lrhs_data = np.transpose(lrhs)
    lrhs_data = cv2.resize(lrhs_data, (90, 90), interpolation=cv2.INTER_CUBIC)
    standard(lrhs_data)

    sio.savemat(mat_path + '1.mat', {'PAN': pan_data, 'LRHS': lrhs_data})
    print('done')

def createSimulateData(data_index, sample_path, B,  ratio):

    if data_index == 0:
        # CAVE
        mat_path = sample_path + 'CAVEMAT_r4/'
        num_start = 1
        num_end = 32

    elif data_index == 1:
        # CHK
        mat_path = sample_path + 'CHKMAT_r4/'
        num_start = 1
        num_end = 64

    elif data_index == 2:
        # BST
        mat_path = sample_path + 'BSTMAT_r4/'
        num_start = 1
        num_end = 20

    for i in range(num_start, num_end + 1):

        mat = sio.loadmat(mat_path + '%d.mat' % i)
        hs = mat['HS']

        pan = np.mean(hs, axis=2)
        pan = np.expand_dims(pan, axis=2)

        lrhs = downSample(hs, B, ratio, False)

        sio.savemat(mat_path + str(i) + '.mat', {'label': hs, 'Y': lrhs, 'P': pan})
        print('%d has finished' % i)


def cutTrainingPiecesForSimulatedDataset(data_index, sample_path):
    '''
    produce training pieces
    :param train_index:
    :return:
    '''
    if data_index == 0:
        # CAVE
        # the first 20 images are patched for training
        piece_size = 32
        stride = 20
        rows, cols = 512, 512
        num_start = 1
        num_end = 20
        mat_path = sample_path + 'CAVEMAT_r4/'
        count = 0
        ratio = 4
        piece_save_path = sample_path + 'CAVE_patch_32/train/'

    elif data_index == 1:
        # CHK
        # the first 45 images are patched for training
        piece_size = 32
        stride = 14
        rows, cols = 256, 256
        num_start = 1
        num_end = 45
        mat_path = sample_path + 'CHKMAT_r4/'
        count = 0
        ratio = 4
        piece_save_path = sample_path + 'CHK_patch_32/train/'

    elif data_index == 2:
        # the first 14 images are patched for training
        piece_size = 32
        stride = 3
        rows, cols = 128, 128
        num_start = 1
        num_end = 14
        mat_path = sample_path + 'BSTMAT_r4/'
        count = 0
        ratio = 4
        piece_save_path = sample_path + 'BST_patch_32/train/'

    elif data_index == 3:
        piece_size = 48
        stride = 3
        rows, cols = 360, 360
        num_start = 1
        num_end = 1
        mat_path = sample_path + 'LAMAT_r4/'
        count = 0
        ratio = 4
        piece_save_path = sample_path + 'LA_patch_48/train/'

    os.makedirs(piece_save_path, exist_ok=True)
    if data_index in range(3):
        print("-----------------------------------")
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            X = mat['label']
            Y = mat['Y']
            Z = mat['P']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    Z_piece = Z[x:x + piece_size, y:y + piece_size, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'Y': Y_piece,  'X': label_piece, 'P': Z_piece})
                    count += 1
                    print('piece num a%d has saved' % count)
            print('%d has finished' % i)

    else:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            Y = mat['LRHS']
            P = mat['PAN']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    p_piece = P[x:x + piece_size, y:y + piece_size, :]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'Y': Y_piece,  'P': p_piece})
                    count += 1
                    print('piece num a%d has saved' % count)
            print('%d has finished' % i)

    reRankfile(piece_save_path)
    print('done')



def reRankfile(path):
    '''
    Reorder mat by renaming and shuffling
    :param path: directory where the files are located
    :return:
    '''
    count = 0
    file_list = os.listdir(path)
    # Filter out non-MAT files (optional but recommended for robustness)
    mat_files = [file for file in file_list if file.endswith('.mat')]

    # Shuffle the list of files
    random.shuffle(mat_files)

    # Rename files starting from 0
    for file in mat_files:
        try:
            # Generate the new filename and check if it exists
            newname = str(count)
            new_filepath = os.path.join(path, f'{newname}.mat')

            # If the file already exists, skip to the next number
            while os.path.exists(new_filepath):
                count += 1
                newname = str(count)
                new_filepath = os.path.join(path, f'{newname}.mat')

            # Rename the file to the new name
            print(f'Renaming {file} to {newname}.mat')
            os.rename(os.path.join(path, file), new_filepath)

            # Increment the counter for the next filename
            count += 1

        except Exception as e:
            print(f'Error renaming {file}: {e}')

    print('Reordering complete')

if __name__ == '__main__':
    # manipulating datasets including CAVE, Harvard, University of Houston, CAVE with non-integer resolution ratio
    data_index = 3  # 0, 1 ,2, 3 represents the four datasets, respectively

    if data_index == 0:
        # CAVE
        path = r'D:\all_datasets\original_data\CAVE/'  # replace with your path which puts the downloading CAVE data
        sample_path = r'D:\all_datasets\Sample_data\CAVE/' # the path of saving the entile HSI with .mat format
        os.makedirs(sample_path + 'CAVEMAT_r4/', exist_ok=True)
        readCAVEData(path, sample_path + 'CAVEMAT_r4/')

        # # # produce the simulated data according to the Wald's protocol
        B = get_kernal(4, 2, 512, 512)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        ratio = 4  # the spatial resolution ratio
        createSimulateData(data_index, sample_path, B, ratio)

        # # produce the training pieces
        cutTrainingPiecesForSimulatedDataset(data_index, sample_path)


    elif data_index == 1:
        # Chikusei

        path = r'D:\all_datasets\original_data\Chikusei/'  # replace with your path which puts the downloading CAVE data
        sample_path = r'D:\all_datasets\Sample_data\Chikusei/'  # the path of saving the entile HSI with .mat format
        os.makedirs(sample_path + 'CHKMAT_r4/', exist_ok=True)

        readChkData(path, sample_path + 'CHKMAT_r4/')

        # # produce the simulated data according to the Wald's protocol
        B = get_kernal(4, 2, 256, 256)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        ratio = 4  # the spatial resolution ratio
        createSimulateData(data_index, sample_path, B, ratio)

        # # # produce the training pieces
        cutTrainingPiecesForSimulatedDataset(data_index, sample_path)


    elif data_index == 2:
        # Bostwana
        path = r'D:\all_datasets\original_data\Bostwana/'  # replace with your path which puts the downloading Harvard data
        sample_path = r'D:\all_datasets\Sample_data\Bostwana/' # the path of saving the entile HSI with .mat format
        # # convert the data into .mat format
        os.makedirs(sample_path + 'BSTMAT_r4/', exist_ok=True)

        readBotswanaData(path, sample_path + 'BSTMAT_r4/')

        B = get_kernal(4, 2, 128, 128)  # the blurring kernel with size of 8*8 and a standard deviation of 2

        ratio = 4  # the spatial resolution ratio
        createSimulateData(data_index, sample_path, B, ratio)

        # ## produce the training pieces
        cutTrainingPiecesForSimulatedDataset(data_index, sample_path)


    elif data_index == 3:

        path = r'D:\all_datasets\original_data\Los_Angeles/'  # replace with your path which puts the downloading Harvard data
        sample_path = r'D:\all_datasets\Sample_data\Los_Angeles/'
        # # convert the data into .mat format
        os.makedirs(sample_path + 'LAMAT_r4/', exist_ok=True)
        readLAData(path, sample_path + 'LAMAT_r4/')

        # ## produce the training pieces
        cutTrainingPiecesForSimulatedDataset(data_index, sample_path)





