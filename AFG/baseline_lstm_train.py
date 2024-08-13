"""
Code from
"AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition"
Fabien Ringeval, Björn Schuller, Michel Valstar, NIcholas Cummins, Roddy Cowie, Leili Tavabi, Maximilian Schmitt, Sina Alisamir, Shahin Amiriparian, Eva-Maria Messner, Siyang Song, Shuo Liu, Ziping Zhao, Adria Mallol-Ragolta, Zhao Ren, Mohammad Soleymani, Maja Pantic
Please see  https://github.com/AudioVisualEmotionChallenge/AVEC2019
"""
import glob
import os
import shutil

import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Masking, LSTM, TimeDistributed, Bidirectional, Flatten, Reshape
from keras.optimizer_v2.rmsprop import RMSprop
from CES_data import load_CES_data, load_features
from calc_scores import calc_scores

from numpy.random import seed
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, bidirectional, dropout,
                  num_targets):
    # Input layer
    inputs = Input(shape=(max_seq_len, num_features))

    # Masking zero input - shorter sequences
    net = Masking()(inputs)

    # 1st layer
    if bidirectional:
        net = Bidirectional(LSTM(num_units_1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(net)
    else:
        net = LSTM(num_units_1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(net)

    # 2nd layer
    if bidirectional:
        net = Bidirectional(LSTM(num_units_2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(net)
    else:
        net = LSTM(num_units_2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(net)
    second_lstm_output = net

    # Output layer (linear activation)
    outputs = []
    out1 = TimeDistributed(Dense(1))(net)
    outputs.append(out1)
    if num_targets >= 2:
        out2 = TimeDistributed(Dense(1))(net)
        outputs.append(out2)
    if num_targets == 3:
        out3 = TimeDistributed(Dense(1))(net)
        outputs.append(out3)

    # Create and compile model
    rmsprop = RMSprop(lr=learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=rmsprop, loss=ccc_loss)  # CCC-based loss function
    return model


def main(features_folders=['AVEC2019_CES_traindevel/audio_features_egemaps_xbow/'], path_output='predictions/audio_features_egemaps_xbow/'):
    # Input
    # features_folders = ['audio_features_egemaps_xbow/', 'visual_features_xbow/']  # Select features to be considered (must have the same hop size as the labels, i.e., 0.1s for CES)
    # path_output = 'predictions/'  # To store the predictions on the test (& development) partitions

    ## Configuration
    base_folder = r'/root/jupyter/'  # features and train/development labels
    targets = ['arousal', 'valence', 'liking']  # Targets to be learned at the same time

    output_predictions_devel = True  # Write predictions on development set
    test_available = False  # True, if test features are available

    # Neural net parameters
    standardise = True  # Standardise the input features (0 mean, unit variance)
    batch_size = 2  # Full-batch: 68 sequences
    learning_rate = 0.001  # default is 0.001
    num_epochs = 2  # Number of epochs
    num_units_1 = 64  # Number of LSTM units in LSTM layer 2
    num_units_2 = 32  # Number of LSTM units in LSTM layer 2
    bidirectional = False  # True/False
    dropout = 0.1  # Dropout

    # Labels
    shift_sec = 2.0  # Shift of annotations for training (in seconds)
    ## End Configuration

    ## Training
    labels_per_sec = 10  # 100ms hop size
    shift = int(np.round(shift_sec * labels_per_sec))
    num_targets = len(targets)

    # Set seeds to make results reproducible 
    # (Note: Results might be different from those reported by the Organisers as training of the seeds depend on hardware!
    seed(1)
    # set_random_seed(2)

    # Load AVEC2019_CES data
    print('Loading data ...')
    train_DE_x, train_HU_x, train_DE_y, train_HU_y, devel_DE_x, devel_HU_x, devel_DE_labels_original, devel_HU_labels_original, test_BR_x, test_CH_x = load_CES_data(
        base_folder, features_folders, targets, test_available)

    # Concatenate German and Hungarian cultures for Training and Development
    train_x = np.concatenate((train_DE_x, train_HU_x), axis=0)
    devel_x = np.concatenate((devel_DE_x, devel_HU_x), axis=0)
    train_y = []
    devel_labels_original = []
    for t in range(0, num_targets):
        train_y.append(np.concatenate((train_DE_y[t], train_HU_y[t]), axis=0))
        devel_labels_original.append(devel_DE_labels_original[t] + devel_HU_labels_original[t])
    # print(len(devel_labels_original[0]))
    # print(len(devel_labels_original))
    # Get some stats
    max_seq_len = train_x.shape[1]  # same for all partitions
    num_features = train_x.shape[2]
    print(' ... done')

    if standardise:
        MEAN, STDDEV = standardise_estimate(train_x)
        standardise_apply(train_x, MEAN, STDDEV)
        standardise_apply(devel_x, MEAN, STDDEV)
        standardise_apply(devel_DE_x, MEAN, STDDEV)
        standardise_apply(devel_HU_x, MEAN, STDDEV)
        # standardise_apply(test_DE_x, MEAN, STDDEV)
        # standardise_apply(test_HU_x, MEAN, STDDEV)
        # standardise_apply(test_CH_x, MEAN, STDDEV)
        # standardise_apply(test_BR_x, MEAN, STDDEV)

    # Shift labels to compensate annotation delay
    print('Shifting training labels to the front for ' + str(shift_sec) + ' seconds ...')
    for t in range(0, num_targets):
        train_y[t] = shift_labels_to_front(train_y[t], shift)
    print(' ... done')

    # Create model
    model = emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, bidirectional, dropout,
                          num_targets)
    print(model.summary())
    from keras.callbacks import LambdaCallback

    # 创建一个空列表，用于存储每一层的输出
    layer_outputs = []

    # Create a LambdaCallback that retrieves the output of each layer at the end of each epoch
    get_layer_outputs = LambdaCallback(
        on_epoch_end=lambda epoch, logs: layer_outputs.append([layer.output for layer in model.layers])
    )

    # 将LambdaCallback添加到模型的回调列表中
    callbacks_list = [get_layer_outputs]

    # Structures to store results (development)
    ccc_devel_best = np.ones(num_targets) * -1.

    # Train and evaluate model
    epoch = 1
    best_epochs = [0] * num_targets  # Initialize a list containing num_targets zeros to record the best epoch for each target
    while epoch <= num_epochs:
        model.fit(train_x, train_y, batch_size=batch_size, initial_epoch=epoch - 1, epochs=epoch,
                  callbacks=callbacks_list)  # Evaluate after each epoch

        # Evaluate on development partition
        ccc_iter = evaluate_devel(model, devel_x, devel_labels_original, shift, targets)

        # Print results
        print('CCC Development (' + ','.join(targets) + '): ' + str(np.round(ccc_iter * 1000) / 1000))

        # If CCC on the development partition improved, store the best models as HDF5 files
        for t in range(0, num_targets):
            if ccc_iter[t] > ccc_devel_best[t]:
                # print("ccc_iter",ccc_iter)
                # print("ccc_devel_best",ccc_devel_best)
                ccc_devel_best[t] = ccc_iter[t]
                best_epochs[t] = epoch
                save_model(model, targets[t] + '.hdf5')

        # first_epoch_first_layer_output = layer_outputs[0][3]



        # Next epoch
        epoch += 1

    # ... Training finished

    # Print best results on development partition
    print('CCC Development best (' + ','.join(targets) + '): ' + str(np.round(ccc_devel_best * 1000) / 1000))
    print("best_epoch:", best_epochs)
    # find_best(r"/root/jupyter/AVEC2019-master/Baseline_systems/CES",best_epochs)
    if output_predictions_devel:
        print('Getting predictions on Devel and shifting back')
        pred_devel_DE = []
        pred_devel_HU = []
        for t in range(0, num_targets):
            if num_targets == 1:
                pred_devel_DE.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(devel_DE_x), shift))
                pred_devel_HU.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(devel_HU_x), shift))
            else:
                pred_devel_DE.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(devel_DE_x)[t], shift))
                pred_devel_HU.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(devel_HU_x)[t], shift))
        print(
            'Writing predictions on Devel partitions for the best models (best CCC on the Development partition) for each dimension into folder ' + path_output)
        write_predictions(path_output, pred_devel_DE, targets, prefix='Devel_DE_', labels_per_sec=labels_per_sec)
        write_predictions(path_output, pred_devel_HU, targets, prefix='Devel_HU_', labels_per_sec=labels_per_sec)
        folder_path = r"/root/jupyter/AVEC2019_CES_traindevel/audio_features_egemaps_xbow"
        # 使用glob来匹配文件夹下的所有CSV文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        # 遍历所有匹配的CSV文件并输出它们的绝对路径
        for csv_file in csv_files:
            train_DE_x = np.empty((1, 1768, 0))

            absolute_path = os.path.abspath(csv_file)
            file_name = os.path.basename(absolute_path)
            with open(absolute_path) as infile:
                line = infile.readline()
            header = None
            if line[:4] == 'name':
                header = 'infer'
            sep = ';'
            if ',' in line:
                sep = ','

            num_inst = 1
            max_seq_len = 1768
            num_features = len(pd.read_csv(absolute_path, sep=";",
                                           header=header).columns) - 2  # do not consider instance name and time stamp
            features = np.empty((num_inst, max_seq_len, num_features))
            for n in range(0, num_inst):
                F = pd.read_csv(absolute_path, sep=sep, header=header,
                                usecols=range(2, 2 + num_features)).values
                if F.shape[0] > max_seq_len: F = F[:max_seq_len, :]  # might occur for some feature representations
                features[n, :, :] = np.concatenate(
                    (F, np.zeros((max_seq_len - F.shape[0], num_features))))  # zero padded
            train_DE_x = np.concatenate((train_DE_x,
                                         features), axis=2)
            mid_model = Model(model.input, model.get_layer('lstm_1').output)
            numpy_output = mid_model.predict(train_DE_x)[0]
            # # 将 NumPy 数组转换为 Pandas 数据帧
            df = pd.DataFrame(numpy_output)
            # 将数据帧保存到 CSV 文件
            csv_filename = fr"/root/jupyter/AVEC2019-master/Baseline_systems/CES/output/egemaps_xbow/{file_name}"  # 选择你要保存的文件名
            df.to_csv(csv_filename, index=False)  #  Select the file name you want to save

            print(f"Tensor has been saved to {csv_filename}")


    # Get predictions on test (and shift back) Write best predictions
    if test_available:
        print('Getting predictions on Test and shifting back')
        pred_test_BR = []
        pred_test_CH = []
        for t in range(0, num_targets):
            if num_targets == 1:
                pred_test_BR.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(test_BR_x), shift))
                pred_test_CH.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(test_CH_x), shift))
            else:
                pred_test_BR.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(test_BR_x)[t], shift))
                pred_test_CH.append(
                    shift_labels_to_back(load_model(targets[t] + '.hdf5', compile=False).predict(test_CH_x)[t], shift))
        print('Writing predictions on Test partitions for the best models (best CCC on the Development partition) for each dimension into folder ' + path_output)
        write_predictions(path_output, pred_test_BR, targets, prefix='Test_BR_', labels_per_sec=labels_per_sec)
        write_predictions(path_output, pred_test_CH, targets, prefix='Test_CH_', labels_per_sec=labels_per_sec)
        

        
        
def find_best(folder_path, best_epochs):
    for filename in os.listdir(folder_path):
        if filename.startswith('output') and filename.endswith('.csv'):
            # 提取文件名中的数字部分
            file_epoch = int(filename[6:-4])

            if file_epoch in best_epochs:
        
                new_filename = f'best_epoch{file_epoch}.csv'

                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)
                shutil.move(old_filepath, new_filepath)


def standardise_estimate(train):
    # Estimate parameters (masked parts are not considered)
    num_features = train.shape[2]
    estim = train.reshape([-1, num_features])
    estim_max_abs = np.max(np.abs(estim), axis=1)
    mask = np.where(estim_max_abs > 0)[0]
    MEAN = np.mean(estim[mask], axis=0)
    STDDEV = np.std(estim[mask], axis=0)
    return MEAN, STDDEV


def standardise_apply(partition, MEAN, STDDEV):
    # Standardise partition with given parameters
    for sub in range(0, partition.shape[0]):
        part_max_abs = np.max(np.abs(partition[sub, :, :]), axis=1)
        mask = np.where(part_max_abs > 0)[0]
        partition[sub, mask, :] = partition[sub, mask, :] - MEAN
        partition[sub, mask, :] = partition[sub, mask, :] / (STDDEV + np.finfo(np.float32).eps)
    return partition  # not required


def evaluate_devel(model, devel_x, label_devel, shift, targets):
    # Evaluate performance (CCC) on the development set
    #  -shift back the predictions in time
    #  -use the original labels (without zero padding)
    num_targets = len(targets)
    CCC_devel = np.zeros(num_targets)
    # Get predictions
    pred_devel = model.predict(devel_x)
    # print(pred_devel)
    # In case of a single target, model.predict() does not return a list, which is required
    if num_targets == 1:
        pred_devel = [pred_devel]
    for t in range(0, num_targets):
        # Shift predictions back in time (delay)
        pred_devel[t] = shift_labels_to_back(pred_devel[t], shift)
        CCC_devel[t] = evaluate_partition(pred_devel[t], label_devel[t])
    return CCC_devel


def evaluate_partition(pred, gold):
    # pred: np.array (num_seq, max_seq_len, 1)
    # gold: list (num_seq) - np.arrays (len_original, 1)
    pred_all = np.array([])
    gold_all = np.array([])
    for n in range(0, len(gold)):
        # cropping to length of original sequence
        len_original = len(gold[n])
        pred_n = pred[n, :len_original, 0]
        # global concatenation - evaluation
        pred_all = np.append(pred_all, pred_n.flatten())
        gold_all = np.append(gold_all, gold[n].flatten())
    ccc, _, _ = calc_scores(gold_all, pred_all)
    return ccc


def write_predictions(path_output, predictions, targets, prefix='Test_DE_', labels_per_sec=10):
    num_targets = len(targets)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    for n in range(0, predictions[0].shape[0]):
        seq_len = predictions[0].shape[1]
        pred_inst = np.empty((seq_len, num_targets))
        for t in range(0, num_targets):
            pred_inst[:, t] = predictions[t][n, :, 0]
        # add time stamp
        time_stamp = np.linspace(0., (seq_len - 1) / float(labels_per_sec), seq_len).reshape(-1, 1)
        pred_inst = np.concatenate((time_stamp, pred_inst), axis=1)
        # create data frame and write file
        instname = prefix + str(n + 1).zfill(2)
        filename = path_output + instname + '.csv'
        data_frame = pd.DataFrame(pred_inst, columns=['timestamp'] + targets)
        data_frame['name'] = '\'' + instname + '\''
        data_frame.to_csv(filename, sep=';', columns=['name', 'timestamp'] + targets, index=False, float_format='%.6f')

def write_mid_output(path_output, mid, prefix='Test_DE_'):
    # num_targets = len(targets)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    for n in range(0, mid.shape[0]):
        seq_len = mid.shape[0]
        pred_inst = np.empty((seq_len))

        # add time stamp
        # time_stamp = np.linspace(0., (seq_len - 1) / float(labels_per_sec), seq_len).reshape(-1, 1)
        # pred_inst = np.concatenate((time_stamp, pred_inst), axis=1)
        # create data frame and write file
        instname = prefix + str(n + 1).zfill(2)
        filename = path_output + instname + '.csv'
        data_frame = pd.DataFrame(pred_inst, columns=['timestamp'])
        data_frame['name'] = '\'' + instname + '\''
        data_frame.to_csv(filename, sep=';', index=False, float_format='%.6f')
def shift_labels_to_front(labels, shift=0):
    labels = np.concatenate((labels[:, shift:, :], np.zeros((labels.shape[0], shift, labels.shape[2]))), axis=1)
    return labels


def shift_labels_to_back(labels, shift=0):
    labels = np.concatenate(
        (np.zeros((labels.shape[0], shift, labels.shape[2])), labels[:, :labels.shape[1] - shift, :]), axis=1)
    return labels


def ccc_loss(gold,
             pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold = K.squeeze(gold, axis=-1)
    pred = K.squeeze(pred, axis=-1)
    gold_mean = K.mean(gold, axis=-1, keepdims=True)
    pred_mean = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold - gold_mean) * (pred - pred_mean)
    gold_var = K.mean(K.square(gold - gold_mean), axis=-1, keepdims=True)
    pred_var = K.mean(K.square(pred - pred_mean), axis=-1, keepdims=True)
    ccc = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    ccc_loss = K.constant(1.) - ccc
    return ccc_loss


if __name__ == '__main__':
    # Uni-modal
    # main(features_folders=['audio_features_mfcc_functionals/'],      path_output='predictions_mfcc-func/')
    main()
