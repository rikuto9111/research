モデル1

model.add(BatchNormalization())#バッチ正規化

model.add(Dropout(0.1))#ドロップアウト層

model.add(LSTM(LSTM_UNIT))#LSTM層

model.add(BatchNormalization())#バッチ正規化
model.add(Dense(train_oneh.shape[1], activation='softmax'))#全結合層



def extract(all_wavs):

    print("Feature extraction")
    all_data=[]
    for sub_wavs in all_wavs:
        sub_data=[]
        for wav in sub_wavs:
#            print(".",end="")
            if True:
                ELEMENT_COUNT=N_MFCC
                mfcc = librosa.feature.mfcc(y=wav, sr=SAMPLING_RATE, n_mfcc=N_MFCC, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)
                data = mfcc.transpose()#転置
            elif False:#実際には実行されない
                ELEMENT_COUNT=int(WINDOW_SIZE/2)+1
                stft = librosa.stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
                mag, phase = librosa.magphase(stft)
#                mag_db = librosa.amplitude_to_db(mag)
                data = mag[0:ELEMENT_COUNT,:].transpose()
            else:
                data = np.expand_dims(wav, axis=1)
                
            sub_data.append(data)
            
        all_data.append(sub_data)

    return all_data
    
モデル2

model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(8, kernel_size=1, strides=1, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(train_oneh.shape[1], activation='softmax'))

モデル3
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(8, kernel_size=1, strides=1, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LSTM(LSTM_UNIT))

    model.add(BatchNormalization())

    model.add(Dense(train_oneh.shape[1], activation='softmax'))


