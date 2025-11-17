触覚
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


触覚音声モデル 1(early confusion)

merged = np.concatenate(([train_hap_data, train_aud_data]),axis = 2)
merged_valid = np.concatenate(([valid_hap_data, valid_aud_data]),axis = 2)

model = Sequential()
model.add(LSTM(LSTM_UNIT))#LSTM層
model.add(Dropout(0.1))#ドロップアウト層
model.add(BatchNormalization())#バッチ正規化
model.add(Dense(train_oneh.shape[1], activation='softmax'))

ほとんど今までの構造

これだと全然ダメ　触覚と音声の関係を学習していない

触覚音声モデル 2(early confusion)

merged = np.concatenate(([train_hap_data, train_aud_data]),axis = 2)
merged_valid = np.concatenate(([valid_hap_data, valid_aud_data]),axis = 2)

model = Sequential()

model.add(Conv1D(16, kernel_size=3, padding='same'))#そのままmfccに対して並べたものを渡しても触覚と音声でmfccは異なるしそう言った関係性とか学びにくいかも？
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.1))

model.add(LSTM(32))#時系列はもちろんConvの後は残っている
model.add(Dropout(0.1))

model.add(Dense(train_oneh.shape[1], activation='softmax'))

これは局所特長を先に抽出してからLSTMにかける感じ
これだと精度が高い

触覚音声モデル 3(early confusion)

merged = np.concatenate(([train_hap_data, train_aud_data]),axis = 2)
merged_valid = np.concatenate(([valid_hap_data, valid_aud_data]),axis = 2)

model = Sequential()

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

中邑モデル3と同じ構造

精度は悪くはないが、2のほうがよい







    model.add(Dense(train_oneh.shape[1], activation='softmax'))


