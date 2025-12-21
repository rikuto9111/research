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












一応データの取り方

モデルはLSTM

for i,target in enumerate(subjects):#被験者31人に対して

        ct = 0#感情文カウンタ

        sub_file=[]#通常の100文
        sub_wavs=[]
        sub_labs=[]

        sub_file_up=[]#50文　各感情10文 テスト用
        sub_wavs_up=[]
        sub_labs_up=[]


        sub_file_down=[]#50文 各感情10文 fine_tuning用
        sub_wavs_down=[]
        sub_labs_down=[]
    
        for fn in sorted(glob.glob(path+"*"+target+"*/*.wav")):#sortedすることで上から順番に100文のうち,0~19..normal,20~39..angry,40~59..bash,60~79..happy,80~99 sad になっている

            lab = load_label(fn)
            
            if lab != -1:
                
                 
                wav = load_wave(fn)
                if i < 8:#前データの正規化
                    wav = wav * scale
                #wav = wav*scale_all[target] 

                if wav.shape[0] != 0:
                    if min_len > wav.shape[0]:
                        min_len = wav.shape[0]

                    if max_len < wav.shape[0]:#音声の時間の最小と最大を見つける
                        max_len = wav.shape[0]

                    sub_file.append(fn)
                    sub_wavs.append(wav)
                    sub_labs.append(lab)

                    if ct < 10:
                        sub_file_up.append(fn)#被験者X[wav0,wav1,..wav9,wav20,wav21,...] #各感情10文ずつ
                        sub_wavs_up.append(wav)
                        sub_labs_up.append(lab)#[lab0,lab1...lab9]
                    
                    if 10 <= ct < 20:#被験者X[wav10,wav11,..wav19,wav30,wav31,...]
                        sub_file_down.append(fn)
                        sub_wavs_down.append(wav)
                        sub_labs_down.append(lab)

            ct = (ct + 1)%20#20文でリセット 次の感情へ

        all_file.append(sub_file)#all_file,all_file_up,down[("被験者名",subfile),[("被験者名",subfile)・・・]]がはいってるってことか
        all_wavs.append(sub_wavs)
        all_labs.append(sub_labs)

        all_file_up.append(sub_file_up)#all_wavs,all_wav_up,downも同様
        all_wavs_up.append(sub_wavs_up)
        all_labs_up.append(sub_labs_up)

        all_file_down.append(sub_file_down)#all_fileには[("被験者名",subfile),[("被験者名",subfile)・・・]]がはいってるってことか
        all_wavs_down.append(sub_wavs_down)
        all_labs_down.append(sub_labs_down)


    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len









def main():
    
    all_file, all_wavs, all_labs,all_file_up, all_wavs_up, all_labs_up,all_file_down, all_wavs_down, all_labs_down = load_data(data_dir)
    
    aug_file, aug_wavs, aug_labs = data_augmentation(all_file, all_wavs, all_labs)#全ファイルに対してのデータ補強4倍

    aug_data = extract(aug_wavs)#MFCC変換
   
    aug_oneh = onehot_label(aug_labs)#全ファイルの感情ラベル1,2,3[0,1,0,0] , [0,0,1,0], [0,0,0,1] 変換


    
    val_file, val_wavs, val_labs = make_testdata(all_file_up, all_wavs_up, all_labs_up)#真テストデータの真ん中部分

    val_file_down, val_wavs_down, val_labs_down = make_testdata(all_file_down, all_wavs_down, all_labs_down)#trainにかける被験者データも真ん中を抜き取り　つまり補強データはかけない

    val_data = extract(val_wavs)#同じように
    val_data_down = extract(val_wavs_down)#同じようにmfcc変換

    val_oneh = onehot_label(val_labs)#真のテストデータの感情ラベル
    val_oneh_down = onehot_label(val_labs_down)#被験者trainデータの感情ラベル

    data_normalization(aug_data, val_data,val_data_down)#
    
    test_corr=np.empty(0)
    test_pred=np.empty(0)

    print("Model training")





通常のtrainを行った後
fine_tuning

def fine_tuning(model,train_fine_set,valid_set):

    train_file_fine, train_data_fine, train_oneh_fine = train_fine_set
    valid_file, valid_data, valid_oneh = valid_set
    base_weights = model.get_weights()

    

    best_acc = -1

    for layer in model.layers:
        layer.trainable = False


    model.layers[-4].trainable = True 
    model.layers[-3].trainable = True 
    model.layers[-2].trainable = True 
    model.layers[-1].trainable = True  

"""
    for layer in model.layers:#結合層だけ動かすパターン
        layer.trainable = False

    model.layers[-1].trainable = True  
  """  

    model.compile(#モデルの重み更新方法 loss,学習中に表示される性能指標を定義 根本的なものの定義
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
     )

    


    
    
    batch_data_array = []
    batch_oneh_array = []

    model.set_weights(base_weights)#最初だけ
#重み引き継ぎ
    
    for i in range(10):#5 10 15 20 ... 文追加 続きの重みを使っていく

        early_stop = EarlyStopping(
        monitor='val_accuracy',    # もしくは 'val_accuracy'
        patience=15,            # 何エポック改善がなければ停止するか
        restore_best_weights=True  # 停止時に最良モデルの重みを復元
        )

        

        for j in range(0,50,10):
            batch_data_array.append(train_data_fine[i+j])
            batch_oneh_array.append(train_oneh_fine[i+j])
        
        batch_data = np.array(batch_data_array)#5 10 15文
        batch_oneh = np.array(batch_oneh_array)

        #modelチューニング 
        history = model.fit(batch_data, batch_oneh, validation_data=(valid_data, valid_oneh), epochs=100,batch_size = 1,callbacks=[early_stop],verbose=2)


        val_acc = max(history.history['val_accuracy'])
        print(f"val_acc = {val_acc:.4f}")

        if val_acc > best_acc:#精度を更新したらその重みを保存
            best_acc = val_acc
            best_weights = model.get_weights()
        
    model.set_weights(best_weights)#最後に最も良くなった重みを復元

    return model









5文 ->> 5文 --> 5文 --> 5文 のやつ
↓
↓
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
     )

    


    
    
    batch_data_array = []
    batch_oneh_array = []

    model.set_weights(base_weights)
    for i in range(10):

        

        batch_data_array = []#5 引き継ぎ 10 引き継ぎではなく  多分ここだけ
        batch_oneh_array = []#5引き継ぎ次の5引き継ぎ -> にする
        
        

        early_stop = EarlyStopping(
        monitor='val_accuracy',  
        patience=15,           
        restore_best_weights=True  
        )

        

        for j in range(0,50,10):
            batch_data_array.append(train_data_fine[i+j])
            batch_oneh_array.append(train_oneh_fine[i+j])
        
        batch_data = np.array(batch_data_array)
        batch_oneh = np.array(batch_oneh_array)

        history = model.fit(batch_data, batch_oneh, validation_data=(valid_data, valid_oneh), epochs=100,batch_size = 1,callbacks=[early_stop],verbose=2)

















