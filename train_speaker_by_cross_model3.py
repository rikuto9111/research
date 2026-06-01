# coding: UTF-8

import os
import glob
import numpy as np
import sys
import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed
#from tensorflow.keras.layers.Permute import Permute
import librosa
import librosa.display

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

MAX_DATA_LENGTH=0
MIN_DATA_LENGTH=100000000

SAMPLING_RATE=16000
WINDOW_SIZE=2048
#WINDOW_SIZE=512
HOP_LENGTH=1024
N_MFCC=13
#HOP_LENGTH=256

ELEMENT_COUNT=int(WINDOW_SIZE/2)+1
#ELEMENT_COUNT=1
ELEMENT_COUNT=N_MFCC
#ELEMENT_COUNT=1025
#LSTM_UNIT=8
#LSTM_UNIT=32#ver11_21
LSTM_UNIT=64

BATCH_SIZE=int(sys.argv[1])#ここでコマンドラインで指定したバッチサイズ(1度に使用するデータ)
EPOCHS=int(sys.argv[2])#データをすべて使用するのを何回繰り返すか　コマンドライン
emo = 5
seed = 0

TRAIN_ID = "B"+str(BATCH_SIZE)+"_E"+str(EPOCHS)

data_hap_dir = "../../new_data/haptic/" 
data_aud_dir = "../../new_data/audio/" 

#data_dir_train = "../new_data/audio_speaker/audio_train_A/" 
#data_dir_test = "../new_data/audio_speaker/audio_test_A/"
#subjects = ["aohara", "hasegawaT", "hodotsuka", "horio", "miyake", "mizuno", "yamadaM", "yamashita","iwasakeisho","hattori_naoki"]
subjects = ["aohara_wav_output_org", "hasegawa_wav_output_org", "hodotsuka_wav_output_org", "horio_wav_output_org", "miyake_wav_output_org", "mizuno_wav_output_org","yamadaM_wav_output_org","yamashita_wav_output_org","newiwa_wav_output_org","hattori_wav_output","miyasita_wav_output_org","nakagawa_wav_output_org","newhatori_wav_output_org","takase_wav_output_org","imamura_wav_output_org","kariya_wav_output_org","kimura_wav_output_org","takagi_wav_output_org","atusi_wav_output_org", "kasiwagi_wav_output_org", "kunda_wav_output_org", "newyamasita_wav_output_org", "satou_wav_output_org", "yasunaga_wav_output_org","asano_wav_output_org","kamiya_wav_output_org","sinohara_wav_output_org","isii_wav_output_org","onda_wav_output_org","tamaki_wav_output_org","iida_wav_output_org"]

#subjects = ["aohara_wav_output", "hasegawa_wav_output", "hodotsuka_wav_output"]


"""
if emo == 4:#4クラス分類の場合
    label_type=['normal', 'angry', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-hap', 'st-sad']
elif emo == 5:#5クラス分類の場合
    label_type=['normal', 'angry', 'bash', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-bas', 'st-hap', 'st-sad']
else:
    label_type=['normal', 'angry', 'bash', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-bas', 'st-hap', 'st-sad']
"""


def main():

    print(BATCH_SIZE, EPOCHS)
        
    all_file_train, all_haptic_wavs_train,all_audio_wavs_train, all_labs_train,all_file_test, all_haptic_wavs_test,all_audio_wavs_test, all_labs_test = load_data(data_hap_dir,data_aud_dir)#labは話者idに変更済み
    
    #pythonだと関数外の変数を関数内から書き換えることはglobalをつけないとだめらしいよ
    #print(len(all_wavs_train))
    aug_file, aug_haptic_wavs, aug_audio_wavs, aug_labs = data_augmentation(all_file_train, all_haptic_wavs_train,all_audio_wavs_train, all_labs_train)
    
    aug_hap_data = extract(aug_haptic_wavs)
    
    aug_aud_data = extract(aug_audio_wavs)

    aug_oneh = onehot_label(aug_labs)
    val_file, val_haptic_wavs, val_audio_wavs, val_labs = make_testdata(all_file_test, all_haptic_wavs_test,all_audio_wavs_test,all_labs_test)
    
    val_hap_data = extract(val_haptic_wavs)
    val_aud_data = extract(val_audio_wavs)

    val_oneh = onehot_label(val_labs)

    data_normalization(aug_hap_data, val_hap_data)
    data_normalization(aug_aud_data, val_aud_data)#音声

    test_corr=np.empty(0)
    test_pred=np.empty(0)

    print("Model training")
    for target_id in range(4):#test_dataから被験者名で取り出す

        valid_file=val_file[target_id]
        valid_hap_data=np.array(val_hap_data[target_id])#val_dataはテストデータ
        valid_aud_data=np.array(val_aud_data[target_id])
        valid_labs=np.array(val_labs[target_id])
        valid_oneh=np.array(val_oneh[target_id])


        train_file = []
        train_hap_data= []#ndarray化
        train_aud_data= []#ndarray化

        train_oneh= []
        train_labs= []

        train_file += aug_file[target_id]
        train_hap_data += aug_hap_data[target_id]#ndarray化
        train_aud_data += aug_aud_data[target_id]#ndarray化

        train_oneh += aug_oneh[target_id]
        train_labs += aug_labs[target_id]

        train_hap_data=np.array(train_hap_data)#ndarray化
        train_aud_data=np.array(train_aud_data)#ndarray化

        train_oneh=np.array(train_oneh)
        train_labs=np.array(train_labs)

    #for target_id in range(len(subjects)):#test_dataから被験者名で取り出す 

       # valid_file=val_file[target_id]
        #valid_data=np.array(val_data[target_id])#val_dataはテストデータ
      #  valid_labs=np.array(val_labs[target_id])
     #   valid_oneh=np.array(val_oneh[target_id])

     #   train_file=[]
      #  train_data=[]
      #  train_labs=[]
      #  train_oneh=[]

        #for sub_id in range(len(subjects)):#テストデータ以外の被験者のデータを訓練データとして使う
        #    if sub_id != target_id:
        #        train_file += aug_file[sub_id]                
         #       train_data += aug_data[sub_id]
         #       train_oneh += aug_oneh[sub_id]

       # train_data=np.array(train_data)#ndarray化
        #train_oneh=np.array(train_oneh)
        #train_labs=np.array(train_labs)


       # train_set=(train_file, train_data, train_oneh)#タプル化
       #valid_set=(valid_file, valid_data, valid_oneh)#タプル化
       # model=train_model(train_set, valid_set, subjects[target_id])
        #それぞれの被験者以外の訓練データでの学習がこれで進んでモデルがいくつかできる
        #test_set=(valid_file, valid_data, valid_labs)
        #corr, pred=predict(model, test_set, subjects[target_id])#テストデータ使って予測精度などを算出する
        #test_corr=np.append(test_corr, corr)#predは感情予測ラベル
        #test_pred=np.append(test_pred, pred)
    
        train_set=(train_file, train_hap_data,train_aud_data,train_oneh)#タプル化
        valid_set=(valid_file, valid_hap_data,valid_aud_data, valid_oneh)#タプル化
        model=train_model(train_set, valid_set, "Train "+str(target_id))
        #それぞれの被験者以外の訓練データでの学習がこれで進んでモデルがいくつかできる
        test_set=(valid_file, valid_hap_data,valid_aud_data,valid_labs)
        corr, pred=predict(model, test_set, "Train"+str(target_id))#テストデータ使って予測精度などを算出する
        test_corr=np.append(test_corr, corr)#predは感情予測ラベル
        test_pred=np.append(test_pred, pred)

    print("========================================")
    accuracy = accuracy_score(test_corr, test_pred)#全モデルに対する平均予測分類精度
    print('Overall'+' '+'accuracy: {:.2%}'.format(accuracy)) 
    print(confusion_matrix(test_corr, test_pred))
    print("\n")

            
def train_model(train_set, valid_set, target):
    tensorflow.random.set_seed(seed)

    train_file, train_hap_data,train_aud_data,train_oneh = train_set
    valid_file, valid_hap_data,valid_aud_data,valid_oneh = valid_set
    
    print("train_hap_data  : "+str(type(train_hap_data))+str(train_hap_data.shape))    
    print("train_aud_data  : "+str(type(train_aud_data))+str(train_aud_data.shape)) 
    print("train_label : "+str(type(train_oneh))+str(train_oneh.shape))  
    print("valid_hap_data  : "+str(type(valid_hap_data))+str(valid_hap_data.shape))    
    print("valid_aud_data  : "+str(type(valid_aud_data))+str(valid_aud_data.shape)) 
    print("valid_label : "+str(type(valid_oneh))+str(valid_oneh.shape))   

#    for fn in train_file: print("train:"+fn)
#    for fn in valid_file: print("valid:"+fn)

    model = Sequential()#空モデル生成　ここにaddしていくことによってどんどん積み重ねていく

#    model.add(Conv1D(8, kernel_size=512, strides=256, padding='same', activation='relu'))
    
    merged = np.concatenate(([train_hap_data, train_aud_data]),axis = 2)
    merged_valid = np.concatenate(([valid_hap_data, valid_aud_data]),axis = 2)

    ### Waveform input ################
    if False:#実行しないやーつ
        model.add(Conv1D(8, kernel_size=32, strides=16, padding='same', activation='relu'))
    
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Conv1D(8, kernel_size=16, strides=8, padding='same', activation='relu'))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Conv1D(8, kernel_size=8, strides=4, padding='same', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Conv1D(8, kernel_size=4, strides=2, padding='same', activation='relu'))
        
    ### STFT input ################
    
    # if True:
        
#        model.add(BatchNormalization())
#        model.add(Dropout(0.1))
        # model.add(Conv1D(8, kernel_size=1, strides=1, padding='same', activation='relu'))



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
    
    # model.add(BatchNormalization())
    # model.add(Dense(train_oneh.shape[1], activation='softmax'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    # model.add(Flatten()) #使用時エラー
    # model.add(Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu'))
    


    #    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    #    model.add(Conv1D(32, kernel_size=2048, strides=1024, padding='same', activation='relu'))
    #   model.add(Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu'))

#    model.add(BatchNormalization())
#    model.add(Dropout(0.1))

#    model.add(BatchNormalization())
#    model.add(Dropout(0.1))
#    model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv1D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
#    model.add(BatchNormalization())


#    model.add(BatchNormalization())

#    model.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
#    model.add(BatchNormalization())

    # model.add(Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(LSTM(LSTM_UNIT))
    
    model.add(BatchNormalization())
    # model.add(Flatten())
    model.add(Dense(31, activation='softmax'))

#    model.add(Dropout(0.1))
#    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
#    model.add(TimeDistributed(Dense(32, input_dim=1025, activation='relu')))

#    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
                             
#    model.add(TimeDistributed(Dense(128, input_dim=1025, activation='relu')))
#    model.add(LSTM(LSTM_UNIT, return_sequences=True))
    #####

    model.compile(#モデルの重み更新方法 loss,学習中に表示される性能指標を定義 根本的なものの定義
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    path="./Model_all/"+target+"/"+target+"_"+TRAIN_ID#Model_all/target/target_B8_E125

    if not os.path.exists(path):#ディレクトリがないときつまり最初 
        os.makedirs(path)#dirを作る

    mdfn="./"+target+"_"+TRAIN_ID+"_"+"model-{epoch:02d}.h5"#/target_B8_E125_model_epoch
    # checkpoint = ModelCheckpoint(filepath=os.path.join(path, mdfn), period=5)
    checkpoint = ModelCheckpoint(filepath=os.path.join(path, mdfn), save_best_only=True)
    #性能が向上したモデルだけ保存しているcheckpoint

    history = model.fit(merged, train_oneh, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(merged_valid, valid_oneh), callbacks=[checkpoint])
    model.summary()#訓練用データとその正解ラベル,検証用データも使う　これは過学習防止のため
    #学習中のモデルに対して精度を確認することで未知性能がどれくらいかを確認する
#verbose = 2で固定後でログをどれくらい詳しく確認するかの精度 callbacks チェックポイントを有効化
    sys.stdout = open(os.path.join(path, 'history.txt'), 'w')#出力制御線を標準入力からファイルに移す
    print(history.history)
    sys.stdout = sys.__stdout__#戻す

    mdfn="./Model_all/"+target+"/"+target+"_"+TRAIN_ID+"_final.h5"
    model.save(mdfn)#125エポック流した後の最終モデル

    return model
""" histroy .txtの中身 loss'
訓練データに対する損失値（小さいほどモデルがうまくフィットしている）

例：1エポック目は約1.68、125エポック目は約1.08に減少している。学習が進んで損失が下がっている。

'accuracy'
訓練データに対する正解率（精度）（大きいほど良い）

例：初めは約30.7%で、125エポック目には約55.5%まで上昇している。

'val_loss'
検証データに対する損失値

これも基本は小さいほど良い。

途中で上下してるけど、全体的には1.28〜1.33あたりで推移。

'val_accuracy'
検証データに対する精度

40〜55%の間で波打っている感じ。

過学習してるかどうかの判断材料になる"""
def load_wave(filepath):#librosaってなに
    print("loading: "+filepath)
    wav, sr = librosa.load(filepath, sr=SAMPLING_RATE)

    return wav

"""
def load_label(filename):

    label=-1
    for id in range(len(label_flag)):#label_flagはst_happy , st_sad,normalとかそのへんのラベル
#長さは4か5
        if "_"+label_flag[id]+"_" in filename:#ファイル名の中にこれらのラベル名が含まれているか否か
            if label == -1:
                label=id
            else:                
                print("label error")
                exit()

    return label

"""
def load_data(path_hap,path_aud):


    global MIN_DATA_LENGTH
    
    max_len=0
    min_len=sys.maxsize
    
    all_file_test=[[] for _ in range(4)]    
    all_hap_wavs_test=[[] for _ in range(4)]  
    all_aud_wavs_test=[[] for _ in range(4)]
    all_labs_test=[[] for _ in range(4)]

    all_file_train=[[] for _ in range(4)]    
    all_hap_wavs_train=[[] for _ in range(4)]
    all_aud_wavs_train=[[] for _ in range(4)]

    all_labs_train=[[] for _ in range(4)]
    
    #ここから変更
    #scale_all = {} #人ごとのscale倍
    scale_h = 0
    scale_a = 0

    rms_values_all_hap = []#人ごとのrms
    rms_values_all_aud = []#人ごとのrms

    for target in subjects:#まず何より元のwavのrmsを計算してscale倍する

        rms_values_hap = []

        for fn in glob.glob(path_hap+"*"+target+"*/*.wav"):#各々のutterance

            wav_h, sr_h = librosa.load(fn, sr=None)
            rms_h = np.sqrt(np.mean(wav_h**2))#アタランスごとのrms
            rms_values_hap.append(rms_h)

        mean_rms_hap = np.mean(rms_values_hap) #それらの100文の和の平均
        rms_values_all_hap.append(mean_rms_hap)#追加
    
    
    avg_bef_hap = sum(rms_values_all_hap[:8])/len(rms_values_all_hap[:8]) #旧データ人ごとの平均rmsの平均
    avg_aft_hap = sum(rms_values_all_hap[8:])/len(rms_values_all_hap[8:]) #新データ人ごとの平均rmsの平均

    scale_h = avg_aft_hap/avg_bef_hap
    
    #print(scale)

    for target in subjects:#まず何より元のwavのrmsを計算してscale倍する

        rms_values_aud = []

        for fn in glob.glob(path_aud+"*"+target+"*/*.wav"):#各々のutterance

            wav_a, sr_a = librosa.load(fn, sr=None)
            rms_a = np.sqrt(np.mean(wav_a**2))#アタランスごとのrms
            rms_values_aud.append(rms_a)

        mean_rms_aud = np.mean(rms_values_aud) #それらの100文の和の平均
        rms_values_all_aud.append(mean_rms_aud)#追加
    
    
    avg_bef_aud = sum(rms_values_all_aud[:8])/len(rms_values_all_aud[:8]) #旧データ人ごとの平均rmsの平均
    avg_aft_aud = sum(rms_values_all_aud[8:])/len(rms_values_all_aud[8:]) #新データ人ごとの平均rmsの平均

    scale_a = avg_aft_aud/avg_bef_aud
    
    print(scale_a)
    
#ここまで　
    #sub_file_train=[]
    #sub_wavs_train=[[] for _ in range(5)]
    #sub_labs_train=[[] for _ in range(5)]

    #sub_file_test=[]
    #sub_wavs_test=[]
    #sub_labs_test=[]

    for i,target in enumerate(subjects):#subfileの中にそのtarget名の人の中のwavファイルが入ってて

        ct = 0
        

        for fn in sorted(glob.glob(path_hap+"*"+target+"*/*.wav")):

            lab = i#話者id
            
            
           
              
            fn_aud = fn.replace(path_hap, path_aud)#ファイルパスをaudioに変更

            hap_wav = load_wave(fn)#wavファイル から波形をwavに読み込み格納する16khz
            aud_wav = load_wave(fn_aud)#音声

            if i < 8:
                hap_wav = hap_wav * scale_h
                aud_wav = aud_wav * scale_a

            if hap_wav.shape[0] != 0:#音声のサンプル数がwav.shape[0] sampling周波*時間
                if min_len > hap_wav.shape[0]:
                    min_len = hap_wav.shape[0]

                if max_len < hap_wav.shape[0]:#音声の時間をminからmaxで固定
                    max_len = hap_wav.shape[0]#wav[0]~wav[sample*時間] それぞれに振幅がはいっている
                
                j = ct//5

                if ct < 5:#01234,20,21,22,23,24..
                    all_file_test[j].append(fn)
                    all_hap_wavs_test[j].append(hap_wav)
                    all_aud_wavs_test[j].append(aud_wav)
                    all_labs_test[j].append(lab)
                
                    for s in range(4):
                        if s != j:

                            all_file_train[s].append(fn)
                            all_hap_wavs_train[s].append(hap_wav)
                            all_aud_wavs_train[s].append(aud_wav)
                            all_labs_train[s].append(lab)

                if 5 <= ct < 10:
                    all_file_test[j].append(fn)
                    all_hap_wavs_test[j].append(hap_wav)
                    all_aud_wavs_test[j].append(aud_wav)
                    all_labs_test[j].append(lab)
                    for s in range(4):
                        if s != j:
                            all_file_train[s].append(fn)
                            all_hap_wavs_train[s].append(hap_wav)
                            all_aud_wavs_train[s].append(aud_wav)
                            all_labs_train[s].append(lab)

                if 10 <= ct < 15:
                    all_file_test[j].append(fn)
                    all_hap_wavs_test[j].append(hap_wav)
                    all_aud_wavs_test[j].append(aud_wav)
                    all_labs_test[j].append(lab)

                    for s in range(4):
                        if s != j:
                            all_file_train[s].append(fn)
                            all_hap_wavs_train[s].append(hap_wav)
                            all_aud_wavs_train[s].append(aud_wav)
                            all_labs_train[s].append(lab)

                if 15 <= ct < 20:
                    all_file_test[j].append(fn)
                    all_hap_wavs_test[j].append(hap_wav)
                    all_aud_wavs_test[j].append(aud_wav)
                    all_labs_test[j].append(lab)

                    for s in range(4):
                        if s != j:
                            all_file_train[s].append(fn)
                            all_hap_wavs_train[s].append(hap_wav)
                            all_aud_wavs_train[s].append(aud_wav)
                            all_labs_train[s].append(lab)

            ct = (ct + 1)%20

    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len

    return all_file_train, all_hap_wavs_train,all_aud_wavs_train, all_labs_train,all_file_test,all_hap_wavs_test,all_aud_wavs_test,all_labs_test


def data_augmentation(all_file, all_haptic_wavs, all_audio_wavs, all_labs):

    global MIN_DATA_LENGTH
    
    print("Data augmentation")
    
    # n_aug=1
    n_aug=4
    amp_aug=0.0
    all_aug_file = []
    all_aug_haptic_wavs = []
    all_aug_audio_wavs = []

    all_aug_labs = []

    for sub_id in range(len(all_file)):#被験者名 0,1,2,3,4

        sub_aug_file=[]
        sub_aug_haptic_wavs=[]
        sub_aug_audio_wavs = []

        sub_aug_labs=[]

        for file_id in range(len(all_file[sub_id])):#その人の中のすべてのwavファイルid

            fn=all_file[sub_id][file_id]
            hap_wav = all_haptic_wavs[sub_id][file_id]
            aud_wav = all_audio_wavs[sub_id][file_id]

            lab=all_labs[sub_id][file_id]

            for n in range(n_aug):

                if n_aug == 1 :
                    START=0
                else:
                    START=int((aud_wav.shape[0]-MIN_DATA_LENGTH)*n/(n_aug-1))
                    # START=int((TOTAL-MIN_DATA_LENGTH)*np.random.random())
#幅MIN_DATA_LENGTHに統一して開始位置をずらしながら4パターン波形を作ってる

#                print(START,TOTAL,TOTAL-(START+MIN_DATA_LENGTH))
            
#                AMP=1.0+amp_aug-2.0*(amp_aug)*np.random.random()
                AMP=1.0+amp_aug-2.0*(amp_aug)*np.random.normal()
#amp_aug = 0 ならAMP = 1になり取り出した波形の振幅に変動性がない
#例えばamp_aug = 0.2ならAMP = 1.2 - 0.4N(0,1)になり波形の振幅に変動性が生まれる
#こうすると同じ感情でも振幅に依存している可能性のあるデータを様々な音量データを作ることによって
#感情を音量に依存させない 同じ感情でひとによって違うをなくしやすくする
                cut_haptic_wav = hap_wav[START:START+MIN_DATA_LENGTH]*AMP
                cut_audio_wav = aud_wav[START:START+MIN_DATA_LENGTH]*AMP

# AMP = 0.8 なら
                sub_aug_file.append(fn)
                sub_aug_haptic_wavs.append(cut_haptic_wav)
                sub_aug_audio_wavs.append(cut_audio_wav)

                sub_aug_labs.append(lab)

        all_aug_file.append(sub_aug_file)        
        all_aug_haptic_wavs.append(sub_aug_haptic_wavs)
        all_aug_audio_wavs.append(sub_aug_audio_wavs)

        all_aug_labs.append(sub_aug_labs)

#    print(all_aug_wavs[0].shape)
#    print(all_aug_labs[0])
#    print(len(all_aug_wavs))

    return all_aug_file, all_aug_haptic_wavs,all_aug_audio_wavs, all_aug_labs


def make_testdata(all_file, all_haptic_wavs, all_audio_wavs,all_labs):

    global MIN_DATA_LENGTH
    
    print("Make testdata")

    all_test_file = []
    all_test_hap_wavs = []
    all_test_aud_wavs = []

    all_test_labs = []

    for sub_id in range(len(all_file)):
        
        sub_test_file=[]
        sub_test_hap_wavs=[]
        sub_test_aud_wavs=[]
        sub_test_labs=[]
        
        for file_id in range(len(all_file[sub_id])):

            fn=all_file[sub_id][file_id]
            hap_wav=all_haptic_wavs[sub_id][file_id]
            aud_wav=all_audio_wavs[sub_id][file_id]

            lab=all_labs[sub_id][file_id]

            TOTAL=hap_wav.shape[0]

            if False:
                START=0
            elif True:
                START=int((TOTAL-MIN_DATA_LENGTH)/2)#データの中央長さDataLENGTH
            else:#中央じゃなくてランダムに選択するっていうのも案として考えてたっぽい
                START=int((TOTAL-MIN_DATA_LENGTH)*np.random.random())

#            print(START,TOTAL,TOTAL-(START+MIN_DATA_LENGTH))
            
            
            cut_hap_wav = hap_wav[START:START+MIN_DATA_LENGTH]#がこれ
            cut_aud_wav = aud_wav[START:START+MIN_DATA_LENGTH]#がこれ

            sub_test_file.append(fn)                
            sub_test_hap_wavs.append(cut_hap_wav)
            sub_test_aud_wavs.append(cut_aud_wav)

            sub_test_labs.append(lab)

        all_test_file.append(sub_test_file)
        all_test_hap_wavs.append(sub_test_hap_wavs)
        all_test_aud_wavs.append(sub_test_aud_wavs)

        all_test_labs.append(sub_test_labs)
        
    return all_test_file,  all_test_hap_wavs, all_test_aud_wavs, all_test_labs 


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
#なんとlibrosaくんはmfcc(波形,サンプリングレート,MFMCの中で利用する特徴数,win_length = 1フレーム
#長,フレームのずらし幅)を指定することでこれだけでmfccを抽出してくれるらしい
#サンプリングレートはこの波形は16000例　でサンプリングされた波形だよって教えてる
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


def onehot_label(all_labs):#[[1,2,3,1,3,],[1,2,2,1,1],[1,2,2,2,2,1] ]から[[[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0],[]],[[0,1,0,0],[]]]みたいにしている

    #dim = len(label_type)
    dim = len(subjects)
    all_ones=[]
    for sub_labs in all_labs:
        sub_ones=[]
        for label in sub_labs:
            onehot = np.identity(dim)[label]#dim = nならlabel(１次元)をn次元のone hot bekutoruにしている3-> [0,0,0,1,0,0,0,00,0,]
            sub_ones.append(onehot)
        all_ones.append(sub_ones)

    return all_ones

def data_normalization(aug_data, val_data):
    print(type(aug_data))
    print(len(aug_data))
    print(np.array(aug_data).shape)
    for i, sub in enumerate(aug_data):
        print(f"aug_data[{i}] type: {type(sub)}, len: {len(sub)}")

    mean = np.mean(np.array(aug_data), axis=(0,1,2))#被験者,発話,時間フレーム全て合わせてそのなかでのmfccの各次元での平均ベクトルを撮っている
    # mean = np.mean(np.array(aug_data), axis=(0,1))


    for sub_data in aug_data:#[1,100000,2,-333] こんな感じだと勾配方向が全く合わなくて学習がすすまないとか入力が巨大だと初期重みを小さくすると勾配がほぼ0になるだから正規化する
        for data in sub_data:
            data/=mean

    for sub_data in val_data:
        for data in sub_data:
            data/=mean


def predict(model, test_set, subject):

    test_file, test_hap_data,test_aud_data, test_label = test_set
    merged_test = np.concatenate(([test_hap_data, test_aud_data]),axis = 2)

#    print("ModelName: "+model_name)
#    model = keras.models.load_model(model_name)

    prob = model.predict(merged_test, batch_size=50)
    #各テスト入力データに対して予測値(各クラスの確率)を出す[angry:0.9,sad:0.09] みたいな
    

    for p in range(prob.shape[0]):#感情ラベル数
        print("["+str(p+1)+"]:"+test_file[p])#wavファイル名前 (入力)

        for l in range(len(subjects)):#それぞれのクラスの%
            print(subjects[l]+": "+str(f'{(prob[p,l]*100):.02f}')+"%   ",end="")
            # print(label_type[l] + f": {(prob[p, l] * 100):.02f}%   ", end="")
            # print(label_type[l] + ": {:.02f}%   ".format(prob[p, l] * 100), end="")
        print("\n")

    pred_label = np.argmax(prob, axis=1)#予測ラベルはprobの中で最大の値を取るもの
    accuracy = accuracy_score(test_label, pred_label)#test_label,pred_labelを勝手に累計して精度を算出してくれる
    print(subject+' '+'accuracy: {:.2%}'.format(accuracy)) 
    print(confusion_matrix(test_label, pred_label))
    print("\n")

    return test_label, pred_label


if __name__ == "__main__":
    main()
