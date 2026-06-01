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
N_MFCC=8
#HOP_LENGTH=256

ELEMENT_COUNT=int(WINDOW_SIZE/2)+1
#ELEMENT_COUNT=1
ELEMENT_COUNT=N_MFCC
#ELEMENT_COUNT=1025
#LSTM_UNIT=8
LSTM_UNIT=32

BATCH_SIZE=int(sys.argv[1])#ここでコマンドラインで指定したバッチサイズ(1度に使用するデータ)
EPOCHS=int(sys.argv[2])#データをすべて使用するのを何回繰り返すか　コマンドライン
emo = 5
seed = 0

TRAIN_ID = "B"+str(BATCH_SIZE)+"_E"+str(EPOCHS)

data_dir = "../new_data/audio/" 
#subjects = ["aohara", "hasegawaT", "hodotsuka", "horio", "miyake", "mizuno", "yamadaM", "yamashita","iwasakeisho","hattori_naoki"]
subjects = ["aohara_wav_output_org", "hasegawa_wav_output_org", "hodotsuka_wav_output_org", "horio_wav_output_org", "miyake_wav_output_org", "mizuno_wav_output_org","yamadaM_wav_output_org","yamashita_wav_output_org","newiwa_wav_output_org","hattori_wav_output","miyasita_wav_output_org","nakagawa_wav_output_org","newhatori_wav_output_org","takase_wav_output_org","imamura_wav_output_org","kariya_wav_output_org","kimura_wav_output_org","takagi_wav_output_org","atusi_wav_output_org", "kasiwagi_wav_output_org", "kunda_wav_output_org", "newyamasita_wav_output_org", "satou_wav_output_org", "yasunaga_wav_output_org","asano_wav_output_org","kamiya_wav_output_org","sinohara_wav_output_org","isii_wav_output_org","onda_wav_output_org","tamaki_wav_output_org","iida_wav_output_org"]

#subjects = ["aohara_wav_output", "hasegawa_wav_output", "hodotsuka_wav_output"]

if emo == 4:#4クラス分類の場合
    label_type=['normal', 'angry', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-hap', 'st-sad']
elif emo == 5:#5クラス分類の場合
    label_type=['normal', 'angry', 'bash', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-bas', 'st-hap', 'st-sad']
else:
    label_type=['normal', 'angry', 'bash', 'happy', 'sad']
    label_flag=['normal', 'st-ang', 'st-bas', 'st-hap', 'st-sad']
    

def main():

    print(BATCH_SIZE, EPOCHS)
        
    all_file, all_wavs, all_labs = load_data(data_dir)
    #dir以下の音声触覚ファイル名,波形,感情ラベル
    
    aug_file, aug_wavs, aug_labs = data_augmentation(all_file, all_wavs, all_labs)
    aug_data = extract(aug_wavs)#MFCC変換　[[data1,data2,data3, ],[data1,data2, ]]
    
    aug_oneh = onehot_label(aug_labs)#感情ラベル1,2,3 ->[0,1,0,0] , [0,0,1,0], [0,0,0,1] 変換

    val_file, val_wavs, val_labs = make_testdata(all_file, all_wavs, all_labs)
    #データ拡張していないもともとのデータの中央部分(無音とかなく情報が詰まった部分)をテストに使う
    val_data = extract(val_wavs)
    val_oneh = onehot_label(val_labs)

    data_normalization(aug_data, val_data)
    
    test_corr=np.empty(0)
    test_pred=np.empty(0)

    print("Model training")

    for target_id in range(len(subjects)):#test_dataから被験者名で取り出す 

        valid_file=val_file[target_id]
        valid_data=np.array(val_data[target_id])#val_dataはテストデータ
        valid_labs=np.array(val_labs[target_id])
        valid_oneh=np.array(val_oneh[target_id])

        train_file=[]
        train_data=[]
        train_labs=[]
        train_oneh=[]

        for sub_id in range(len(subjects)):#テストデータ以外の被験者のデータを訓練データとして使う
            if sub_id != target_id:
                train_file += aug_file[sub_id]                
                train_data += aug_data[sub_id]
                train_labs += aug_labs[sub_id]
                train_oneh += aug_oneh[sub_id]

        train_data=np.array(train_data)#ndarray化
        train_oneh=np.array(train_oneh)
        train_labs=np.array(train_labs)


        train_set=(train_file, train_data, train_oneh)#タプル化
        valid_set=(valid_file, valid_data, valid_oneh)#タプル化
        model=train_model(train_set, valid_set, subjects[target_id])

        test_set=(valid_file, valid_data, valid_labs)
        corr, pred=predict(model, test_set, subjects[target_id])#テストデータ使って予測精度などを算出する
        test_corr=np.append(test_corr, corr)#predは感情予測ラベル
        test_pred=np.append(test_pred, pred)


    print("========================================")
    accuracy = accuracy_score(test_corr, test_pred)#全モデルに対する平均予測分類精度
    print('Overall'+' '+'accuracy: {:.2%}'.format(accuracy)) 
    print(confusion_matrix(test_corr, test_pred))
    print("\n")

            
def train_model(train_set, valid_set, target):
    tensorflow.random.set_seed(seed)

    train_file, train_data, train_oneh = train_set
    valid_file, valid_data, valid_oneh = valid_set
    
    print("train_data  : "+str(type(train_data))+str(train_data.shape))    
    print("train_label : "+str(type(train_oneh))+str(train_oneh.shape))  
    print("valid_data  : "+str(type(valid_data))+str(valid_data.shape))    
    print("valid_label : "+str(type(valid_oneh))+str(valid_oneh.shape))   

#    for fn in train_file: print("train:"+fn)
#    for fn in valid_file: print("valid:"+fn)

    model = Sequential()#空モデル生成　ここにaddしていくことによってどんどん積み重ねていく

#    model.add(Conv1D(8, kernel_size=512, strides=256, padding='same', activation='relu'))
    

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


    # ### MFCC input ################
    
    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(8, kernel_size=1, strides=1, padding='same', activation='relu'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())#バッチ正規化
    model.add(Dropout(0.1))#ドロップアウト層
    model.add(LSTM(LSTM_UNIT))#LSTM層
    
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

    model.add(BatchNormalization())#バッチ正規化
    model.add(Dense(train_oneh.shape[1], activation='softmax'))#全結合層

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

    mdfn="./"+target+"_"+TRAIN_ID+"_"+"model-{epoch:02d}.h5"

    checkpoint = ModelCheckpoint(filepath=os.path.join(path, mdfn), save_best_only=True)

    history = model.fit(train_data, train_oneh, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(valid_data, valid_oneh), callbacks=[checkpoint])
    model.summary()#訓練用データとその正解ラベル,検証用データも使う　これは過学習防止のため
    #学習中のモデルに対して精度を確認することで未知性能がどれくらいかを確認する

    sys.stdout = open(os.path.join(path, 'history.txt'), 'w')
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


def load_label(filename):

    label=-1
    for id in range(len(label_flag)):

        if "_"+label_flag[id]+"_" in filename:
            if label == -1:
                label=id
            else:                
                print("label error")
                exit()

    return label


def load_data(path):


    global MIN_DATA_LENGTH
    
    max_len=0
    min_len=sys.maxsize
    
    all_file=[]    
    all_wavs=[]
    all_labs=[]
    
    #ここから変更
    #scale_all = {} #人ごとのscale倍
    scale = 0

    rms_values_all = []#人ごとのrms

    for target in subjects:#まず何より元のwavのrmsを計算してscale倍する

        rms_values = []

        for fn in glob.glob(path+"*"+target+"*/*.wav"):#各々のutterance

            wav, sr = librosa.load(fn, sr=None)
            rms = np.sqrt(np.mean(wav**2))#utteranceごとのrms
            rms_values.append(rms)

        mean_rms = np.mean(rms_values) #それらの100文の和の平均
        rms_values_all.append(mean_rms)
    
    
    avg_bef = sum(rms_values_all[:8])/len(rms_values_all[:8]) #旧データ人ごとの平均rmsの平均
    avg_aft = sum(rms_values_all[8:])/len(rms_values_all[8:]) #新データ人ごとの平均rmsの平均

    scale = avg_aft/avg_bef
    
    print(scale)
    
#ここまで　

    for i,target in enumerate(subjects):

        sub_file=[]
        sub_wavs=[]
        sub_labs=[]

        for fn in glob.glob(path+"*"+target+"*/*.wav"):#指定されたパターンにマッチするファイル名をすべてリストで返す

            lab = load_label(fn)
            
            if lab != -1:
                
                # 変更箇所1. 
                wav = load_wave(fn)
                if i < 8:
                    wav = wav * scale#前収録と本収録の環境差を揃える
                

                if wav.shape[0] != 0:
                    if min_len > wav.shape[0]:
                        min_len = wav.shape[0]

                    if max_len < wav.shape[0]:#音声の時間をminからmaxで固定
                        max_len = wav.shape[0]

                    sub_file.append(fn)
                    sub_wavs.append(wav)
                    sub_labs.append(lab)
                    
        all_file.append(sub_file)#all_fileには[("被験者名",subfile),[("被験者名",subfile)・・・]]
        all_wavs.append(sub_wavs)
        all_labs.append(sub_labs)

    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len

    return all_file, all_wavs, all_labs


def data_augmentation(all_file, all_wavs, all_labs):

    global MIN_DATA_LENGTH
    
    print("Data augmentation")
    
    # n_aug=1
    n_aug=4
    amp_aug=0.0
    all_aug_file = []
    all_aug_wavs = []
    all_aug_labs = []

    for sub_id in range(len(all_file)):

        sub_aug_file=[]
        sub_aug_wavs=[]
        sub_aug_labs=[]

        for file_id in range(len(all_file[sub_id])):

            fn=all_file[sub_id][file_id]
            wav=all_wavs[sub_id][file_id]
            lab=all_labs[sub_id][file_id]

            for n in range(n_aug):

                if n_aug == 1 :
                    START=0
                else:
                    START=int((wav.shape[0]-MIN_DATA_LENGTH)*n/(n_aug-1))
                    
                AMP=1.0+amp_aug-2.0*(amp_aug)*np.random.normal()
#amp_aug = 0 ならAMP = 1になり取り出した波形の振幅に変動性がない
#振幅に変動性を持たせる
                cut_wav = wav[START:START+MIN_DATA_LENGTH]*AMP

                sub_aug_file.append(fn)
                sub_aug_wavs.append(cut_wav)
                sub_aug_labs.append(lab)

        all_aug_file.append(sub_aug_file)        
        all_aug_wavs.append(sub_aug_wavs)
        all_aug_labs.append(sub_aug_labs)


    return all_aug_file, all_aug_wavs, all_aug_labs


def make_testdata(all_file, all_wavs, all_labs):

    global MIN_DATA_LENGTH
    
    print("Make testdata")

    all_test_file = []
    all_test_wavs = []
    all_test_labs = []

    for sub_id in range(len(all_file)):
        
        sub_test_file=[]
        sub_test_wavs=[]
        sub_test_labs=[]
        
        for file_id in range(len(all_file[sub_id])):

            fn=all_file[sub_id][file_id]
            wav=all_wavs[sub_id][file_id]
            lab=all_labs[sub_id][file_id]

            TOTAL=wav.shape[0]

            if False:
                START=0
            elif True:
                START=int((TOTAL-MIN_DATA_LENGTH)/2)#データの中央長さDATA_LENGTH
            else:#中央じゃなくてランダムに選択するっていうのも案として考えてたっぽい
                START=int((TOTAL-MIN_DATA_LENGTH)*np.random.random())

#            
            cut_wav = wav[START:START+MIN_DATA_LENGTH]
            
            sub_test_file.append(fn)                
            sub_test_wavs.append(cut_wav)
            sub_test_labs.append(lab)

        all_test_file.append(sub_test_file)
        all_test_wavs.append(sub_test_wavs)
        all_test_labs.append(sub_test_labs)
        
    return all_test_file, all_test_wavs, all_test_labs 


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
#mfcc抽出
                data = mfcc.transpose()
            elif False:
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

    dim = len(label_type)

    all_ones=[]
    for sub_labs in all_labs:
        sub_ones=[]
        for label in sub_labs:
            onehot = np.identity(dim)[label]
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

    test_file, test_data, test_label = test_set


    prob = model.predict(test_data, batch_size=50)
    #各テスト入力データに対して予測値(各クラスの確率)を出す[angry:0.9,sad:0.09]
    for p in range(prob.shape[0]):#感情ラベル数
        print("["+str(p+1)+"]:"+test_file[p])#wavファイル名前

        for l in range(len(label_type)):
            print(label_type[l]+": "+str(f'{(prob[p,l]*100):.02f}')+"%   ",end="")
            # print(label_type[l] + f": {(prob[p, l] * 100):.02f}%   ", end="")
            # print(label_type[l] + ": {:.02f}%   ".format(prob[p, l] * 100), end="")
        print("\n")
        
    pred_label = np.argmax(prob, axis=1)#予測ラベルはprobの中での最大値
    accuracy = accuracy_score(test_label, pred_label)
    print(subject+' '+'accuracy: {:.2%}'.format(accuracy)) 
    print(confusion_matrix(test_label, pred_label))
    print("\n")

    return test_label, pred_label


if __name__ == "__main__":
    main()
