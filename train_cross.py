# coding: UTF-8

import os
import glob
import numpy as np
import sys
import tensorflow
import keras
#from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input,LSTM, Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten,Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed
#from tensorflow.keras.layers.Permute import Permute
import librosa
import librosa.display

from tensorflow.keras.callbacks import ModelCheckpoint
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
LSTM_UNIT=8

#ver2
N_MFCC_a= 13#音声用の設定
WINDOW_SIZE_a = 512
HOP_LENGTH_a = 256



BATCH_SIZE=int(sys.argv[1])#ここでコマンドラインで指定したバッチサイズ(1度に使用するデータ)
EPOCHS=int(sys.argv[2])#データをすべて使用するのを何回繰り返すか　コマンドライン
emo = 5
seed = 0

TRAIN_ID = "B"+str(BATCH_SIZE)+"_E"+str(EPOCHS)

data_hap_dir = "../../new_data/haptic/" 
data_aud_dir = "../../new_data/audio/" 
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
        
    all_file, all_haptic_wavs, all_audio_wavs,all_labs = load_data(data_hap_dir,data_aud_dir)#ここまで理解した
    #dir以下の音声触覚ファイル名,波形,感情ラベル
    #pythonだと関数外の変数を関数内から書き換えることはglobalをつけないとだめらしいよ
    aug_file, aug_haptic_wavs,aug_audio_wavs,aug_labs = data_augmentation(all_file, all_haptic_wavs,all_audio_wavs, all_labs)
    aug_hap_data = extract(aug_haptic_wavs)#MFCC変換　
    aug_aud_data = extract(aug_audio_wavs)#ver1
    #ver 2 aug_aud_data = extract_a(aug_audio_wavs)#音声MFCC
    aug_oneh = onehot_label(aug_labs)

    val_file, val_haptic_wavs,val_audio_wavs,val_labs = make_testdata(all_file, all_haptic_wavs,all_audio_wavs, all_labs)
    #データ拡張していないもともとのデータの中央部分(無音とかなく情報が詰まった部分)をテストに使う
    val_hap_data = extract(val_haptic_wavs)#触覚テスト同じように
    #ver 2 val_aud_data = extract_a(val_audio_wavs)#音声テスト

    val_aud_data = extract(val_audio_wavs)#ver1

    val_oneh = onehot_label(val_labs)#同じように

    data_normalization(aug_hap_data, val_hap_data)#触覚正規化
    data_normalization(aug_aud_data, val_aud_data)#音声

    test_corr=np.empty(0)
    test_pred=np.empty(0)

    print("Model training")

    for target_id in range(len(subjects)):#test_dataから被験者名で取り出す 

        valid_file=val_file[target_id]
        valid_hap_data=np.array(val_hap_data[target_id])#触覚
        valid_aud_data=np.array(val_aud_data[target_id])#音声

        valid_labs=np.array(val_labs[target_id])
        valid_oneh=np.array(val_oneh[target_id])

        train_file=[]

        train_hap_data=[]
        train_aud_data=[]

        train_labs=[]
        train_oneh=[]

        for sub_id in range(len(subjects)):#テストデータ以外の被験者のデータを訓練データとして使う
            if sub_id != target_id:
                train_file += aug_file[sub_id]                
                train_hap_data += aug_hap_data[sub_id]#触覚
                train_aud_data += aug_aud_data[sub_id]#音声
                train_labs += aug_labs[sub_id]
                train_oneh += aug_oneh[sub_id]

        train_hap_data=np.array(train_hap_data)#ndarray化 触覚
        train_aud_data=np.array(train_aud_data)#音声
        train_oneh=np.array(train_oneh)
        train_labs=np.array(train_labs)


        train_set=(train_file, train_hap_data,train_aud_data,train_oneh)#タプル化 音声触覚
        valid_set=(valid_file, valid_hap_data,valid_aud_data, valid_oneh)#タプル化 音声触覚
        model=train_model(train_set, valid_set, subjects[target_id])
        #それぞれの被験者以外の訓練データでの学習がこれで進んでモデルがいくつかできる

        test_set=(valid_file, valid_hap_data,valid_aud_data, valid_labs)#音声触覚　タプル化

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

    train_file, train_hap_data,train_aud_data, train_oneh = train_set
    valid_file, valid_hap_data,valid_aud_data, valid_oneh = valid_set
    
    print("train_hap_data  : "+str(type(train_hap_data))+str(train_hap_data.shape))
    print("train_aud_data  : "+str(type(train_aud_data))+str(train_aud_data.shape))    

    print("train_label : "+str(type(train_oneh))+str(train_oneh.shape))  
    print("valid_hap_data  : "+str(type(valid_hap_data))+str(valid_hap_data.shape))
    print("valid_aud_data  : "+str(type(valid_aud_data))+str(valid_aud_data.shape))

    print("valid_label : "+str(type(valid_oneh))+str(valid_oneh.shape))   


#触覚
    hap_input = Input(shape=(train_hap_data.shape[1], train_hap_data.shape[2]))
    x1 = BatchNormalization()(hap_input)
    x1 = Dropout(0.1)(x1)
    x1 = LSTM(LSTM_UNIT)(x1)
    x1 = BatchNormalization()(x1)

# ---- 音声入力 ----
    aud_input = Input(shape=(train_aud_data.shape[1], train_aud_data.shape[2]))
    x2 = BatchNormalization()(aud_input)
    x2 = Dropout(0.1)(x2)
    x2 = LSTM(LSTM_UNIT)(x2)
    x2 = BatchNormalization()(x2)

    # ---- 特徴の統合 ----
    merged = Concatenate()([x1, x2])

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
        
    output = Dense(train_oneh.shape[1], activation='softmax')(merged)#クロスデータの出力層の定義
    
    model = Model(inputs=[hap_input,aud_input],outputs=output)

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

    history = model.fit([train_hap_data,train_aud_data], train_oneh, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=([valid_hap_data,valid_aud_data], valid_oneh), callbacks=[checkpoint])
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


def load_data(path_hap,path_aud):

    global MIN_DATA_LENGTH
    
    max_len=0
    min_len=sys.maxsize
    
    all_file=[]    
    all_hap_wavs=[]
    all_aud_wavs=[]
    all_labs=[]
    
    scale_h = 0
    scale_a = 0#audio hapticそれぞれの補正倍率

    rms_values_all_hap = []#人ごとのrms触覚
    rms_values_all_aud = []#人ごとのrms音声

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
    print(scale_h)

    for i,target in enumerate(subjects):#subfileの中にそのtarget名の人の中のwavファイルが入ってて

        sub_file=[]
        sub_hap_wavs=[]
        sub_aud_wavs=[]
        sub_labs=[]

        for fn in glob.glob(path_hap+"*"+target+"*/*.wav"):#指定されたパターンにマッチする
#ファイル名をすべてリストで返す　今pathはnewdata/haptic/
# subjects = ["aohara", "hasegawaT", "hodotsuka", "horio", "miyake", "mizuno", "yamadaM", "yamashita"]からtargetは取り出されていて,target名を含むフォルダの中のマッチするすべてのwavファイル
            lab = load_label(fn)#ファイル名の中のst_happy st_Sadなどを数値に変換　01234などの
            
            if lab != -1:#感情ラベル名に一致するファイルが一つもないとかつまりst_embaraとか以外

                fn_aud = fn.replace(path_hap, path_aud)#ファイルパスをaudioに変更


                hap_wav = load_wave(fn)#触覚
                aud_wav = load_wave(fn_aud)#音声
                
                if i < 8:
                    hap_wav = hap_wav * scale_h
                    aud_wav = aud_wav * scale_a
                    
                if hap_wav.shape[0] != 0:#音声のサンプル数がwav.shape[0] sampling周波*時間
                    if min_len > hap_wav.shape[0]:
                        min_len = hap_wav.shape[0]

                    if max_len < hap_wav.shape[0]:#音声の時間をminからmaxで固定
                        max_len = hap_wav.shape[0]#wav[0]~wav[sample*時間] それぞれに振幅がはいっている

                    sub_file.append(fn)
                    sub_hap_wavs.append(hap_wav)
                    sub_aud_wavs.append(aud_wav)

                    sub_labs.append(lab)
                    
        all_file.append(sub_file)#all_fileには[("被験者名",subfile),[("被験者名",subfile)・・・]]がはいってるってことか
        all_hap_wavs.append(sub_hap_wavs)
        all_aud_wavs.append(sub_aud_wavs)

        all_labs.append(sub_labs)

    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len

    return all_file, all_hap_wavs, all_aud_wavs,all_labs, 


def data_augmentation(all_file, all_haptic_wavs,all_audio_wavs, all_labs):

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


def make_testdata(all_file, all_haptic_wavs,all_audio_wavs, all_labs):

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
        
        
    return all_test_file, all_test_hap_wavs, all_test_aud_wavs,all_test_labs 


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

def extract_a(all_wavs):

    print("Feature extraction")
    all_data=[]
    for sub_wavs in all_wavs:
        sub_data=[]
        for wav in sub_wavs:
#            print(".",end="")
            if True:
                ELEMENT_COUNT=N_MFCC_a
                mfcc = librosa.feature.mfcc(y=wav, sr=SAMPLING_RATE, n_mfcc=N_MFCC_a, win_length=WINDOW_SIZE_a, hop_length=HOP_LENGTH_a)
#なんとlibrosaくんはmfcc(波形,サンプリングレート,MFMCの中で利用する特徴数,win_length = 1フレーム
#長,フレームのずらし幅)を指定することでこれだけでmfccを抽出してくれるらしい
#サンプリングレートはこの波形は16000例　でサンプリングされた波形だよって教えてる
                data = mfcc.transpose()#転置
            elif False:#実際には実行されない
                ELEMENT_COUNT=int(WINDOW_SIZE_a/2)+1
                stft = librosa.stft(wav, n_fft=WINDOW_SIZE_a, hop_length=HOP_LENGTH_a)
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

#    print("ModelName: "+model_name)


    prob = model.predict([test_hap_data,test_aud_data],batch_size=50)
    #各テスト入力データに対して予測値(各クラスの確率)を出す[angry:0.9,sad:0.09] みたいな
    for p in range(prob.shape[0]):#感情ラベル数
        print("["+str(p+1)+"]:"+test_file[p])#wavファイル名前 (入力)

        for l in range(len(label_type)):#それぞれのクラスの%
            print(label_type[l]+": "+str(f'{(prob[p,l]*100):.02f}')+"%   ",end="")
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
