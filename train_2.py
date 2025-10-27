def load_data(path):

    global MIN_DATA_LENGTH
    
    max_len=0
    min_len=sys.maxsize
    
    all_file=[]    
    all_wavs=[]
    all_labs=[]
    
    #ここから変更
    scale_all = {} #人ごとのscale倍
    rms_values_all = []#人ごとのrms

    for target in subjects:#まず何より元のwavのrmsを計算してscale倍する

        rms_values = []

        for fn in glob.glob(path+"*"+target+"*/*.wav"):#各々のutterance

            wav, sr = librosa.load(fn, sr=None)
            rms = np.sqrt(np.mean(wav**2))#アタランスごとのrms
            rms_values.append(rms)

        mean_rms = np.mean(rms_values) #それらの100文の和の平均
        rms_values_all.append(mean_rms)#追加
    
    avg = sum(rms_values_all)/len(rms_values_all) #人ごとの平均rmsの平均

    for i,target in enumerate(subjects):#被験者
        scale_all[target] = (avg/rms_values_all[i])#それぞれの人の拡大すべきscaleを求める
    
    
#ここまで　

    for target in subjects:#subfileの中にそのtarget名の人の中のwavファイルが入ってて

        sub_file=[]
        sub_wavs=[]
        sub_labs=[]

        for fn in glob.glob(path+"*"+target+"*/*.wav"):#指定されたパターンにマッチする
#ファイル名をすべてリストで返す　今pathはnewdata/haptic/
            lab = load_label(fn)
            
            if lab != -1:#感情ラベル名に一致するファイルが一つもないとかつまりst_embaraとか以外
                
                # 変更箇所1. 
                wav = load_wave(fn)#wavファイル から波形をwavに読み込み格納する16khz
                wav = wav*scale_all[target] #補正倍の正規化 ndarrayのbroadcast

                if wav.shape[0] != 0:#音声のサンプル数がwav.shape[0] sampling周波*時間
                    if min_len > wav.shape[0]:
                        min_len = wav.shape[0]

                    if max_len < wav.shape[0]:#音声の時間をminからmaxで固定
                        max_len = wav.shape[0]#wav[0]~wav[sample*時間] それぞれに振幅がはいっている

                    sub_file.append(fn)
                    sub_wavs.append(wav)
                    sub_labs.append(lab)
                    
        all_file.append(sub_file)#all_fileには[("被験者名",subfile),[("被験者名",subfile)・・・]]がはいってるってことか
        all_wavs.append(sub_wavs)
        all_labs.append(sub_labs)

def load_wave(filepath):
    print("loading: "+filepath)
    wav, sr = librosa.load(filepath, sr=SAMPLING_RATE)

    return wav

    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len
