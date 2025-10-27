cor = 1.5
for i,target in enumerate(subjects):#indexも使う 8人目まで定数倍

        sub_file=[]
        sub_wavs=[]
        sub_labs=[]

        for fn in glob.glob(path+"*"+target+"*/*.wav"):

            lab = load_label(fn)
            
            if lab != -1:#感情ラベル名に一致するファイルが一つもないとかつまりst_embaraとか以外
                
                wav = load_wave(fn)

                if i < 8:#前データに対しての補正(1.5)
                    wav = wav*cor

                if wav.shape[0] != 0:
                    if min_len > wav.shape[0]:
                        min_len = wav.shape[0]

                    if max_len < wav.shape[0]:#音声の時間をminからmaxで固定
                        max_len = wav.shape[0]
                sub_file.append(fn)
                    sub_wavs.append(wav)
                    sub_labs.append(lab)
                    
        all_file.append(sub_file)#all_fileには[("被験者名",subfile),[("被験者名",subfile)・・・]]がはいってるってことか
        all_wavs.append(sub_wavs)
        all_labs.append(sub_labs)

    MAX_DATA_LENGTH=max_len
    MIN_DATA_LENGTH=min_len

    return all_file, all_wavs, all_labs, 
