S = [int(n) for n in input()]

seqback = 1
seqafter = 0


b = 0#ひとつ前

ans = 0

for i in range(len(S)):
  if i == 0:
    b = S[i]
  
  else:
    if b == S[i]:
      seqback += 1
      seqafter += 1
      
    elif S[i] == b+1:#連続ストップして次の数字が+1の値
      #seqback = 0#左の連続カウントリセット
      ans += 1
      seqafter += 1#右の連続カウントスタート
      
    elif S[i] != b+1:#次の数字が+1になっていない
      seqback = 0
      seqback += 1
      
    b = S[i]#過去の人にする
    if seqback == seqafter:
      ans += min(seqback,seqafter)
      seqback = seqafter#右にカウントしていたものを左カウントに持っていく
      seqafter = 0
