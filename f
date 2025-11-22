    b = S[i]#過去の人にする
  S = input().strip()
n = len(S)

ans = 0

i = 0
while i < n:
    # 左側（a の連続区間）
    a = S[i]
    lenA = 0
    while i < n and S[i] == a:
        lenA += 1
        i += 1

    # 右側（b = a+1 の連続区間）が続いているか？
    if i < n and int(S[i]) == (int(a) + 1) % 10:
        b = S[i]
        lenB = 0
        while i < n and S[i] == b:
            lenB += 1
            i += 1

        # 1122 文字列の個数
        ans += min(lenA, lenB)

    # 次のループへ（右側にならなかった場合はそのまま）
    # i はすでに進んでいるので何もしない

print(ans)
  if seqback == seqafter:
      ans += min(seqback,seqafter)
      seqback = seqafter#右にカウントしていたものを左カウントに持っていく
      seqafter = 0
