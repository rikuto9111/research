N,M,K = map(int,input().split(" "))


H = input().split(" ")
Hl = []


  
  
for i in range(N):
  Hl.append(int(H[i]))
  
B = input().split(" ")
Bl = []

for m in range(M):
  Bl.append([int(B[m]),0])

Hs = sorted(Hl)
Bs = sorted(Bl,key = lambda x:x[0])


def search(x):
  low = 0
  high = M-1
  
  
  
  if Bs[low][0] >= x:#左はじ
    return low
  if Bs[high][0] < x:#右端
    return -1
  
  while low < high + 1:
    print(low)
    mid = (low + high)//2
    
    if x == Bs[mid][0]:
      return mid
    
    elif x > Bs[mid][0]:
      low = mid
    else:
      high = mid
  

  while high <= M -1:
    if Bs[high][1] == 1:
      high += 1
    else:
      return -1
  
  if high == M:
    return -1
  
  

  return high

print(Bs)
ans = 0
for i in range(N):
  t = search(Hs[i])
  if t != -1:
    Bs[t][1] = 1
    ans += 1

print(ans)
