N,X,Y = map(int,input().split(" "))

given = [int(n) for n in input().split(" ")]


cor = []

for i in range(N):
  corlists = []
  corlists.append(X*given[i])
  corlists.append(Y*given[i])
  cor.append(corlists)

corf = sorted(cor,key = lambda x:x[0])
corl = sorted(cor,key = lambda x:x[1])

f = corf[len(corf)-1][0]
l = corl[0][1]





dif = Y - X



if f <= l:
  kosuu = 0
  for i in range(N):
    number = (l - X * given[i]) // dif
    kosuu += number
  print(kosuu)
else:
  print(-1)
