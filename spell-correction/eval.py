import nltk
anspath='./ans.txt'
resultpath='./result.txt'
ansfile=open(anspath,'r')
resultfile=open(resultpath,'r')

count=0
n=1000
for i in range(n):
    ansline=ansfile.readline().split('\t')[1]
    ansset=set(nltk.word_tokenize(ansline))
    resultline=resultfile.readline().split('\t')[1]
    resultset=set(nltk.word_tokenize(resultline))
    if ansset==resultset:
        count+=1
    else:
        diff_set = ansset - resultset
        print(ansset - resultset)
        print(resultset - ansset)
print("Accuracy is : %.2f%%" % (count*1.00/n* 100))
