# subsamples the electricity data file
# ARG1: input file
# ARG2: ourput file
# ARG3: subsampling step. Example: for 2hours subsampling, put 8 (as the original data sampling rate is every 15min)
import sys
fin = open(sys.argv[1],'rb')
fout = open(sys.argv[2],'wb')
i=0
step = int(sys.argv[3])
for line in fin:
    if i%step == 0:
        print(i)
        fout.write(line)
    i = i + 1
fin.close()
fout.close()
    
