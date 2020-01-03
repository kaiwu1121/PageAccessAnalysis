import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from scipy import stats
from pylab import *
from matplotlib.pyplot import figure


fileName = "bt.D.x.t25.p2800000.f0" #"cg.D.48.f0"
output = 'out.bt.D.x.t25.p2800000.f0'
base = 1000
accessLength = 0
top = 400000
count_s = 0
count_f = 0
count_m = 0
interval_buffer = []
fastmem = {}

def readData():
        lineList = list()
        with open(fileName) as f:
		row = 0
		last  = 0
		curr = 0	
                for line in f:
			data = line.split()
			if len(data) > 1:
	#			print data
				data[-1] = data[-1].strip()
	#			print data
				row = row + 1
				curr = int(data[0])
				if row == 1:
					last = curr + base
	#			print curr - last
				if last < curr:
					print -1111
					lineList.append(int(-1111))
					lineList.append(int(data[1]))
					last = curr + base
				else:
					lineList.append(int(data[1]))			

        accessLength = len(lineList)
        print('Total app page accesses: {}'.format(accessLength))
#	of = open(output, "w")
##	of.writelines(lineList)
	with open(output, 'w') as f:
    		for item in lineList:
        		f.write("%d\n" % item)
#	with open(output, "w") as of:
#    		of.write(str(lineList))
	f.close()
        return lineList



def loadData():
	lineList = list()
	with open(output) as f:
		for line in f:
			lineList.append(int(line))
	return lineList


def sort_in_interval(List):
	global top
	global count_s
	global count_f
	global count_m
	global fastmem
	print top, base, count_s, count_f, count_m
	i = 0
	result = {}
	while i < len(List):
		line = List[i]
		if line in fastmem:
			count_s += 1
		else:
			count_f += 1
		if line in result:
			result[line] += 1
		else:
			result[line] = 1
		i = i + 1
#	print(sorted(result.items(), key=lambda k:k[1], reverse=True)[:k])
	fastmem_new = dict(sorted(result.items(), key=lambda k:k[1], reverse=True)[:top])
	count_m += len(set(fastmem_new)) - len(set(fastmem) & set(fastmem_new))
	fastmem = fastmem_new
#	print fastmem
#	print len(fastmem)
#	fastmem = {}
#	print len(fastmem)
#	fastmem = sorted(result.items(), key=lambda k:k[1], reverse=True)[:k]
#	print len(fastmem)

def sim_run(access):
	i = 0
	s = 0
#	print accessLength, len(access)
	buff = list()
#	dram = list()
	fastmem = {}
	while i < len(access): 
#		print access[i]
    		if access[i] == -1111: 
#			print access[i]
			s += 1	
#			print(len(set(buff) & set(dram)))
			sort_in_interval(buff)
			#dram = buff
			buff = []
		else:
			buff.append(access[i])	
    		i += 1
	print s

def main():
        print 1111

        accessData = readData()
	data = loadData()
	sim_run(data)
	print count_s, count_f, count_s + count_f, count_m
#	sim_run(accessData)

if __name__ == '__main__':
    main()
