import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from scipy import stats
from pylab import *
from matplotlib.pyplot import figure


fileName = "cg.D.48.f0.r10" #"cg.D.48.f0"
aceessLength = 0

def readData():
	lineList = list()
	with open(fileName) as f:
  		for line in f:
    			lineList.append(line)
	aceessLength = len(lineList)
	print('Total app page accesses: {}'.format(aceessLength))
	return lineList

def getUnique(arr):
	#u1, c1 = np.unique(arr, return_counts=True)
	#print c1
	u = np.unique(arr)
	print('# of unique pages: {}'.format(u.size))
	return u

def getOffset(arr):
	m = np.max(arr)
	n = np.min(arr) 
	print('max page number: {}'.format(m))
	print('min page number: {}'.format(n))
	print('offset: {}'.format(m - n))
	return m


def plot_arr(arr, name, y_ax):
#    plt.ylim(ymin = 0)
#    plt.set_ylim(ymin=0) 
    plt.plot(arr)
    # plt.title(name)
    plt.xlabel("Interval")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name + " " + y_ax + "T100"+".png")
    plt.close()

def plot_stacked(arr1, arr2):
	#x = np.arange(numBar)
	plt.bar(range(len(arr1)), arr1, label='non-dup',fc = 'y')
	plt.bar(range(len(arr1)), arr2, bottom=arr1, label='dup',fc = 'r')
	plt.legend()
	plt.xlabel("Interval")
	plt.savefig("duplicated100.png")
	plt.close()



def get_interval(raw_address, num_chunk):
    #raw_address = load_address_array(subfolder, address_txt)
    # plot_arr(raw_address, 'app_addr')

    print('len of original address: {}'.format(len(raw_address)))
#    raw_address = raw_address[0:(len(raw_address) // num_chunk * num_chunk)]
#    print('len of sliced address: {}'.format(len(raw_address)))

#    reshape_address = np.reshape(raw_address, (num_chunk, -1))
    reshape_address = np.array_split(raw_address, num_chunk)
    # res_list1 = [[] for x in range(num_chunk - 1)]
    # for i in range(num_chunk - 1):
    #     for j in range(i + 1, num_chunk):
    #         overlap = np.intersect1d(reshape_address[i], reshape_address[j])
    #         res_list1[i].append(len(overlap))

    overlap_res_list2 = []
    unique_res = []
#    for m in range(num_chunk - 1):
    for m in [0,20,40,60,80]:
	print "-------------"
	print "interval " + str(m)
	for i in range(1, num_chunk - 1 - m, 1):
        	j = m + i
        	overlap = np.intersect1d(reshape_address[m], reshape_address[j])
		u, c = np.unique(reshape_address[m], return_counts=True)
		unique_res.append(len(u))
		print len(overlap)
#	if m == num_chunk - 1:
#		overlap = 0
#	else:
#        	overlap_res_list2.append(len(overlap))
#    arr = overlap_res_list2
#    print len(arr), len(unique_res)
"""    print "duplicated pages:"
    for i in range(len(arr)):
	print arr[i]
    print "----------------------"
    print "unique pages:"
    for i in range(len(unique_res)):
	print unique_res[i]
"""
"""    barwidth = 0.25
    r1 = [x + barwidth for x in unique_res]
    r2 = [x + barwidth for x in arr]
    plt.bar(r1, unique_res, color='#7f6d5f', width=barwidth, edgecolor='white', label='unique pages')
    plt.bar(r2, arr, color='#557f2d', width=barwidth, edgecolor='white', label='duplicated pages')
    plt.xlabel('Intervals', fontweight='bold')
    plt.legend()
    plt.savefig("duplicated100.png")
    plt.close()
"""
    # for i in range(num_chunk - 1):
    #     arr.append(res_list1[i][0])
#    plot_stacked(unique_res, arr)	
#    plot_arr(arr, 'cg_D_48t', 'Duplicated pages')
#    plot_arr(unique_res, 'cg_D_48t', 'Num unique addresses')

    # with open('{}/address_overlap_len_{}.txt'.format(subfolder, num_chunk), 'w') as f:
    #     for item in res_list:
    #         f.write("%s\n" % item)

#    return overlap_res_list2

#def drawCDF(arr):
#	plt.subplot(221)
#	plt.plot()
#def drawPDF(arr):

def plot_cdf(arr, name, y_ax, x_max):
    x=np.arange(x_max)
#    x = [(x + 1) * 100 for x in range(len(arr))]
    plt.plot(x, arr)
    # plt.title(name)
    plt.xlabel("Top X frequent accessed pages")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name+y_ax)
    plt.close()


def plot_pdf(arr, name, y_ax, x_max):
    x=np.arange(x_max)
#    x = [(x + 1) * 100 for x in range(len(arr))]
    plt.plot(x, arr)
    # plt.title(name)
    plt.xlabel("Page number")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name+y_ax)
    plt.close()

def main():
	print 1111
#	maxPage=34357617469
#	pdf=[]
#	for i in range(34357617469)
#		pdf.append(0)
	
	accessData = readData()
	accessArr =  np.asarray(accessData, dtype=int)
	get_interval(accessArr, 100)
    # get unique application address
    	u, c = np.unique(accessArr, return_counts=True)
    	print('len of unique application page accesses: {}'.format(len(u)))
	maxPage = getOffset(u)
	
	c.sort()
	sort_c = c[::-1]
    	n = len(u)
	aceessLength = len(accessArr)
   #	print sort_c
#####CDF
"""	cdf = []
	cnt = 0
	for i in range(n):
		cnt = cnt + sort_c[i]
#		print cnt,aceessLength
		cdf.append(cnt*1.0/aceessLength)
#	print cdf
	print cnt,aceessLength
	plot_cdf(cdf, 'D_48t_f0_r10', 'CDF', n)
	print('over')
"""
#####PDF
	#pdf = [maxPage]
	#print maxPage
	#pdf = np.zeros(maxPage, dtype=int)
	#print pdf
	#for i in range(n):
	#	print u[i],c[i]
	#	pdf[u[i]] = c[i]
	#plot_pdf(pdf, 'cg_D_48t_f0_r10', 'PDF')
#	hist, ed = np.histogram(accessArr, bins=34357617469/100)

#        plt.plot(hist)
#        plt.savefig('05_1_24_pdf.tif')

    # ## CDF
    #	sort_c = sorted(c, reverse=True)
    #	cdf = []
    #	for i in range(0, n // 100):
    #		cdf.append(np.sum(sort_c[0: (i+1)*100]))
   #	s = len(accessData)
   #	new_cdf = [x / s for x in cdf]
   	#plot_cdf(new_cdf, 'D_24t', 'CDF')
   	#print('over')

## PDF
#    	if not np.array_equal(sorted(u), u):
#    	    raise ValueError('u')

#   	pdf = []
#    	for i in range(0, n//100):
#    	     pdf.append(np.sum(c[i*100: (i+1)*100]))

#    	plot_pdf(pdf, '1B_1t', 'PDF')

#    	hist, ed = np.histogram(app_address, bins=343576174)

#    	plt.plot(hist)
#    	plt.savefig('05_1_24_pdf.tif')
#	get_interval(accessData, 20)	
#	fs_xk = np.sort(accessData)
#	val, cnt = np.unique(accessData, return_counts=True)
#	pmf = cnt/len(accessData)

#	fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))
#	plt.subplot(221)
#	plt.plot(val, fs_rv_dist2.cdf(val), 'r-', lw=5, alpha=0.6)
#	plt.title("CDF")
#	plt.subplot(224)
#	plt.plot(val, pmf, 'g-', lw=5, alpha=0.6)
#	plt.title("PMF")
#	plt.savefig('cg_D_48t_f0_pdf.png')
#	plt.show()

#	plt.subplot('cg.D.48t.f0.pdf')
#	plt.hist(accessData)
#	plot.show()
#	plot.savefig('cg_D_48t_f0_pdf.png')

#	plt.subplot(121)
#	hist, bin_edges = np.histogram(accessData)
#	cdf = np.cumsum(hist)
#	plt.plot(cdf)
#	plt.savefig('cg_D_48t_f0_cdf.png')
#	accessArr =  np.asarray(accessData, dtype=int)
	#print accessArr.size
#	accessArrUnique = getUnique(accessArr)
	#print accessArrUnique.size
#	getOffset(accessArrUnique)

if __name__ == '__main__':
    main()
