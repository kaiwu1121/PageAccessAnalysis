from collections import OrderedDict

count_s = 0
count_f = 0
count_m = 0 
 
class LRUCache(OrderedDict):

    def __init__(self,capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
     

    def get(self,key):
        if self.cache.has_key(key):
            value = self.cache.pop(key)
            self.cache[key] = value
        else:
            value = None
         
        return value
     

    def set(self,key,value):
	global count_s
	global count_f
	global count_m
        if self.cache.has_key(key):
            value = self.cache.pop(key)
            self.cache[key] = value
	    count_s += 1
        else:
	    count_f += 1
            if len(self.cache) == self.capacity:
                self.cache.popitem(last = False)   
                self.cache[key] = value
		count_m += 1
            else:
                self.cache[key] = value




top = 400000
c = LRUCache(top)



output = 'out.bt.D.x.t25.p2800000.f1000'
lineList = list()
with open(output) as f:
	for line in f:
        	lineList.append(int(line))
       

#for i in range(5,10):
for i in lineList:
    c.set(i,1)
  
print count_s, count_f, count_s+count_f, count_m  
#print c.cache, c.cache.keys()
  
#c.get(5)
#c.get(7)
  
#print c.cache, c.cache.keys()
  
#c.set(10,100)
#print c.cache, c.cache.keys()
  
#c.set(9,44)
#output = 'out.cg.D.x.t10.p400000.f1000'#print c.cache, c.cache.keys()
