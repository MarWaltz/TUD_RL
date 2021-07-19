"""
This helper file consists of a specialized MinHeap that allows to safe entire transitions. The heap is orders by 
the 'key_idx' value, which is the nth position of the transition. 

The zipf quantile function calculates the static rank distribution of the transisions in the heap, and also their
probabilities of sampling.
"""

import numpy as np 

class MinHeap:
    def __init__(self, max_size: int, key_idx: int):
        # Use leading zero as parent for whole tree as this will enable integer division later on
        self.heaplist = [(0,0,0,0,0,-float("Inf"))]
        self.size = 0
        self.max_size = max_size
        self.key_idx = key_idx # key element of the tuple to use as the comparing value by the heap
        
    def perc_up(self, i):
        while i // 2 > 0:
            if self.heaplist[i][self.key_idx] < self.heaplist[i // 2][self.key_idx]: # is item smaller than its parent?
                tmp = self.heaplist[i // 2] # save parent 
                self.heaplist[i // 2] = self.heaplist[i] # replace last item with parent
                self.heaplist[i] = tmp # replace child with former parent
            i = i // 2
            
    def insert(self,k):
        if self.size == self.max_size:
            self.heaplist.pop()
            self.heaplist.append(k)
        else:
            self.heaplist.append(k) # Append value to heap
            self.size += 1 # Increase size by one
        self.perc_up(self.size) # Walk up nodes until the value fits
        
    def perc_down(self,i):
        while (i * 2) <= self.size:
            mc = self.min_child(i) # Determine which child of i is the smallest
            if self.heaplist[i][self.key_idx] > self.heaplist[mc][self.key_idx]: #
                tmp = self.heaplist[i]
                self.heaplist[i] = self.heaplist[mc]
                self.heaplist[mc] = tmp
            i = mc
    
    def min_child(self, i): # Determine the index of the smallest child 
        if i * 2 + 1 > self.size:
            return i * 2
        else:
            if self.heaplist[i*2][self.key_idx] < self.heaplist[i*2+1][self.key_idx]:
                return i * 2
            else:
                return i * 2 + 1
            
    def del_min(self):
       retval = self.heaplist[1]
       self.heaplist[1] = self.heaplist[self.size]
       self.size -= 1
       self.heaplist.pop()
       self.perc_down(1)
       return retval

    def replace_key(self,i,n_key):
        last = self.heaplist[self.size] # Save last entry
        current = list(self.heaplist[i]) # Save entry at replacement index
        self.heaplist[i] = last # Replace current entry with the last

        # If the value of the replaced index is less than its parent, percolate up, else down
        while i // 2 > 0:
            if self.heaplist[i][self.key_idx] < self.heaplist[i // 2][self.key_idx]:
                self.perc_up(i)
            else:
                self.perc_down(i)
            i = i // 2

        self.heaplist.pop()
        self.size -= 1
        current[self.key_idx] = n_key # Replace the key in the current obs
        self.insert(tuple(current))


    def build_heap(self,alist):
        i = len(alist) // 2
        self.size = len(alist)
        self.heaplist = [0] + alist[:]
        while i > 0:
            self.perc_down(i)
            i = i - 1
            
# Calculate k segments of equal probability for the Zipf Distribution on the support 1,...,N
# Also return the sampling probability of each rank
def zipf_quantiles(N,k,alpha):
    pmf = np.array([i**(-alpha) for i in range(1,N+1)]) # Probability mass function
    cdf = np.cumsum(pmf) # Cumsum of probabilities
    cdf /= cdf[-1] # Normalize to get P(Ω) = 1
    quants = cdf//(1/k) # Calculate the quantile, each point falls into
    
    # Also output probabilities of sampling each rank
    probs = pmf/np.sum(pmf)

    ranges = [] # List of tuples (left,right) that are inclusive ranges of indices to sample from
    prev_range = (1,1)   # When no points fall into a range
    for i in range(k):
        where = np.nonzero(quants == i)[0]
        if len(where) > 0:
            this_range = (where[0] + 1,where[-1] + 1) # Due to zero indexing
            ranges.append(this_range)
            prev_range = this_range
        else:
            ranges.append(prev_range)
    return ranges, probs
            
#h=MinHeap(5,0)
#a=[(5,88),(3,6),(6,12),(5,7),(3,8)]
#b=[(0,0,0,0,0)]
#h.build_heap(a)
#h.replace_key(4,1)
#print(h.heaplist)
#h.insert((1,12))
#print(h.heaplist)