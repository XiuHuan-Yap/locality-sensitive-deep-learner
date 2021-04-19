import numpy as np

from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import time 
from itertools import product
from . import empty



def initProcess(toShare):
    empty.sh_arr=toShare

def arr_(arg):
    time.sleep(10)
    return np.dot(empty.sh_arr, empty.sh_arr[:,arg])

def arr_func(arr, arg):
    time.sleep(10)
    return np.dot(arr, arr[:,arg])  

class testClass():
    def __init__(self, arr, threads=8):
        self.arr=arr
        self.threads=threads
    
    def notshared(self):
        args=np.reshape(range(len(self.arr)), (10, int(len(self.arr)/10)))
        with Pool(self.threads) as p:
            out=p.starmap(arr_func, [(self.arr, args[i]) for i in range(10)])
        return out
    
    def shared(self):
        args=np.reshape(range(len(self.arr)), (10, int(len(self.arr)/10)))
        with SharedMemoryManager() as smm:
            shm=smm.SharedMemory(size=self.arr.nbytes)
            sh_arr=np.ndarray(self.arr.shape, 
                              dtype=self.arr.dtype,
                              buffer=shm.buf
                             )
            sh_arr=self.arr
            with Pool(self.threads,
                      initializer=initProcess,
                      initargs=(sh_arr,)
                     ) as p:
                out=p.starmap(arr_, [(args[i], ) for i in range(10)])
        return out
    
    def run_func(self):
        return arr_func(self.arr)
    