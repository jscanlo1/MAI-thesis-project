import numpy as np

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind
    #return ind[np.argsort(array[ind])][::-1]

with open('Final_Layer_Weights.txt') as f:
    polyShape = []
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            polyShape.append(line)
print(polyShape)


fake = polyShape[0]
real = polyShape[1]
print(fake)

print(f"FAKE: {top_elements(fake,5)}")
print(f"REAL: {top_elements(real,5)}")






