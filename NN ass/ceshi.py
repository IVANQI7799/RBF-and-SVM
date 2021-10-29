import numpy as np
A=np.array([[0.998,0.4997],[-0.008,0.998]])
B=np.array([[0.0002],[0.0008]])
K=np.array([[2490,1873.75]])
C=np.dot(B,K)
print(C)
D=A-C
print(D)
E=np.dot(D,D)

print(E)