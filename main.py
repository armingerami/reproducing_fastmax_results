import sys
import subprocess
try:
    import numpy as np
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    import numpy as np
import math

try:
    import torch, torchvision, lightning, kornia
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchvision'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightning'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kornia'])
    import torch, torchvision, lightning, kornia


def custom_score1(Q, K, V, n, d):
    # Q, K, and V are n by d matrices
    p = 100
    sqrtd = d**(-0.25)
    # sqrtd = 1
    R = np.zeros([d,n,p]) # expansion terms (R(q)). Expansion is based on query.
    A = np.zeros([d,n,p]) # expansion coefficients (A(k)) for the rest of the dimensions.
    RR = np.ones([n,p]) # R part of the dot product of the series
    AA = np.ones([n,p]) # A part of the dot product of the series
    U = np.zeros([d,p]) # bundeling up As of different Keys with respect to the Value column for the matrix multiplication
    D = np.zeros(p) # bundeling up As of different Keys for division part of fastmax
    Div = np.zeros(n) # division part of fastmax
    Ans = np.zeros([n,d]) # result of fastmat(QK^T)*V
    fastmax = np.zeros([n,n])
    x = 0 # center                


    for i in range(d):
        for j in range(n):
            temp_fac = 1
            temp_k = np.exp(K[j,i]*x*sqrtd)
            temp_q = 1
            for k in range(p):
                A[i,j,k] = temp_k/temp_fac
                R[i,j,k] = temp_q
                temp_fac *= k+1
                temp_k *= K[j,i]*sqrtd
                temp_q *= (Q[j,i]*sqrtd - x)


    for i in range(d):
        for j in range(n):
            for k in range(p):
                RR[j,k] *= R[i,j,k]
                AA[j,k] *= A[i,j,k]


    for i in range(d):
        for j in range(n):
            for k in range(p):
                U[i,k] += V[j,i]*AA[j,k]

    for j in range(n):
            for k in range(p):
                D[k] += AA[j,k]

    for j in range(n):
        for k in range(p):
            Div[j] += RR[j,k]*D[k]

    for i in range(d):
        for j in range(n):
            for k in range(p):
                Ans[j,i] += RR[j,k]*U[i,k]/Div[j]

    for i in range(n):
        for j in range(n):
            for k in range(p):
                fastmax[i,j] += RR[i,k]*AA[j,k]/Div[i]
    
    # print(AA)
    print(fastmax)
    return Ans


def custom_score2(Q, K, V, n, d):
    p = 10
    sqrtd = np.sqrt(d)
    fastmax = np.zeros([n,n])
    div = np.zeros(n)
    for i in range(n):
        for j in range(n):
            for kk in range(d):
                k = kk
                fastmax[i][j] += (Q[i,k] - K[j,k])**2
            # fastmax[i][j] = np.exp(-fastmax[i][j])/sqrtd
            fastmax[i][j] = 1/fastmax[i][j]
            
    for i in range(n):
        for j in range(n):
            div[i] += fastmax[i][j]

    for i in range(n):
        for j in range(n):
            fastmax[i][j] /= div[i]

    print(fastmax) 

    return 0

def custom_score3(Q, K, V, n, d):
    p = 10
    sqrtd = np.sqrt(d)
    fastmax = np.zeros([n,n])
    Qn = np.zeros([n,d])
    Kn = np.zeros([n,d])
    Ka = np.zeros([n])
    a = np.zeros([n,n])
    div = np.zeros(n)
    b = 0.1

    for i in range(n):
        temp1 = 0
        temp2 = 0
        for j in range(d):
            temp1 += Q[i,j]**2
            temp2 += K[i,j]**2
        Ka[i] = np.sqrt(temp2)
        Qn[i] = Q[i]/np.sqrt(temp1)
        Kn[i] = K[i]/np.sqrt(temp2)

    for i in range(n):
        for j in range(n):
            for kk in range(d):
                k = kk
                a[i][j] += (Q[i,k] -K[j,k])**2

    # a = np.ones([n,n])*10
    for i in range(n):
        for j in range(n):
            for kk in range(d):
                k = kk
                # fastmax[i][j] += ((-a[i][j])**k)/(math.factorial(k))
                # fastmax[i][j] += ((-a[i][j]+1)**k)
                fastmax[i][j] += Ka[j]*min(np.exp(-(Qn[i,k] -Kn[j,k])**2/b)/(b*np.sqrt(3.1415)),5)
            
    for i in range(n):
        for j in range(n):
            div[i] += fastmax[i][j]

    for i in range(n):
        for j in range(n):
            fastmax[i][j] /= div[i]

    print(fastmax[0]) 

    # print(a)

    return 0


def custom_score4(Q, K, V, n, d):
    p = 2
    sqrtd = np.sqrt(d)
    # sqrtd = d
    fastmax = np.zeros([n,n])
    Qn = np.zeros([n,d])
    Kn = np.zeros([n,d])
    Qa = np.zeros([n])
    Ka = np.zeros([n])
    div = np.zeros(n)
    b = 0.1

    for i in range(n):
        temp1 = 0
        temp2 = 0
        for j in range(d):
            temp1 += Q[i,j]**2
            temp2 += K[i,j]**2
        Qa[i] = np.exp(np.sqrt(temp1))
        Ka[i] = np.exp(np.sqrt(temp2))
        Qa[i] = np.sqrt(temp1)
        Ka[i] = np.sqrt(temp2)
        Qn[i] = Q[i]/np.sqrt(temp1)
        Kn[i] = K[i]/np.sqrt(temp2)

    # a = np.ones([n,n])*10
    for i in range(n):
        for j in range(n):
            a = np.dot(Q[i], K[j])/sqrtd
            for k in range(p):
                # fastmax[i][j] += (np.dot(Qn[i], Kn[j])**k)/math.factorial(k)
                fastmax[i][j] += a**k/math.factorial(k)
            
    for i in range(n):
        for j in range(n):
            div[i] += fastmax[i][j]

    for i in range(n):
        for j in range(n):
            fastmax[i][j] /= div[i]

    print(fastmax[0]) 

    # print(a)

    return 0

def custom_score5(Q, K, V, n, d):
    p = 2
    sqrtd = np.sqrt(d)
    sqrtd = d
    fastmax = np.zeros([n,n])
    Qn = np.zeros([n,d])
    Kn = np.zeros([n,d])
    Qa = np.zeros([n])
    Ka = np.zeros([n])
    div = np.zeros(n)
    b = 0.1

    for i in range(n):
        temp1 = 0
        temp2 = 0
        for j in range(d):
            temp1 += Q[i,j]**2
            temp2 += K[i,j]**2
        Qa[i] = np.exp(np.sqrt(temp1))
        Ka[i] = np.exp(np.sqrt(temp2))
        Qa[i] = np.sqrt(temp1)
        Ka[i] = np.sqrt(temp2)
        Qn[i] = Q[i]/np.sqrt(temp1)
        Kn[i] = K[i]/np.sqrt(temp2)

    # a = np.ones([n,n])*10
    for i in range(n):
        for j in range(n):
            fastmax[i][j] = np.dot(Q[i],K[j])
            
    for i in range(n):
        for j in range(n):
            div[i] += fastmax[i][j]

    # for i in range(n):
    #     for j in range(n):
    #         fastmax[i][j] /= div[i]

    print(fastmax[0]) 

    # print(a)

    return 0

n = 20
d = 100
np.random.seed(86481938)
sqrtd = np.sqrt(d)
# sqrtd = 1
# K = np.array([[0.1],[0.05],[0.05],[0.8]])

Q = np.ones([n,d])
K = np.ones([n,d])
V = np.ones([n,d])
Q = np.random.rand(n,d)
K = np.random.rand(n,d)
V = np.random.rand(n,d)
Q = np.random.normal(0,1,[n,d])
K = np.random.normal(0,1,[n,d])
V = np.random.normal(0,1,[n,d])

# for i in range(n):
#     temp1 = 0
#     temp2 = 0
#     for j in range(d):
#         temp1 += Q[i,j]**2
#         temp2 += K[i,j]**2
#     Q[i] = Q[i]/np.sqrt(temp1)
#     K[i] = K[i]/np.sqrt(temp2)

direct = np.zeros([n,d])
div = np.zeros(n)
softmax = np.zeros([n,n])

for i in range(n):
    for k in range(n):
        div[i] += np.exp(np.dot(Q[i],K[k])/sqrtd)
        # div[i] += (np.dot(Q[i],K[k]))


for i in range(n):
    for j in range(d):
        for k in range(n):
            direct[i,j] += V[k,j]*np.exp(np.dot(Q[i],K[k])/sqrtd)/div[i]
            softmax[i,k] = np.exp(np.dot(Q[i],K[k])/sqrtd)/div[i]
            # softmax[i,k] = (np.dot(Q[i],K[k]))/div[i]

print(softmax[0])

# ans_fmm = custom_score1(Q,K,V,n,d)
# ans_fmm = custom_score2(Q,K,V,n,d)
ans_fmm = custom_score4(Q,K,V,n,d)

# print(Q)
# print(K)

# print()
# print("############################")
# print(direct)
# print(ans_fmm)

# mx = 0
# mxr = 0
# for i in range(n):
#     for j in range(d):
#         mx = max(mx, abs(ans_fmm[i,j] - direct[i,j]))
#         mxr = max(mxr, abs(ans_fmm[i,j] - direct[i,j])/abs(direct[i,j]))

# print("maximum abs error = ", mx)
# print("maximum relative error = ", mxr)
# print(np.amax(Q))
# print(np.amax(K))




# n = 1
# d = 10
# p = 10
# k = np.random.rand(n,d)
# q = np.random.rand(n,d)
# x = 0 # expansion center

# direct = np.zeros(n)
# for i in range(n):
#     for j in range(n):
#         direct[i] += np.exp(np.dot(q[i],k[j]))


# ans = np.zeros(n)
# A = np.zeros(p) # expansion coefficients (A(k))
# R = np.zeros([n,p]) # expansion terms (R(q)). Expansion is based on query.

# for i in range(n):
#     temp_fac = 1
#     temp_k = np.exp(k[i]*x)
#     temp_q = 1
#     for j in range(p):
#         A[j] += temp_k/temp_fac
#         R[i,j] += temp_q
#         temp_fac *= j+1
#         temp_k *= k[i]
#         temp_q *= (q[i] - x)

# for i in range(n):
#     for j in range(p):
#         ans[i] += A[j]*R[i,j]

# print(direct)
# print(ans)
