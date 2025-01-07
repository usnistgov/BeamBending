#Benjamin Schreyer benontheplanet@gmail.com
import matplotlib.pyplot as plt
from mpmath import mpmathify
import mpmath as mp
import numpy as np
import math
import time
#Define the material profile
S = mpmathify("1E-2")
E = mpmathify("78E9")
F =  mpmathify(120)
#floating precision
mp.dps = 10

#smallest width
mini = 0.0001

#Define geometry of the flexure
def If(s):
    return ((s - 0.005)**2/ ((0.005**(2))) + mini)**3 * ((1*10**-3)**3 * 10**-4/12) #* np.exp(np.sin(2/0.01 * 13 * s))

def I(s):
    return mpmathify(If(float(s)))

plt.plot(np.linspace(0,10**-2,10000),[mp.log(I(x), b= 10) for x in np.linspace(0,10**-2,10000)])
plt.title("log plot moment area")
plt.show()

plt.plot(np.linspace(0,10**-2,10000),[(I(x)) for x in np.linspace(0,10**-2,10000)])
plt.title("plot moment area")
plt.show()

plt.plot(np.linspace(0,10**-2,10000),[float(I(x))**(1/3) for x in np.linspace(0,10**-2,10000)], c= "b")
plt.plot(np.linspace(0,10**-2,10000),[-float(I(x))**(1/3) for x in np.linspace(0,10**-2,10000)], c="b")
plt.title("cube root moment area (geometry of flexure up to constant x/y scale)")
plt.ylabel("z(m)")
plt.xlabel("y(m)")
plt.show()

#variable spaced thetas along flexure convert to cartesian points by integrating
def integrate_xz(t,s):
    t = [float(x) for x in t]
    s = [float(x) for x in s]
    x = [0.0]
    z = [0.0]
    for i in range(len(s) - 1):
        
        x.append(np.sin(t[i]) * (s[i + 1] - s[i]) + x[-1])
        z.append(-np.cos(t[i]) * (s[i + 1] - s[i]) + z[-1])
    return np.array(x),np.array(z)

    #Estimate the exponent using the linearized piecewise approximation
es= 10**-2/10000 * np.sum(1/np.sqrt(If(np.linspace(0,10**-2,10000))))

ase = int(es * np.sqrt(F/E) * np.log10(np.exp(1)))
print(ase, "approximate scale exponent")

#Key and tolerance
K = mpmathify(0.83232) #Desired bending angle in shooting
T = mpmathify(0.001)


#A single shot of bending implemented with arbitrary precision
def bending_run(M0,I_z,key):

    ME = 0
    #derivatives
    def dM_ds(t):
        return F * mp.sin(t)

    def dt_ds(s,M):
        return M/E/I_z(s)
    
    M = M0
    t = mpmathify(0)
    s = mpmathify(0)
   
    Ms =[]
    ts = []
    ss = []

    j = 0
    i = 0
    esc = 0
    #Iterate until the end of the flexure S
    while s < S:
        i += 1
        j += 1
        j = j % 10000

        #step s a small amount so that theta is small change and so is s
        #s case                                 #theta case
        ds = min(mpmathify(S)/mpmathify(1000),mp.fabs(mpmathify(T)*E * I_z(s)/ M))
        if esc != 0:
            ds = 0.001 * mpmathify(S)
        if j == 0:
            print(i,"s",s,"M",M,"ds",ds,"ds_default",mpmathify(S)/mpmathify(1000), "M0", M0)
        M+= ds * dM_ds(t)
        t+= ds * dt_ds(s,M)
        s+= ds
        if t > 3 or i > 20000:
            return mpmathify(3), Ms, ss, ts
        
        #We can approximate the local error by truncation of taylor series, note that the chained error of Euler's methods
        #is not so easy, so to really check error decrease ds in the min statement and see if the result changes significantly
        #Euler integration is linear in ds in the global error
        err = np.abs((ds**2 )/ 2 * ( M / E /(I_z(s)**2) * (I_z(s + 10**-8) - I_z(s)) * 10**8 + F * mp.sin(t)/E/I_z(s)))
        
        #accumulate local error for debug
        ME += err
        
        Ms.append(M)
        ss.append(s)
        ts.append(t)
    
    print("THETA ERR SUM", ME)
    return t, Ms, ss, ts

#Find the initial condition order of magnitude by shooting with binary search since
#there is a monotonicity to the relation between M0, and theta_end
def binary_search_exponent_bending(LEXP, REXP,I_z,key):
    theta, _,_ ,_ = bending_run(mpmathify('1E-' + str((LEXP + REXP)//2)),I_z,key)
    
    print(LEXP,REXP,key,theta,"k,t")
    
    def nearest_power(x):
        return np.sign(np.log10(float(x)))*math.ceil(np.abs(np.log10(float(x))))
    
    #Tolerance of integer exponent stopping condition
    if not math.isinf(np.log10(float(theta))):
        if nearest_power(theta) == nearest_power(key):
            print("OOM match",int(np.log10(float(theta))),int(np.log10(float(key))))
            return (LEXP + REXP)//2
    
    #Binary search
    if abs(REXP-LEXP) < 2:
        return LEXP
    if(theta < mpmathify(0.1)):
        return binary_search_exponent_bending(LEXP,(LEXP+REXP)//2,I_z,key)
    if(theta > mpmathify(0.1)):
        return binary_search_exponent_bending((LEXP+REXP)//2, REXP,I_z,key)
    return EXP

t0= time.time_ns()
EX = binary_search_exponent_bending(0,ase - int(mp.log10(float(K))),I,K) 
t1= time.time_ns()
print("EXPONENT", EX , "TIME (us)", (t1-t0) / 10**6)

#Find the linear scale adjustment by binary search, to be used after finding the order of magnitude
def binary_search_bending(L, R,EXP,I_z,key, tol):
    theta, _,_ ,_  = bending_run((L + R)/mpmathify(2) * mpmathify('1E-' + str(EXP)),I_z,key)
    print(L,R,"\n",theta,mp.fabs(theta - key),"t")
    if mp.fabs(theta - key) < tol:
        return (L + R)/mpmathify(2)
    if(theta > key):
        return binary_search_bending(L,(L+R)/mpmathify(2), EXP,I_z,key, tol)
    if(theta < key):
        return binary_search_bending((L+R)/mpmathify(2), R,EXP,I_z,key, tol)
    return (L + R)/mpmathify(2)

t0= time.time_ns()
res= binary_search_bending(mpmathify(0),mpmathify(1000),EX,I,mpmathify(K),mpmathify(T))
t1= time.time_ns()
print("Fine adjustment: ",res,(t1-t0) / 10**6,'(us)')

#Test run with the shooting discovered parameters
theta, Ms, ss, ts = bending_run(res*mpmathify('1E-' + str(EX)),I,10)

x,z = integrate_xz(ts,ss)

plt.plot(x,z)
#plt.title("Moment =" +str(Ms[-1]) + " theta = " + str(ts[-1]))
plt.title("Typical bending geometry theta (rad) = " + str(round(ts[-1],5)))
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.xlim([-1.1*np.max([np.max(np.abs(x)),np.max(np.abs(z))]),1.1*np.max([np.max(np.abs(x)),np.max(np.abs(z))])])
plt.ylim([-1.1*np.max([np.max(np.abs(x)),np.max(np.abs(z))]),1.1*np.max([np.max(np.abs(x)),np.max(np.abs(z))])])
plt.show()