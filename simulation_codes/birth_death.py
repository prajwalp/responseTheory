import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy as sp
import scipy.special as special
from mpmath import gamma
from mpmath import hyp2f1
from math import factorial as fac
from numba import prange
from sympy import factorint

b = 0.4
d = 0.5
T_st = 25
n0 = 20

n=np.arange(1,251)

t=50
dt=0.001

## Time evolution of probability

def poch(a,n):
    return special.gamma(a+n)/special.gamma(a)

def nfac(nt):
    return np.float128(np.math.factorial(nt))

def stir(n):
    n=np.float128(n)
    return  np.sqrt(2*np.pi*n)*np.power(n/np.e,n)  


def can_mu(mul_list,div_list):
    mul = []
    allkeys = []
    for i in mul_list:
        val = factorint(i)
        allkeys = allkeys + list(val.keys())
        mul.append(val)
    div = []
    for i in div_list:
        val = factorint(i)        
        allkeys = allkeys + list(val.keys())
        div.append(val)
    allkeys = list(set(allkeys))
    allkeys.sort()
    
    prod = np.float128(1.0)
    for k in allkeys:
        kval = 0
        for i in range(len(mul_list)):            
            if(k in mul[i].keys()):
                kval += mul[i][k]
        for j in range(len(div_list)):
            if(k in div[j].keys()):
                kval -= div[j][k]
        prod *= np.float128(k**kval)
    return prod
    
def pij_gen(i,j,b,d,t):
    s1 = np.float128(0.0)
    
    hyp_fn = hyp2f1(-i,-j,-1-i-j,(d*np.exp(b*t) - b*np.exp(d*t))*(b*np.exp(b*t) - d*np.exp(d*t))/(b*d*(np.exp(b*t) - np.exp(d*t))**2))
    s1 = hyp_fn  * ((d*(np.exp((b-d)*t) - 1))/(b*np.exp((b-d)*t) - d))**j *  nfac(i+j+1) / nfac(i+1) / nfac(j)
    probFac = (b/d -1)**2 * (b/d)**j * np.exp((b-d)*t) * (1-np.exp((b-d)*t))**i * (1-b*np.exp((b-d)*t)/d)**(-i-2) * (i+1)/(j+1)
    prob1 = probFac * s1
    
    return prob1

def Pij(i,j,lam,mu,t):
    beta = 2.0
    if(type(j)==int or type(j)==np.int64):
        if(i<=j):
            prob = pij_gen(i,j,lam,mu,t)
        else:
            prob = pij_gen(j,i,lam,mu,t) * (lam/mu)**j * (i+1) / (lam/mu)**i /(j+1)
    elif(type(j)==np.ndarray):
        z = len(j)
        prob = np.zeros(z)
        for y in range(j[0],j[z-1]):
            if(i<=y):
                prob[y-j[0]] = pij_gen(i,y,lam,mu,t)
            else:
                prob[y-j[0]] = pij_gen(y,i,lam,mu,t) * (lam/mu)**y / (y+1) / (lam/mu)**i *(i+1)
    return prob

temp = Pij(70,70,b,d,0.0009)

@numba.njit
def evolg(x,b,d):
    r_tot = (b+d)*x
    t_reac = -np.log(np.random.random())/r_tot
    r_reac = np.random.random()
    x_new = x
    if(r_reac<b*x/r_tot):
        x_new = x_new + 1
    else:
        x_new = x_new - 1
    return t_reac,x_new

@numba.njit
def evol(x,b,d,dt):
    check = np.random.random()
    x_new = x
    if(check<=b*x*dt):
        x_new = x_new +1
    elif(check<=(b+d)*x*dt):
        x_new = x_new -1
    return x_new        

@numba.njit
def time_evol(x_init,b,d,dt,t_total):
    t_track = 0
    x_new = x_init
    while(t_track<t_total):
        x_new = evol(x_init,b,d,dt)
        if(x_new ==0):
            break
        t_track +=dt
        x_init=x_new
    return x_new

@numba.njit
def gtime_evol(x_init,b,d,T_st):   
    x = x_init
    t_stgen=0
    while(True):
        dt2,x_temp = evolg(x,b,d)
        t_stgen=t_stgen+dt2
        if(t_stgen>T_st):
            break
        x = x_temp
        if(x==0):
            break
    return x

temp = evolg(2,b,d),gtime_evol(2,b,d,1),evol(2,b,d,dt),time_evol(2,b,d,dt,1)

n_trials = int(1e5)
p_sim = np.zeros(n_trials)
for i in range(n_trials):
    x_init = n0
    p_sim[i] = gtime_evol(x_init,b,d,t)

red_p_sim = p_sim[np.where(p_sim!=0)[0]]

values,edges = np.histogram(red_p_sim,bins='auto')
pos = np.where(values!=0)[0]
new_edges = edges[np.append(pos,pos[-1]+1)]
edge_width = (np.diff(new_edges)-np.diff(new_edges)/5.0)

plot_x = new_edges[:-1]
plot_y = values[pos]/(n_trials)

## Perturbation

x = np.arange(1,301)
y = np.arange(1,301)

n_t = 20
t_array = np.linspace(0.1,10,n_t)
r_xv = np.zeros(n_t)
r_firstTerm = np.zeros(n_t)
r_secontTerm = np.zeros(n_t)
v = 1-b/d

def resp_calc_xv(b,d,t_resp):
    sum_xy = 0.0
    adt = 0
    xdt = 0
    ax_avg = 0
    for i in range(len(x)):
        ax = x[i]
        adty = 0
        xdty = 0
        for j in range(len(y)):
            dphib = (b/d)**(y[j]-2) * (y[j]*(1-b/d) - 1) /d
            if(i!=0):
                adty += ax * (b * (x[i]-1) * Pij(y[j]-1,x[i]-2,b,d,t_resp) + d*(x[i]+1)*Pij(y[j]-1,x[i],b,d,t_resp) - ((b+d)*(x[i]) - (d-b))*Pij(y[j]-1,x[i]-1,b,d,t_resp))*np.exp((d-b)*t_resp)* dphib
                xdty += (b*(x[i]-1) * Pij(y[j]-1,x[i]-2,b,d,t_resp) + d*(x[i]+1)*Pij(y[j]-1,x[i],b,d,t_resp) - ((b+d)*(x[i]) - (d-b))*Pij(y[j]-1,x[i]-1,b,d,t_resp))*np.exp((d-b)*t_resp)* dphib
            else:
                adty += ax * (d*(x[i]+1)*Pij(y[j]-1,x[i],b,d,t_resp) - ((b+d)*(x[i]) - (d-b))*Pij(y[j]-1,x[i]-1,b,d,t_resp))*np.exp((d-b)*t_resp)* dphib
                xdty += (d*(x[i]+1)*Pij(y[j]-1,x[i],b,d,t_resp) - ((b+d)*(x[i]) - (d-b))*Pij(y[j]-1,x[i]-1,b,d,t_resp))*np.exp((d-b)*t_resp)* dphib
        
        ax_avg += ax*(1-b/d) * (b/d)**(x[i]-1)
        adt +=adty
        xdt +=xdty
        
    sum_xy = -adt + ax_avg*xdt    
            
    return -adt,ax_avg*xdt 
        
temp = resp_calc_xv(b,d,0.1)

for t_iter in range(n_t):
    values = resp_calc_xv(b,d,t_array[t_iter])
    r_xv[t_iter] = values[0]+values[1]
    r_firstTerm[t_iter] = values[0]
    r_secontTerm[t_iter] = values[1]

del_b = 0.1*b

@numba.njit(parallel=True)
def gst_gen(b,d,n_trials,T_st):
    long_time_state = np.zeros(n_trials)
    for i in prange(n_trials):
        x = np.random.randint(1,50)
        t_stgen=0
        while(True):
            dt2,x_temp = evolg(x,b,d)
            t_stgen=t_stgen+dt2
            if(t_stgen>T_st):
                break
            x = x_temp
            if(x==0):
                break
        long_time_state[i] = x
    long_time_state = long_time_state[np.where(long_time_state!=0)[0]]
    return long_time_state

temp = gst_gen(b,d,1,T_st)

n_c = int(1e5)
T_c = 90

lt_state = gst_gen(b,d,n_c,T_c)

survived = lt_state[np.where(lt_state!=0)[0]]
x_avg_unp = np.mean(survived)
T = 25
t_list = np.arange(0,T+dt,dt)

x_pert_avg_d = np.zeros(len(t_list))
survival_counter = np.zeros(len(t_list))

@numba.njit
def gill_resp(survived,b,d,resp_dt,del_b,T,nt):
    t_list = np.linspace(resp_dt,T,nt)
    x_avg = np.zeros(len(t_list))
    avg_counter = np.zeros(len(t_list))
    for i in range(len(survived)):
        x = survived[i]
        xi=[]
        ti=[]
        t_resp = 0
        while(t_resp<T):
            x_temp = x
            if(t_resp==0):
                x_temp = evol(x,b+del_b/resp_dt,d,resp_dt)
                t_resp = t_resp+resp_dt
            else:
                dtg,x_temp=evolg(x,b,d)
                t_resp=t_resp+dtg  
            x = x_temp
            xi.append(x)
            ti.append(t_resp)
            if(x_temp==0):
                break
        last_k = 0
        for j in range(nt):
            tj = t_list[j]
            for k in range(last_k,len(ti)-1):
                tk = ti[k]
                tkp = ti[k+1]
                if(tk<= tj and tj <tkp and xi[k]!=0):
                    x_avg[j] += xi[k]
                    avg_counter[j] +=1
                    last_k = k
                    break    
    x_avg = x_avg/(avg_counter)
    print(avg_counter[-1])
    return x_avg
    
temp = gill_resp(np.array([2,1]),b,d,0.001,del_b,2,5)

Tr = 30
nt = 500

rdt = 1e-4
x_avg = gill_resp(survived,b,d,rdt,del_b,Tr,nt)



