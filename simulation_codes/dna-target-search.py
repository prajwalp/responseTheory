import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange
import scipy.linalg
import mpmath
from scipy.interpolate import interp1d
import gc
import scipy.optimize
import scipy.signal
import xrft
import xarray
import scipy.special

L = 100
kon = 0.5
koff = 1
u = 2

target = 60

## Evolution and Matrix Functions

@numba.njit
def normal_evol(j,u,kon,koff,L,target,dt):
    if(j==0):
        check = np.random.random()
        if(check<kon*(L-1)*dt):
            jnew = np.random.randint(1,L)
        else:
            jnew = j
    else:
        if(j>1 and j<L-1):
            check = np.random.random()
            if(check<=u*dt):
                jnew = j+1
            elif(check <=2*u*dt):
                jnew = j-1
            elif(check<=(2*u+koff)*dt):
                jnew = 0
            else:
                jnew = j
        elif(j==1):
            check = np.random.random()
            if(check<=u*dt):
                jnew = j+1
            elif(check<=(u+koff)*dt):
                jnew = 0
            else:
                jnew = j
        elif(j==L-1):
            rate_total = u + koff
            check = np.random.random()
            if(check<=u*dt):
                jnew = j-1
            elif(check<=(u+koff)*dt):
                jnew = 0
            else:
                jnew = j
    return jnew

@numba.njit
def evol(j,u,kon,koff,L,target):
    dt = 0
    jnew = 0
    if(j==0):
        rate_total = kon*(L-1)
        dt = -np.log(np.random.random())/rate_total
        jnew = np.random.randint(1,L)
    else:
        if(j>1 and j<L-1):
            rate_total = 2*u + koff*1.0
            dt = -np.log(np.random.random())/rate_total
            check = np.random.random()
            if(check<=u/rate_total):
                jnew = j-1
            elif(check<=2*u/rate_total):
                jnew = j+1
            else:
                jnew = 0
        elif(j==1):
            rate_total = u + koff*1.0
            dt = -np.log(np.random.random())/rate_total
            check = np.random.random()
            if(check<=u/rate_total):
                jnew = j+1
            else:
                jnew = 0
        elif(j==L-1):
            rate_total = u + koff*1.0
            dt = -np.log(np.random.random())/rate_total
            check = np.random.random()
            if(check<=u/rate_total):
                jnew = j-1
            else:
                jnew = 0
        else:
            print(j)
    return dt,jnew


@numba.njit(parallel=True)
def pt_gen(ntrials,T,j0,u,kon,koff,L,target):
    pt = np.zeros(ntrials)
    for i in prange(ntrials):
        j = j0
        t = 0
        while(True):
            dt,jnew = evol(j,u,kon,koff,L,target)
            if(t+dt>T):
                break
            t = t+dt
            j = jnew
            if(j==target):
                break
        pt[i] = j
    return pt
    

temp = evol(0,u,kon,koff,L,target),normal_evol(20,u,kon,koff,L,target,1e-3),pt_gen(2,0.001,10,u,kon,koff,L,target)

def transitionMatrixGen(u,kon,koff,L,target):
    W = np.zeros((L,L))
    H = np.zeros(W.shape)
    for i in range(L):
        for j in range(L):
            if(i!=j and j!=target):
                if(j>1 and j<L-1):
                    if(i == j-1):
                        W[i,j] = u
                    elif(i==j+1):
                        W[i,j] = u
                    elif(i==0):
                        W[i,j] = koff
                elif(j==1):
                    if(i==0):
                        W[i,j] = koff
                    if(i==j+1):
                        W[i,j] = u
                elif(j==L-1):
                    if(i==0):
                        W[i,j] = koff
                    if(i==j-1):
                        W[i,j] = u
                elif(j==0):
                    W[:,j] = kon
                    W[0,0] = 0
    H = W - np.sum(W,axis=0) * np.identity(L)
    
    return H
                    

def eigGen(H):
    eig_vals,right_eig_vecs = np.linalg.eig(-H)
    sortIndex = np.argsort(eig_vals)
    eig_vals = eig_vals[sortIndex]
    right_eig_vecs = right_eig_vecs[:,sortIndex]

    left_eig_vals,left_eig_vecs = np.linalg.eig(-H.T)
    sortIndex = np.argsort(left_eig_vals)
    left_eig_vals = left_eig_vals[sortIndex]
    left_eig_vecs = left_eig_vecs[:,sortIndex]

    left_eig_vecs[:,0] = left_eig_vecs[:,0]/left_eig_vecs[0,0]
    
    abssite = np.where(right_eig_vecs[:,0]==1)[0][0]
    if(abssite!=target):
        print('error')
    right_eig_vecs[abssite,1] = 0
    right_eig_vecs[:,1] = right_eig_vecs[:,1] / np.sum(right_eig_vecs[:,1])
    right_eig_vecs[target,1] = -1


    for i in range(1,H.shape[0]):
        prod = np.dot(left_eig_vecs[:,i] , right_eig_vecs[:,i])
        left_eig_vecs[:,i] = left_eig_vecs[:,i] /prod
    
    return eig_vals,right_eig_vecs,left_eig_vecs

## Quasi stationary state and eigenfunctions

H = transitionMatrixGen(u,kon,koff,L,target)
eig_vals,right_eig_vecs,left_eig_vecs = eigGen(H)
nonTargetSites = np.delete(np.arange(0,L,1),target)
fullSites = np.arange(0,L,1)


T = 200
ntrials = int(5e7)
j0 = 0

removedpt = np.array([])

for i in range(100):
    temp = pt_gen(ntrials,T,j0,u,kon,koff,L,target)
    removedpt = np.concatenate((removedpt,temp[np.where(temp!=target)[0]]))

    del(temp)
    gc.collect()

j_avg = np.sum(nonTargetSites*right_eig_vecs[:,1][nonTargetSites])
j2_avg = np.sum(nonTargetSites**2*right_eig_vecs[:,1][nonTargetSites])
j3_avg = np.sum(nonTargetSites**3*right_eig_vecs[:,1][nonTargetSites])

## Perturbation : Theory

# delphiu calculation

eps_u = 1e-3

H0 = transitionMatrixGen(u-eps_u,kon,koff,L,target)
Heps = transitionMatrixGen(u+eps_u,kon,koff,L,target)

l10,eig0 = eigGen(H0)[0][1],eigGen(H0)[1][:,1][nonTargetSites]
l1eps,eigEps = eigGen(Heps)[0][1],eigGen(Heps)[1][:,1][nonTargetSites]
dl1u = (l1eps - l10)/2/eps_u

phi0 = np.log(eig0)
phiEps = np.log(eigEps)

dphiu_reduced = (phiEps - phi0)/2/eps_u


dphiu = np.zeros(L)
dphiu[nonTargetSites] = dphiu_reduced

# delphikon calculation

eps_kon = 1e-3

H0 = transitionMatrixGen(u,kon-eps_kon,koff,L,target)
Heps = transitionMatrixGen(u,kon+eps_kon,koff,L,target)

l10,eig0 = eigGen(H0)[0][1],eigGen(H0)[1][:,1][nonTargetSites]
l1eps,eigEps = eigGen(Heps)[0][1],eigGen(Heps)[1][:,1][nonTargetSites]
dl1kon = (l1eps - l10)/2/eps_kon

phi0 = np.log(eig0)
phiEps = np.log(eigEps)

dphikon_reduced = (phiEps - phi0)/2/eps_kon

dphikon = np.zeros(L)
dphikon[nonTargetSites] = dphikon_reduced


# delphikoff calculation

eps_koff = 1e-3

H0 = transitionMatrixGen(u,kon,koff-eps_koff,L,target)
Heps = transitionMatrixGen(u,kon,koff+eps_koff,L,target)

l10,eig0 = eigGen(H0)[0][1],eigGen(H0)[1][:,1][nonTargetSites]
l1eps,eigEps = eigGen(Heps)[0][1],eigGen(Heps)[1][:,1][nonTargetSites]
dl1koff = (l1eps - l10)/2/eps_koff

phi0 = np.log(eig0)
phiEps = np.log(eigEps)

dphikoff_reduced = (phiEps - phi0)/2/eps_koff

dphikoff = np.zeros(L)
dphikoff[nonTargetSites] = dphikoff_reduced


tcheck = 0
respuj0 = 0
respkonj0 = 0
respkoffj0 = 0
l1 = eig_vals[1]

atempu,chitempu = 0,0
atempkon,chitempkon = 0,0
atempkoff,chitempkoff = 0,0
for i in range(L):
    obsi = i
    for j in range(L):
        if(i!=target and j!=target):                
            e1j = right_eig_vecs[:,1][j]
            duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
            matmulij =  (H+l1)[i,j] 

            atempu += obsi*matmulij*duj*e1j
            chitempu += matmulij*duj*e1j

            atempkon += obsi*matmulij*dkonj*e1j
            chitempkon += matmulij*dkonj*e1j

            atempkoff += obsi*matmulij*dkoffj*e1j
            chitempkoff += matmulij*dkoffj*e1j

print("Resp for j at t=0")
print(atempu - j_avg*chitempu)
print(atempkon - j_avg*chitempkon)
print(atempkoff - j_avg*chitempkoff)

atempu,chitempu = 0,0
atempkon,chitempkon = 0,0
atempkoff,chitempkoff = 0,0
for i in range(L):
    obsi = i**2
    for j in range(L):
        if(i!=target and j!=target):                
            e1j = right_eig_vecs[:,1][j]
            duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
            matmulij =  (H+l1)[i,j] 

            atempu += obsi*matmulij*duj*e1j
            chitempu += matmulij*duj*e1j

            atempkon += obsi*matmulij*dkonj*e1j
            chitempkon += matmulij*dkonj*e1j

            atempkoff += obsi*matmulij*dkoffj*e1j
            chitempkoff += matmulij*dkoffj*e1j

print("\nResp for j2 at t=0")
print(atempu - j2_avg*chitempu)
print(atempkon - j2_avg*chitempkon)
print(atempkoff - j2_avg*chitempkoff)


atempu,chitempu = 0,0
atempkon,chitempkon = 0,0
atempkoff,chitempkoff = 0,0
for i in range(L):
    obsi = i**3
    for j in range(L):
        if(i!=target and j!=target):                
            e1j = right_eig_vecs[:,1][j]
            duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
            matmulij =  (H+l1)[i,j] 

            atempu += obsi*matmulij*duj*e1j
            chitempu += matmulij*duj*e1j

            atempkon += obsi*matmulij*dkonj*e1j
            chitempkon += matmulij*dkonj*e1j

            atempkoff += obsi*matmulij*dkoffj*e1j
            chitempkoff += matmulij*dkoffj*e1j
            
ttheoryj = np.linspace(0,0.1,50)
respFuncu_j = np.zeros(ttheoryj.size)
respFunckon_j = np.zeros(ttheoryj.size)
respFunckoff_j = np.zeros(ttheoryj.size)
l1 = eig_vals[1]


for ti in range(ttheoryj.size):
    atempu,chitempu = 0,0
    atempkon,chitempkon = 0,0
    atempkoff,chitempkoff = 0,0
    for i in range(L):
        obsi = i
        for j in range(L):
            if(i!=target and j!=target):                
                e1j = right_eig_vecs[:,1][j]
                duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
                matmulij =  np.matmul((H+l1),scipy.linalg.expm((H+l1)*ttheoryj[ti]))[i,j] 

                atempu += obsi*matmulij*duj*e1j
                chitempu += matmulij*duj*e1j

                atempkon += obsi*matmulij*dkonj*e1j
                chitempkon += matmulij*dkonj*e1j

                atempkoff += obsi*matmulij*dkoffj*e1j
                chitempkoff += matmulij*dkoffj*e1j

    respFuncu_j[ti] = atempu - j_avg*chitempu
    respFunckon_j[ti] = atempkon - j_avg*chitempkon
    respFunckoff_j[ti] = atempkoff - j_avg*chitempkoff

ttheorysurv = np.linspace(0,0.1,50)
respFuncu_surv = np.zeros(ttheorysurv.size)
respFunckon_surv = np.zeros(ttheorysurv.size)
respFunckoff_surv = np.zeros(ttheorysurv.size)
l1 = eig_vals[1]


for ti in range(ttheorysurv.size):
    atempu,chitempu = 0,0
    atempkon,chitempkon = 0,0
    atempkoff,chitempkoff = 0,0
    for i in range(L):
        for j in range(L):
            if(i!=target and j!=target):                
                e1j = right_eig_vecs[:,1][j]
                duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
                matmulij =  np.matmul((H+l1),scipy.linalg.expm((H+l1)*ttheorysurv[ti]))[i,j] 

                chitempu += matmulij*duj*e1j

                chitempkon += matmulij*dkonj*e1j

                chitempkoff += matmulij*dkoffj*e1j

    respFuncu_surv[ti] = (chitempu + dl1u)
    respFunckon_surv[ti] = (chitempkon + dl1kon)
    respFunckoff_surv[ti] = (chitempkoff + dl1koff)

dist_obs = (np.linspace(0,L-1,L) - target)**2
dist_avg = np.sum(dist_obs*right_eig_vecs[:,1])

print(dist_avg)

l1 = eig_vals[1]

atempu,chitempu = 0,0
atempkon,chitempkon = 0,0
atempkoff,chitempkoff = 0,0
for i in range(L):
    obsi = (i-target)**2
    for j in range(L):
        if(i!=target and j!=target):                
            e1j = right_eig_vecs[:,1][j]
            duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
            matmulij =  (H+l1)[i,j] 

            atempu += obsi*matmulij*duj*e1j
            chitempu += matmulij*duj*e1j

            atempkon += obsi*matmulij*dkonj*e1j
            chitempkon += matmulij*dkonj*e1j

            atempkoff += obsi*matmulij*dkoffj*e1j
            chitempkoff += matmulij*dkoffj*e1j

ttheoryjdist = np.linspace(0,0.1,50)
respFuncu_dist = np.zeros(ttheoryj.size)
respFunckon_dist = np.zeros(ttheoryj.size)
respFunckoff_dist = np.zeros(ttheoryj.size)
l1 = eig_vals[1]


for ti in range(ttheoryjdist.size):
    atempu,chitempu = 0,0
    atempkon,chitempkon = 0,0
    atempkoff,chitempkoff = 0,0
    for i in range(L):
        obsi = (i-target)**2
        for j in range(L):
            if(i!=target and j!=target):                
                e1j = right_eig_vecs[:,1][j]
                duj,dkoffj,dkonj = dphiu[j],dphikoff[j],dphikon[j]
                matmulij =  np.matmul((H+l1),scipy.linalg.expm((H+l1)*ttheoryjdist[ti]))[i,j] 

                atempu += obsi*matmulij*duj*e1j
                chitempu += matmulij*duj*e1j

                atempkon += obsi*matmulij*dkonj*e1j
                chitempkon += matmulij*dkonj*e1j

                atempkoff += obsi*matmulij*dkoffj*e1j
                chitempkoff += matmulij*dkoffj*e1j

    respFuncu_dist[ti] = atempu - dist_avg*chitempu
    respFunckon_dist[ti] = atempkon - dist_avg*chitempkon
    respFunckoff_dist[ti] = atempkoff - dist_avg*chitempkoff

### Convolution

@numba.njit(parallel=True)
def respIntegrate(respFunc,perturbation,tpoints):
    dobs = np.zeros(tpoints.size)
    for ti in prange(tpoints.size):
        t = tpoints[ti]
        integration = 0
        for taui in range(ti):
            tau = tpoints[taui]
            dtau = tpoints[taui+1] - tpoints[taui]
            integration += respFunc[ti-taui] * perturbation[taui] * dtau
        dobs[ti] = integration
    return dobs

temp = respIntegrate(respFunckoff_j,respFunckoff_j,ttheoryj)

## Perturbation : Simulation

@numba.njit
def newAlgo(tpoints=np.array([]),xtrack=np.array([]),ttrack=np.array([])):
    equalLength = 0
    for i in range(len(tpoints)):
        equalLength += len(np.where(tpoints[i] == ttrack)[0])
    
    newLength = len(tpoints) + len(ttrack) - equalLength

    newSamples = np.ones(newLength)*-1.0
    newTime = np.ones(newSamples.size)*-1.0

    samplingtrack = 0
    referencetrack = 0
    samplingIndices = []
    
    for i in range(newTime.size):
        if(samplingtrack == len(tpoints)):
            newTime[i] = ttrack[referencetrack]
            referencetrack +=1
        elif(referencetrack == len(ttrack)):
            newTime[i] = tpoints[samplingtrack]
            samplingIndices.append(i)
            samplingtrack +=1
        elif(ttrack[referencetrack] > tpoints[samplingtrack]):
            newTime[i] = tpoints[samplingtrack]
            samplingIndices.append(i)
            samplingtrack +=1
        elif(ttrack[referencetrack] < tpoints[samplingtrack]):
            newTime[i] = ttrack[referencetrack]
            referencetrack +=1
        else:
            newTime[i] = ttrack[referencetrack]
            samplingIndices.append(i)
            referencetrack+=1
            samplingtrack+=1

    referencetrack = 0
    for i in range(newTime.size):
        if(newTime[i] in ttrack):
            newSamples[i] = xtrack[referencetrack]
            referencetrack+=1
        else:
            newSamples[i] = newSamples[i-1]
    samplingIndices = np.array(samplingIndices)
    return newSamples[samplingIndices]

temp = newAlgo(np.linspace(0,1,10),np.random.randint(0,10,100),np.linspace(0,1,100))

@numba.njit(parallel=True)
def perturbedTrajGen(pt,T,u,kon,koff,L,target,tpoints,del_u,del_kon,del_koff,dt_pert):
    ntrials = pt.size
    perturbedTraj = np.ones((ntrials,tpoints.size))*-1
    for i  in prange(ntrials):
        t = 0
        tnew = 0
        j0 = pt[i]
        lastFilled = 0
        xtrack = [j0]
        ttrack = [t]
        while(t<T):            
            if(t==0):
                jnew = normal_evol(j0,u+del_u/dt_pert,kon+del_kon/dt_pert,koff+del_koff/dt_pert,L,target,dt_pert)
                tnew = t+dt_pert
                #print("done")
            else:
                dt_new,jnew = evol(j0,u,kon,koff,L,target)
                tnew = t+dt_new
            j0 = jnew
            t = tnew
            xtrack.append(j0)
            ttrack.append(t)
            if(t>=T):
                break
            if(j0 == target):
                break
        perturbedTraj[i] = newAlgo(tpoints,np.array(xtrack),np.array(ttrack))
    return perturbedTraj


@numba.njit(parallel=True)
def timeDependentPerturbation(pt,tintegrate,tpoints,u,kon,koff,L,target,del_u,del_kon,del_koff):
    ntrials = pt.size
    perturbedTraj = np.ones((ntrials,tpoints.size))*-1
    
    for i  in prange(ntrials):
        t = 0
        tnew = 0
        j0 = pt[i]
        lastFilled = 0
        finalCheck = 0

        for tintIndex in range(len(tintegrate)-1):
            dt_cur = tintegrate[tintIndex+1] - tintegrate[tintIndex]
            jnew = normal_evol(j0,u+del_u[tintIndex],kon+del_kon[tintIndex],koff+del_koff[tintIndex],L,target,dt_cur)
            tnew = t + dt_cur


            if(lastFilled == tpoints.size-1 and finalCheck ==0 and tpoints[lastFilled] <= t):
                perturbedTraj[i,lastFilled] = j0
                finalCheck = 1
            if(tpoints[lastFilled] == tnew):
                perturbedTraj[i,lastFilled] = jnew
                lastFilled +=1
            if(tpoints[lastFilled] < tnew and tnew< tpoints[lastFilled+1]):
                perturbedTraj[i,lastFilled] = j0
                lastFilled +=1

            
            j0 = jnew
            t = tnew
            
            if(j0 == target):
                perturbedTraj[i,lastFilled:] = j0
                break
    return perturbedTraj


##### Initializing perturbations functions through non-standard perturbations

dt_pert = 1e-4

tintegrate = np.arange(0,0.08+dt_pert,dt_pert)
tpoints = np.linspace(tintegrate[1],tintegrate[-2],25)

delt_u = np.ones(tintegrate.size)*u * 0
delt_u[delt_u.size//3:] = u/3.0 * 0
delt_u[2*delt_u.size//3:] = u/2.0 * 0

delt_kon = np.ones(tintegrate.size)*kon/10
delt_kon[delt_kon.size//2:] = 0

delt_koff = np.ones(tintegrate.size)*koff * 0
delt_koff[delt_koff.size//4:] = 0
delt_koff[delt_koff.size//2:] = koff*1.5 * 0
delt_koff[3*delt_koff.size//4:] = 0

temp = timeDependentPerturbation(removedpt[:10],tintegrate,tpoints,u,kon,koff,L,target,delt_u,delt_kon,delt_koff)

pertTracks = timeDependentPerturbation(removedpt[:1000000],tintegrate,tpoints,u,kon,koff,L,target,delt_u,delt_kon,delt_koff)

# avg pos

avg_pos_surv = np.zeros(tpoints.size)
avg_pos2_surv = np.zeros(tpoints.size)
avg_pos3_surv = np.zeros(tpoints.size)
avg_var = np.zeros(tpoints.size)
surv_traj = np.zeros(tpoints.size)

for ti in range(tpoints.size):
    survpositions = np.where(pertTracks[:,ti]!=target)[0]
    surv_traj[ti] = len(survpositions)

    avg_pos_surv[ti] = np.mean(pertTracks[:,ti][survpositions] - removedpt[:pertTracks.shape[0]][survpositions])
    avg_pos2_surv[ti] = np.mean((pertTracks[:,ti][survpositions])**2 - (removedpt[:pertTracks.shape[0]][survpositions])**2)
    avg_pos3_surv[ti] = np.mean((pertTracks[:,ti][survpositions])**3 - (removedpt[:pertTracks.shape[0]][survpositions])**3)
    avg_var[ti] = np.mean((pertTracks[:,ti][survpositions])**2) - np.mean(pertTracks[:,ti][survpositions]) **2

interp_fnu_surv = interp1d(ttheorysurv,respFuncu_surv,kind="cubic")
interp_fnkon_surv = interp1d(ttheorysurv,respFunckon_surv,kind="cubic")
interp_fnkoff_surv = interp1d(ttheorysurv,respFunckoff_surv,kind="cubic")

interp_respu_surv = interp_fnu_surv(tintegrate)
interp_respkon_surv = interp_fnkon_surv(tintegrate)
interp_respkoff_surv = interp_fnkoff_surv(tintegrate)

interp_fnu_dist = interp1d(ttheoryjdist,respFuncu_dist,kind="cubic")
interp_fnkon_dist = interp1d(ttheoryjdist,respFunckon_dist,kind="cubic")
interp_fnkoff_dist = interp1d(ttheoryjdist,respFunckoff_dist,kind="cubic")

interp_respu_dist = interp_fnu_dist(tintegrate)
interp_respkon_dist = interp_fnkon_dist(tintegrate)
interp_respkoff_dist = interp_fnkoff_dist(tintegrate)

interp_fnu = interp1d(ttheoryj,respFuncu_j,kind="cubic")
interp_fnkon = interp1d(ttheoryj,respFunckon_j,kind="cubic")
interp_fnkoff = interp1d(ttheoryj,respFunckoff_j,kind="cubic")

interp_respu = interp_fnu(tintegrate)
interp_respkon = interp_fnkon(tintegrate)
interp_respkoff = interp_fnkoff(tintegrate)

dobs_u = respIntegrate(interp_respu,delt_u,tintegrate)
dobs_kon = respIntegrate(interp_respkon,delt_kon,tintegrate)
dobs_koff = respIntegrate(interp_respkoff,delt_koff,tintegrate)

## Temperature Dependent Perturbation of Rates

def arrheniusRateLaw(const,Ea,Temp):
    gas_const = 8.31
    return const*np.exp(-Ea/gas_const/Temp)

Eu = 20*1e3
Ekon = 10*1e3
Ekoff = 40*1e3

initTemp = 300

constu = np.exp(np.log(u) + Eu/8.31/initTemp)
constkon = np.exp(np.log(kon) + Ekon/8.31/initTemp)
constkoff = np.exp(np.log(koff) + Ekoff/8.31/initTemp)

dt_pert = 5e-7

tintegrate = np.arange(0,0.05+dt_pert,dt_pert)
tpoints = np.linspace(tintegrate[1],tintegrate[-2],80)

finalTemp = 305

temperature = np.ones(tintegrate.size) * finalTemp

temperature[temperature.size//4:] = initTemp
temperature[temperature.size//2:] = finalTemp
temperature[3*temperature.size//4:] = initTemp

unew_t = arrheniusRateLaw(constu,Eu,temperature)
konnew_t = arrheniusRateLaw(constkon,Ekon,temperature)
koffnew_t = arrheniusRateLaw(constkoff,Ekoff,temperature)

deltemp_u = 1* (unew_t - u)
deltemp_kon =1*( konnew_t - kon)
deltemp_koff = 1*(koffnew_t - koff)

pertTracks_temp = timeDependentPerturbation(removedpt,tintegrate,tpoints,u,kon,koff,L,target,deltemp_u,deltemp_kon,deltemp_koff)

# avg pos

avg_pos_surv_temp = np.zeros(tpoints.size)
avg_surv = np.zeros(tpoints.size)

for ti in range(tpoints.size):
    survpositions = np.where(pertTracks_temp[:,ti]!=target)[0]
    avg_surv[ti] = len(survpositions)
    avg_pos_surv_temp[ti] = np.mean(pertTracks_temp[:,ti][survpositions] - removedpt[:pertTracks_temp.shape[0]][survpositions])

# avg dist

avg_dist_surv_temp = np.zeros(tpoints.size)

for ti in range(tpoints.size):
    survpositions = np.where(pertTracks_temp[:,ti]!=target)[0]
    avg_dist_surv_temp[ti] = np.mean((pertTracks_temp[:,ti][survpositions] - target)**2) #- (removedpt[:pertTracks_temp.shape[0]][survpositions] - target)**2)

interp_fnu_surv = interp1d(ttheorysurv,respFuncu_surv,kind="cubic")
interp_fnkon_surv = interp1d(ttheorysurv,respFunckon_surv,kind="cubic")
interp_fnkoff_surv = interp1d(ttheorysurv,respFunckoff_surv,kind="cubic")

interp_respu_surv = interp_fnu_surv(tintegrate)
interp_respkon_surv = interp_fnkon_surv(tintegrate)
interp_respkoff_surv = interp_fnkoff_surv(tintegrate)

interp_fnu_dist = interp1d(ttheoryjdist,respFuncu_dist,kind="cubic")
interp_fnkon_dist = interp1d(ttheoryjdist,respFunckon_dist,kind="cubic")
interp_fnkoff_dist = interp1d(ttheoryjdist,respFunckoff_dist,kind="cubic")

interp_respu_dist = interp_fnu_dist(tintegrate)
interp_respkon_dist = interp_fnkon_dist(tintegrate)
interp_respkoff_dist = interp_fnkoff_dist(tintegrate)

interp_fnu = interp1d(ttheoryj,respFuncu_j,kind="cubic")
interp_fnkon = interp1d(ttheoryj,respFunckon_j,kind="cubic")
interp_fnkoff = interp1d(ttheoryj,respFunckoff_j,kind="cubic")

interp_respu = interp_fnu(tintegrate)
interp_respkon = interp_fnkon(tintegrate)
interp_respkoff = interp_fnkoff(tintegrate)

dobs_u_temp = respIntegrate(interp_respu,deltemp_u,tintegrate)
dobs_kon_temp = respIntegrate(interp_respkon,deltemp_kon,tintegrate)
dobs_koff_temp = respIntegrate(interp_respkoff,deltemp_koff,tintegrate)

dobs_u_dist_temp = respIntegrate(interp_respu_dist,deltemp_u,tintegrate)
dobs_kon_dist_temp = respIntegrate(interp_respkon_dist,deltemp_kon,tintegrate)
dobs_koff_dist_temp = respIntegrate(interp_respkoff_dist,deltemp_koff,tintegrate)

dobs_u_surv = respIntegrate(interp_respu_surv,deltemp_u,tintegrate)
dobs_kon_surv = respIntegrate(interp_respkon_surv,deltemp_kon,tintegrate)
dobs_koff_surv = respIntegrate(interp_respkoff_surv,deltemp_koff,tintegrate)

### Convolution and Control

desiredPerturbation = - 2*np.sin(150*tintegrate)* np.exp(-250*tintegrate) - 2*np.sqrt(tintegrate)*np.sin(50*tintegrate)#* np.exp(-100*tintegrate)
longerDesiredPerturbation = - 2*np.sin(150*longerTimeMesh)* np.exp(-250*longerTimeMesh) - 2*np.sqrt(longerTimeMesh)*np.sin(50*longerTimeMesh)#* np.exp(-100*longerTimeMesh)

longerRespInterp = interp1d(ttheoryj,respFunckoff_j)
longerRespMesh = longerRespInterp(longerTimeMesh)

ft_freq =  np.fft.fftfreq(longerTimeMesh.size,d=(longerTimeMesh[1]-longerTimeMesh[0]))* np.pi * 2
ft_respkoff = np.fft.fft(longerRespMesh)*(longerTimeMesh[1]-longerTimeMesh[0])

ft_desiredPert = np.fft.fft(longerDesiredPerturbation)*(longerTimeMesh[1]-longerTimeMesh[0])

ft_neededPert = -ft_desiredPert/ft_respkoff

ifft_respkoff_dist = np.fft.ifft(ft_respkoff)/(longerTimeMesh[1]-longerTimeMesh[0])
ifft_desiredPert = np.fft.ifft(ft_desiredPert)/(longerTimeMesh[1] - longerTimeMesh[0])
ifft_neededPert = np.fft.ifft(ft_neededPert) / (longerTimeMesh[1] - longerTimeMesh[0])

ifft_neededPert[0] = 2*(ifft_neededPert[1] - ifft_neededPert[2])/(longerTimeMesh[1] - longerTimeMesh[2]) * (longerTimeMesh[0] - longerTimeMesh[1]) + ifft_neededPert[1]

shorterNeededPerturbation = np.real(ifft_neededPert)[:tintegrate.size]
act_resp = respIntegrate(interp_respkoff,shorterNeededPerturbation,tintegrate)

testpert = timeDependentPerturbation(removedpt,tintegrate,tpoints,u,kon,koff,L,target,np.zeros(tintegrate.size),np.zeros(tintegrate.size),shorterNeededPerturbation)

# avg pos

avg_pos_surv_test = np.zeros(tpoints.size)

for ti in range(tpoints.size):
    survpositions_test = np.where(testpert[:,ti]!=target)[0]
    avg_pos_surv_test[ti] = np.mean(testpert[:,ti][survpositions_test] - removedpt[:testpert.shape[0]][survpositions_test])


## First passage time distribution

l1 = eig_vals[1]

perturbation_strength = np.linspace(0.05,0.1,10) * koff
delta_tau_avg = np.zeros(perturbation_strength.size)

sumxy = 0
sumx = 0
for x in range(L):
    for y in range(L):
        if(x!=target and y!=target):
            eigenvector_contribution = 0
            for k in range(1,L):
                lk = eig_vals[k]
                eigenvector_contribution += (left_eig_vecs[:,k][y] * right_eig_vecs[:,k][x])/lk
            sumxy += -l1 * eigenvector_contribution * right_eig_vecs[:,1][y] * dphikoff[y]
    sumx += dphikoff[x] * right_eig_vecs[:,1][x]

const_term = (sumx + l1*dl1koff - sumxy)
delta_tau_avg = perturbation_strength * (sumx + l1*dl1koff - sumxy)

@numba.njit(parallel=True)
def absorptionTime(ntrials,j0array,u,kon,koff,L,target):
    pt = np.zeros(ntrials)
    for i in prange(ntrials):
        j = j0array[i]
        t = 0
        while(True):
            dt,jnew = evol(j,u,kon,koff,L,target)
            t = t+dt
            j = jnew
            if(j==target):
                pt[i] = t
                break        
    return pt
    

temp = absorptionTime(2,np.array([0]),u,kon,koff,L,target)

nabstrials = removedpt.size 
j0array = removedpt[:nabstrials]

absTimeDist = absorptionTime(nabstrials,j0array,u,kon,koff,L,target)

@numba.njit(parallel=True)
def perturbedAbsTime(pt,u,kon,koff,L,target,del_u,del_kon,del_koff,dt_pert):
    ntrials = pt.size
    perturbedAbs = np.ones(ntrials)*-1
    for i  in prange(ntrials):
        t = 0
        tnew = 0
        j0 = pt[i]
        while(True):            
            if(t==0):
                jnew = normal_evol(j0,u+del_u/dt_pert,kon+del_kon/dt_pert,koff+del_koff/dt_pert,L,target,dt_pert)
                tnew = t+dt_pert
            else:
                dt_new,jnew = evol(j0,u,kon,koff,L,target)
                tnew = t+dt_new
            j0 = jnew
            t = tnew
            if(j0 == target):
                perturbedAbs[i] = t
                break
    return perturbedAbs


temp = perturbedAbsTime(removedpt[:10],u,kon,koff,L,target,u/10,kon/10,koff/10,1e-6)

dt_pert = 1e-6

del_u = 0
del_kon = 0
del_koff = 0.1

perturbedAbsDist = perturbedAbsTime(removedpt,u,kon,koff,L,target,del_u,del_kon,del_koff,dt_pert)

t_abs_linspace = np.linspace(np.min(perturbedAbsDist),1,100)
tau_t = np.zeros(t_abs_linspace.size)

for ti in range(t_abs_linspace.size):
    t = t_abs_linspace[ti]

    expmatrix = scipy.linalg.expm((H+l1)*t)
    matrixmul = np.matmul((H+l1),expmatrix)
    matrixmul2 = np.matmul((H+l1),matrixmul)

    term1 = np.sum(np.matmul(matrixmul,dphikoff*right_eig_vecs[:,1])[nonTargetSites])
    term2 = np.sum(np.matmul(matrixmul2,dphikoff*right_eig_vecs[:,1])[nonTargetSites])
    
    tau_t[ti] = -l1*np.exp(-l1*t) *term1 - np.exp(-l1*t) * term2 + l1*dl1koff*np.exp(-l1*t)

tmin = np.min([np.min(absTimeDist),np.min(perturbedAbsDist)])
tmid = t_abs_linspace[-1]
tmax = np.max([np.max(perturbedAbsDist),np.max(absTimeDist)])

tbins = np.linspace(tmin,tmid,50)
tbins = np.concatenate((tbins,np.array([tmid*1.1,tmax])))

originalHist,edges = np.histogram(absTimeDist,bins=tbins,density=True)
perturbedHist,edges = np.histogram(perturbedAbsDist,bins=tbins,density=True)

plotBins = (tbins[1:]+tbins[:-1])/2.0
binwidth = tbins[1:] - tbins[:-1]

flatline = np.mean(originalHist[:-2])