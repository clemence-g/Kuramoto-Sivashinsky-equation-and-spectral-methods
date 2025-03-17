# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:50:29 2024

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.fft as sc
from matplotlib import animation
from IPython.display import HTML


#############################
###### PREMIERE PARTIE ######
#############################

def fourier(nu,l):
    #CONSTANTES
    #Paramètres spatiaux
#    N=1024#discrétisation du domaine
    N= 64
    dx = l/N
    x = np.arange(0,l,dx)
    k = np.arange(-N/2,N/2,1) #nombre d'onde
    
    #Paramètres temporels
    tmax=200 #temps maximum de la simulation
    dt=0.05 #pas de temps
    M=int(tmax/dt)+1 #Nombre de pas de temps
    t = np.linspace(0,tmax,M)
    
    #1: TRANSFORMEE DE FOURIER  
    
    #Condition initiale : u(x,t=0) = u0
    u0 = np.cos(2*np.pi*x/l) + 0.1*np.cos(4*np.pi*x/l)
    
    #Valeurs dans l'espace physique
    u = np.zeros((N,M)) #u(x,t)
    u[:,0] = u0 #u(x,t=-dt) = u0
    u[:,1] = u0 #u(x,t=0) = u0
    
    #Stockage des coef de Fourier 
    u_k = np.zeros((N,M),dtype='complex') #F(u(x,t))
    
    u_k[:,0] = sc.fftshift(sc.fft(u[:,0])) #F(u(x,t=-dt))
    u_k[:,1] = sc.fftshift(sc.fft(u[:,1])) #F(u(x,t=0))
    
    u_k2 = np.zeros((N,M),dtype='complex') #F(u**2(x,t))
    u_k2[:,0] = sc.fftshift(sc.fft(u[:,0]**2)) #F(u**2(x,t=-dt))
    
    ##2: EVOLUTION TEMPORELLE

    fl = (2*np.pi*k/l)**2 - nu*(2*np.pi*k/l)**4 #transformée de Fourier de L

    a = 1+(dt/2)*fl
    b = 1-(dt/2)*fl
    c = dt*1j*np.pi*k/l

    for i in range(1,M-1): #boucle qui itère jusqu'au temps final

        u_k2[:,i] = sc.fftshift(sc.fft(u[:,i]**2))
        u_k[:,i+1] = (a/b)*u_k[:,i] - c*(1.5*(u_k2[:,i]) - 0.5*(u_k2[:,i-1]))/b
        
        #3: ESPACE PHYSIQUE
        u[:,i+1]=np.real(sc.ifft(sc.ifftshift(u_k[:,i+1])))
        
    return x,t,u,l,nu
    

#animation


U_list = []

L_min = 1 #ne pas mettre 0 sinon ça ne marche pas
L_max = 40
dl = 0.1
L_list = np.arange(L_min,L_max,dl)

for li in L_list:
    X,T,U,L,nu = fourier(nu=2,l=li)
    U_list.append(U)
   
xx,tt = np.meshgrid(X,T)
fig,ax = plt.subplots()

#dim = np.linspace(np.min(np.min(U_list[-1])),np.max(np.max(U_list[-1])),100) idée mais c'est pas suffisant
dim = np.linspace(-3,3,100) #ça a l'air pas mal comme ça mais pas sûre

contU0 = ax.contourf(xx,tt,U_list[0].T,dim,cmap=cm.jet)
plt.colorbar(contU0)

def init():
    return contU0

def update(frame):
    ax.clear()
    contour = ax.contourf(xx,tt,U_list[frame].T, dim, cmap=cm.jet)
    ax.set_title(f"L = {round(L_list[frame],2)} \n\u03BD = {nu}")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return contour,

anim = animation.FuncAnimation(fig, update, frames=len(U_list), interval=200, blit=False)
HTML(anim.to_html5_video())

FFwriter = animation.FFMpegWriter(fps=10)
anim.save(filename= f'Kuramoto-{L_max}-nu={nu}.mp4', writer=FFwriter)

#Note pour Clem : Animation est stockée dans Ce PC -> Windows SSD -> Utilisateurs -> User
