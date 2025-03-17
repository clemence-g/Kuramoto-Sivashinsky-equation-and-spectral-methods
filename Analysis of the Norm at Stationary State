# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:43:52 2024

@author: Elodie Sorée & Clémence Georges

Code du projet de méthodes spectrales sur l'équation de Kuramoto-Sivashinsky

/!\ La fonction Fourier rend le code très long à exécuter à cause des fonctions fft.
Ce code permet d'obtenir des graphiques en 2d de u(x,t), de A(L) et de L_nu(nu)
ainsi qu'une animation de u(x,t) avec différentes longueurs maximale.
Il est conseillé de mettre les graphiques dont on n'a pas besoin en commentaires (#)
pour réduire le temps d'exécution du code.

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

def fourier(nu,l,tmax):
    '''
    Calcule les solutions de l'équation grâce aux méthodes spectrales
    
    Parameters
    ----------
    nu : Viscosité.
    l : Longueur maximale du domaine
    tmax : Temps maximal considéré pour l'évolution du système

    Returns
    -------
    x : Domaine spatial discrétisé
    t : Domaine temporel
    u : Solution de l'équation
    l : Longueur maximale du domaine
    nu : Viscosité

    '''
    #CONSTANTES
    #Paramètres spatiaux
    N=1024 #Nombre de points dans le domaine
    dx = l/N
    x = np.arange(0,l,dx)
    k = np.arange(-N/2,N/2,1) #nombre d'onde

    #Paramètres temporels
    dt= 0.05 #pas de temps
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
    
#4: GRAPHIQUE
def graph(xxx,ttt,UUU,LLL,NU):
    '''

    Parameters
    ----------
    XXX : Domaine spatial discrétisé
    TTT : Domaine temporel
    UUU : Solution de l'équation
    LLL : Longueur maximale du domaine
    NU : Viscosité

    Returns
    -------
    None.

    '''
    xx,tt = np.meshgrid(xxx,ttt)

    fig,ax=plt.subplots()

    dim = np.linspace(np.min(np.min(UUU)),np.max(np.max(UUU)),100)
    contU=ax.contourf(xx,tt,UUU.T,dim,cmap=cm.jet)
    
    ax.set_title(f"Kuramoto-Sivashinsky L = {LLL} nu={NU}")
    plt.colorbar(contU)
    plt.xlabel("x")
    plt.ylabel("t")
    

#On appelle les fonctions en introduisant dans Fourier les paramètres constants

X,T,U,L,NU = fourier(nu=1,l=100,tmax=200)
graph(X,T,U,L,NU)

X,T,U,L,NU = fourier(nu=1,l = 2,tmax=200)
graph(X,T,U,L,NU)

X,T,U,L,NU = fourier(nu=1,l=6,tmax=200)
graph(X,T,U,L,NU)

X,T,U,L,NU = fourier(nu=2,l=12,tmax=200)
graph(X,T,U,L,NU)


#############################
###### DEUXIEME PARTIE ######
#############################

def A(l, nu):
    '''
    Génère la formule pour calculer A, la norme de la solution à l'état stable
    '''
    #On dit qu'une limite suffisamment grande est tmax=200,
    X,T,U,L,NU = fourier(nu=nu,l=l, tmax = 200)
    
    #puisque le système est discrétisé, on peut remplacer l'intégrale par une somme de Riemann
    somme_U2 = np.sum(U[:,-1]**2)
    
    A2 = 1/l * somme_U2
    A = np.sqrt(A2)/1024
    
    return A

def graphA(L_max, dl, nu):
    '''
    Génère un graphique de A en fonction de L
    '''
    L_list = np.arange(1, L_max,dl)
    
    #Boucle qui va calculer A pour différentes valeurs de L_list et les stocker dans dataA
    dataA = []

    for li in L_list:
        dataA.append(A(l=li, nu=nu))
        
    #graphique    
    plt.plot(L_list, dataA, label = f"nu={nu}")
    plt.title("A en fonction de L")
    plt.xlabel("L")
    plt.ylabel("A")
    plt.legend()
    plt.xticks(np.arange(1,L_max,5))
    plt.show()

plt.subplots()
graphA(L_max = 30, dl = 0.5, nu = 1) #L_nu = 6.25
graphA(L_max = 30, dl = 0.5, nu = 2) #8.75
graphA(L_max = 30, dl = 0.5, nu = 3) #10.75

#On essaye avec plus de valeurs de nu pour pouvoir trouver une loi de L_nu en fonction de nu
graphA(L_max = 20, dl = 0.5, nu = 4) #11.75
graphA(L_max = 20, dl = 0.5, nu = 5) #13.25
graphA(L_max = 20, dl = 0.5, nu = 6) #14.25
graphA(L_max = 20, dl = 0.5, nu = 7) #15.25
graphA(L_max = 20, dl = 0.5, nu = 8) #16.25
graphA(L_max = 20, dl = 0.5, nu = 9) #17.25
graphA(L_max = 20, dl = 0.5, nu = 10) #18.5
graphA(L_max = 25, dl = 0.5, nu = 15) #21
graphA(L_max = 25, dl = 0.5, nu = 20) #23.25
graphA(L_max = 30, dl = 0.5, nu = 30) #26.25
graphA(L_max = 40, dl = 0.5, nu = 40) #30
graphA(L_max = 40, dl = 0.5, nu = 50) #33
graphA(L_max = 60, dl = 0.5, nu = 70) #35.5

#Valeurs des nu testées
list_nu = [1,2,3,4,5,6,7,8,9,10, 15, 20, 30, 40, 50, 70]

#Valeurs des L_nu trouvées approximativement sur le graphique de A en fonction de L
list_L_nu = [6.25, 8.75, 10.75, 11.75, 13.25, 14.25, 15.25, 16.25, 17.25, 18.5, 21, 23.25, 26.25, 30, 33, 35.5]

#On pense que l'équation est de la forme a*nu**b
#Par essai/erreur:
list_nu_racine_1 = [6.9*nu**(0.39) for nu in list_nu]

#graphique
plt.subplots()
plt.scatter(list_nu, list_L_nu, label = "Valeurs mesurées", color = "blue")
plt.plot(list_nu, list_nu_racine_1, label = "7*nu**(0.395)", color = "green")

#Pour être plus efficaces, on peut essayer par régression linéaire. Il faut appliquer un log des deux côtés
list_nu_log = np.log(list_nu)
list_L_nu_log = np.log(list_L_nu)

#Régression linéaire : l'équation est log(L_nu) = log(a) + b*log(nu) --> équation linéaire de la forme y = a + bx
coeffs = np.polyfit(list_nu_log, list_L_nu_log, 1)
b = coeffs[0]
log_a = coeffs[1]
a = np.exp(log_a)

print(f"Les coefficients sont a={a} et b ={b}")

#liste des valeurs calculées avec la régression:
list_nu_reg = [a*nu**b for nu in list_nu]

#Graphique qui se superpose au précédent pour mieux comparer la courbe des valeurs observées, de l'essai/erreur et de la régression linéaire
plt.plot(list_nu, list_nu_reg, label = f"{round(a,2)}*nu**{round(b,2)}", color = "red") #la fonction round sert à ce qu'il y ait seulement 2 chiffres après la virgule
plt.xlabel("nu")
plt.ylabel("L_nu")
plt.title("L_nu en fonction de nu")
plt.legend()
plt.show()


#############################
######### ANIMATION #########
#############################

U_list = [] #Permet de stocker les solutions de u(x,t) pour différnts L_max

L_min = 1 #Doit être différent de 0
L_max = 40
dl = 0.1
L_list = np.arange(L_min,L_max,dl)

for li in L_list:
    X,T,U,L,nu = fourier(nu=1,l=li) #Spécifier nu pour voir les différents résultats possibles
    U_list.append(U)
   
xx,tt = np.meshgrid(X,T)
fig,ax = plt.subplots()

dim = np.linspace(-3,3,100)
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

#Si besoin d'enregistrer l'animation.
#FFwriter = animation.FFMpegWriter(fps=10)
#anim.save(filename= f'Kuramoto-{L_max}-nu={nu}.mp4', writer=FFwriter)
