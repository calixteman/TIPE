#Alberic de La Villegeorges :  albericdelavillegeorges@gmail.com
##D'apres "the contact of elastic regular wavy surface" de K.L Johnson and al.
#contact d'un surface sinusoidale et d'un plan rigide en dimension une
#calculs de la distribution de pression et du deplacement du solide

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt

import matplotlib

# nice math font
matplotlib.rcParams['mathtext.fontset'] = 'cm'


K = [0.1,0.2,0.4,0.6,0.8,1]         #valeurs de p(barre)/p*
delta = 20                          #amplitude de l'onde (micromètres)
abscisse = 1000                     #nombre de points en abscisses
n = 10                              #nombre de termes de la série de Fourier
m = 26                              #nombre de containtes



def resolution_quadratique(p2):         #methode de resolution quadratique pour determiner les coefficients de la decompositon de Fourier avec m contraintes (6 valeurs de phi entre 0 et pi)
                                        #p2 = p(barre)/p*
    #création de P :
    def mat_P():
        L = []
        for i in range(1,n+1): L.append(1/i)
        return(L)
    P = matrix(np.diag(mat_P()))

    # plus simple
    # P = matrix(np.diag(1 / np.arange(1, n + 1)))
    
    #création de q:
    q = matrix(-np.eye(n,1))
    
    #création de G:
    def mat_G():                                #on creer une grande liste qu'on reforme en matrice apres 
        phi = np.linspace(0, np.pi, m)
        L = []
        for i in range(0,m):
            for j in range(1,n+1):
                L.append(-np.cos(j*phi[i]))
        return(L)
                         
    G = matrix(np.reshape(mat_G(),(m,n)))

    # phis = np.array([np.linspace(0, np.pi, m)]).T
    # js = np.arange(1, n + 1)
    # G = matrix(-np.cos(phis * js))
    
    #création de h:
    h = matrix(p2*np.ones((m,1),float))
    
    #solution:
    sol = solvers.qp(P,q,G,h)
    A = np.array(sol['x'])
    return(A)



#creation de la matrice de l'ensemble des solutions
matrice_solution = []                                           #creation d'une matrice des solutions de la resolution quadratique pour les differentes valeurs de p2
for p2 in K:                                                    #boucle pour chaques valeurs de p2                             
    matrice_solution.append([resolution_quadratique(p2)])       #y'a une petite bizarerie, il affiche le 'array' 







####tracé de la distribution de pression
def pressure_distribution():                             

    for h in range(0,len(K)):
        p2 = K[h]
        X = np.linspace(0,1,abscisse)                    #points de l'abscisse
        Y = []                                           #liste des ordonnees
        
        for x in X:                                      #calcul des ordonees                                            
            s = 0
            for i in range(0,n):
                s += matrice_solution[h][0][i]*np.cos((i+1)*x*np.pi)     #le [0] est pour enlever l'anomalie du 'array' vu precedement
            Y.append(s + p2)
    
        plt.plot(X,Y, label=r"$\frac{\bar{p}}{p^*} = " + str(p2) + "$")                #formalisation graphique
        plt.legend()

        plt.xlabel(r'$\frac{\varphi}{\pi}$', fontsize=15)
        plt.ylabel(r'$\frac{p}{p^*}$', fontsize=15)
        plt.title('distribution de la pression, solution analytique, Weestergaard')
        plt.ylim(-0.03,2.0)
        plt.xlim(0.0,1.0)

pressure_distribution()
plt.show()
        
        
####trace de la variation de la la separation moyenne separant les deux surfaces en fonction de la pression moyenne   
def variation_separation():                         
    
    X1 = np.linspace(0.0001,1,abscisse)             #listes des abscisses
    X2 = np.linspace(0.1,0.9,len(K))
    Y1, Y2 = [], []                                 #listes des ordonnees
    
    for x in X1 :                                   #calcul des ordonnees methode Kuznetsov
        y = 1 - x*(1-np.log(x))
        Y1.append(y)
    
     
    for h in range(0,len(K)) :                      #calcul des ordonnes methode numerique
        s = 0
        for i in range(0,n):
            s += matrice_solution[h][0][i]/(i+1)
        Y2.append(1 - s)
    
        
    plt.plot(X1,Y1, label = 'kuznetsov')            #formalisation de trace
    plt.scatter(X2,Y2, label = 'solution numerique', color = 'darkorange', marker = '+')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('pression moyenne p-/p*')
    plt.ylabel('separation moyenne G = g-/delta')
    plt.title('Variation de la separation moyenne en fonction de la pression moyenne')








   
####deplacement uz pour p2 = 0.4
def deplacement():                                                   
    
    X = np.linspace(0,1,abscisse)                                   #liste des abscisses            
    Y = []                                                          #liste des ordonnees    
    
    for x in X:                                                     #calcul des ordonnees 
        s = 0
        for i in range (0,n):
            s += (matrice_solution[2][0][i]/(i+1)) * np.cos((i+1)*x*np.pi)      #utilisation de la matrice solution pour p2 = 0.4, bizzarerie de array ==> [0]
        Y.append(s*delta)
        
    plt.plot(X,Y, label="p-/p* = {}".format(p2))                    #formalisation graphique
    plt.legend()
    plt.xlim(0.0,1.0)
    plt.xlabel('phi/pi')
    plt.ylabel('Uz')
    plt.title("déplacement de la surface suivant l'axe des z")
    
    
    
    


