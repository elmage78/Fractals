import numpy as np
import sympy as sp
from manimlib import *
from sympy import Symbol,Derivative,simplify,lambdify
import matplotlib.pyplot as plt

#Definimos a función cuxas raíces queremos aproximar (cada grupo usará unha función distinta):

z=Symbol('z')
f=(z**3-1)**(1/z)


#Definimos a derivada da función anterior. Se fose necesario para algún método
#definiríase de forma similar a segunda derivada.
print("determining derf")
derf=Derivative(f,z,1).doit()
print("determining dderf")
dderf=Derivative(f,z,2).doit()

#Definimos a fórmula do método que queremos implementar. Neste exemplo 
#tratase do Método de Newton. Para implementar outro metodo debemos definir 
# a fórmula correspondente:

#Metodo de ***
print("determining Chebysev")
g=simplify(z-(f/derf)-((f/derf)**2)*(dderf/(2*derf)))

# Os seguintes parámetros poden ser modificados co obxetivo de 
#conseguir os mellores gráficos posibles: 

# Número máximo de iteracións para o método elexido:

maxiter=30

#O fractal representarase no rectángulo [a,b]x[c,d]: 

a=-5
b=5
c=-3
d=3

#Número de puntos usados nos eixos OX e OY para representar o fractal: canto
#maior sexa o número de puntos máis preciso sera o gráfico, pero tamén máis
#tempo se necesitará para levar a cabo os cálculos.

npuntos=400

################### NON MODIFICAR ESTA PARTE DO PROGRAMA #####################
##############################################################################

print("lambdifying")
ff=lambdify(z,f,"numpy")
gg=lambdify(z,g,"numpy")
fractal = np.zeros((npuntos+1,npuntos+1))
tol=1.0e-6

x=np.linspace(a,b,npuntos+1)
y=np.linspace(c,d,npuntos+1)

for i in range(0,npuntos):
    for j in range(0,npuntos):
        if ((i+j)%100):
            print ("going...")
        z = complex(x[i],y[j])
        n=0
        try:
            abs(gg(z))
            abs(ff(z))
        except ZeroDivisionError:
            fractal[npuntos-j,i]=float(0) 
            continue

        while (n<maxiter and abs(ff(z))>tol):
            if abs(gg(z))<1/tol: # si el numero se va al infinito sale con x iteraciones (cuanto tarda en ir al infinito)
               z=gg(z) # si llega a 30 iteraciones podemos aproximar su convergencia
               n=n+1
            else:
               break
        fractal[npuntos-j,i]=float(n)  

##############################################################################
##############################################################################

print('MÉTODO DE NEWTON')       
print('A función utilizada é: f(z)=',f)
print('A súa derivada é: derf(z)=',derf)
print('A función do método de Newton é: g(z)=',g)

#A continuación represéntase a imaxen fractal: recoméndase buscar unha gama de 
#cores atractiva. 

plt.imshow(fractal,cmap='spring', extent=(a, b, c, d))
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")

#A continuación gárdase a imaxen do fractal nun ficheiro: cambiar o nome do 
#ficheiro según o método usado 

plt.savefig('fractal_Chebysev.png', dpi=2000)
plt.show()