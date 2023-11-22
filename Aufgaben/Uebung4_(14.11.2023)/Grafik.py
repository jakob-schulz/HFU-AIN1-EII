import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 4 * np.pi, 200) 
#Angabe der Definitionswerte. Bei x handelt es sich nach diesem Befehl wahrscheinlich um einen Array.
#np.linespace erstellt eine gleichmaesig verteilte Squenz von Zahlen. 
#0-> Beginn der Sequenz, niedrigster x-Wert
#4*np.pi -> Ende der Sqeuenz, hoechster x-Wert
#200 -> Anzahl der gewuenschten Werte in der Sequenz

y = 10*np.square(np.sin(x)) 
#Berechnen der Funktionswerte, Funktionsterm wurde vorgegeben.

qsin, ax = plt.subplots() 
#Die Funktion erstellt eine neue grafische Darstellung, was hier als Figur bezeichnet wird. Diese Figur wird an qsin uebergeben
#Zudem erstellt die Funktion eine Axe, welche in ax abgespeichet wird.

qsin.suptitle('f(x)= 10*sin(x)^2')
ax.set_xlabel('x-Achse')
ax.set_ylabel('y-Achse')
#Beschriftungen

ax.plot(x, y) #Graph wird mit angegeben x- und y- Werten erstellt 
plt.show() #Graph wird gezeigt

