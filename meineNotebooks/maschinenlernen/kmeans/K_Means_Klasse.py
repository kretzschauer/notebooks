import numpy as np
import random
import matplotlib.pyplot as plt
#https://github.com/gkabbe/Python-Kurs2015/wiki/5---Numpy
#Berechnet den Massenmittelpunkt
#https://mubaris.com/posts/kmeans-clustering/

class k_Means():
    """
        Erhaelt eine Punkt-Liste als Array
        die Anzahl der Clusterpunkte
        eventuell die Cluster-Liste, wenn nicht,
        werden die Cluster-Punkte zufaellig aus den
        Punkten der PL bestimmt
    """
    def __init__(self, punkte, anzahl, clLi=None):        
        #print(self.clLi)
        self.puLi = punkte
        self.anzahlCluster = anzahl
        self.laengePuLi, self.dimPunkte =self.puLi.shape
        if clLi:
            self.clLi=clLi
        else:
            self.bestimmeCL()
        print("StartCluster: \n", self.clLi)
        self.cluster= np.zeros((self.laengePuLi,),dtype=int)
        self.error=20
        self.cls=[]
        
    def bestimmeCL(self):
        #zufaellige Auswahl der Clusterpunkte aus den gegebenen
        self.clLi=np.zeros((self.anzahlCluster,self.dimPunkte),dtype=int)
        clListe=[]
        i=0
        while len(clListe) < self.anzahlCluster:
            index= random.randint(0,self.laengePuLi-1)
            cl=self.puLi[index]
            if not self.istEnthalten(clListe,cl): #vermeide Doppelte
                clListe.append(cl)
                self.clLi[i]=self.puLi[index]
                i+=1
                
    def istEnthalten(self,array,el):
        for e in array:
            if (e==el).all():
                return True
        return False
        
    def einteilRunde(self):        
        for i in range(self.laengePuLi):
            #print("Abstaende: ",self.clLi,self.puLi[i])
            abstand=np.linalg.norm(self.clLi-self.puLi[i],axis=1)
            #print(abstand)
            cluster = np.argmin(abstand)
            #print(cluster)
            self.cluster[i]=cluster        
        self.clLi_old=np.copy(self.clLi)
        self.cls.append(self.clLi_old)
        for i in range(self.anzahlCluster):            
            punkte= [self.puLi[j] for j in range(self.laengePuLi) if self.cluster[j]==i]                 
            self.clLi[i]=np.mean(punkte,axis =0)
        print("Neue Cluster: \n", self.clLi)
        self.error = np.linalg.norm(self.clLi-self.clLi_old)
        print(self.error)
        
    def einteilen(self,bis=0):           
        self.einteilRunde()
        while self.error> bis:            
            self.einteilRunde()            
            self.gibZahlen()
            
    def gibZahlen(self):
        print("Verteilung der Punkte auf die Cluster")
        #liefert die Anzahlen der Punkte je Cluster
        for i in range(self.anzahlCluster):            
            punkte= [self.puLi[j] for j in range(self.laengePuLi) if self.cluster[j]==i]
            print(i,len(punkte))
        
    def zeigen(self):
        fig = plt.figure()
        fig.suptitle("Einteilung in Cluster")
        ax = fig.add_subplot(111)        
        farben=["red","yellow","blue","green","lightblue","grey"]*2
        m=["+","x","*","o","v"]*20
        for i in range(self.anzahlCluster):
            xs= [self.puLi[j][0] for j in range(self.laengePuLi) if self.cluster[j]==i]
            ys= [self.puLi[j][1] for j in range(self.laengePuLi) if self.cluster[j]==i]
            #print("xs: ",xs," ys: ",ys)            
            ax.scatter(xs,ys, c=farben[i])        
        si=-1#Markerzaehler
        for c in self.cls:            
            si+=1
            for i in range(self.anzahlCluster):
                ax.scatter(c[i][0],c[i][1], c="black",marker=m[si])                      
        #for c in self.clLi:
            #ax.scatter(c[0],c[1], c="black",marker=m[0])
        maxis=np.amax(self.puLi,0)
        minis=np.amin(self.puLi,0)
        print("Maxis ",maxis, " Minis: ",minis)
        plt.axis([int(minis[0])-1,int(maxis[0])+1,
                  int(minis[1])-1,int(maxis[1])+1])
        plt.show()
               
def csv2array(datei):
    # liest eine csv-Datei ein und liefert ein PunktArray
    arr=np.loadtxt(datei,delimiter=",",skiprows=1,usecols=(1,2))
    return arr                    
    


if __name__=="__main__":
    punktliste= [[2,12],[3,11],[3,8],[5,4],[7,5],[7,3],[10,8],[13,8]]
    puAr=np.array(punktliste,dtype=int)
    
    puAr=csv2array("xclara.csv")
    print(puAr)
    k=k_Means(puAr,6)#listeClusterPunkte)
    k.einteilen()
    k.zeigen()