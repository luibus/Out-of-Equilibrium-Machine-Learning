# el objetivo es crear un proceso de difusion para datos 2D
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
import gc
from tqdm import tqdm

# barrido de memoria
gc.collect()
torch.cuda.empty_cache()

DEVICE =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Difusion:
    def __init__(self,data,W,paso_temp):
        self.DEVICE_dos =torch.device("cuda:0" if data.is_cuda==True else "cpu")#segun el device de la entrada me pone mi clase en cpu o en cuda tarda lo mismo o un segundo menos
        self.n=data.size(dim=0)                                                 #numero de puntos o variables
        self.m=data.size(dim=1)                                                 #numero de samples
        self.T = paso_temp                                                      #pasos de tiempo
        self.mu=0                                                               #media de el ruido que se añade en los procesos
        self.var=1                                                          #varianza del ruido
        self.data=data                                                          #dos definiciones de data para varios usos (unused)
        self.data2=data
        self.start=(1e-4)                                                       #inicio y final del difusion rate, se hace linspace para retrasar los efectos del proceso
        self.end=0.02 #0.02
        self.betas = self.linear_beta_schedule()                                #llamada interior que genera el linspace
        self.alphas = 1. - self.betas                                           #generación de las alphas
        self.alphas_cumprod = torch.from_numpy(np.cumprod(self.alphas, axis=0)).to(self.DEVICE_dos) #generación de las alphas acumulativas
        self.alpha_hat=self.alphas_cumprod                                      #cambio de nombre
    
    
    def salida_factor(self,i):
        if i>0:
          sal=self.alphas_cumprod[i-1]
        else:
          sal=None
        return self.alphas[i],self.alphas_cumprod[i],self.betas[i],sal

        
    def linear_beta_schedule(self):
        return np.linspace(self.start, self.end, self.T)

    
    def forward_alpha(self):                                                    # Paso forward con alphas acumulativas
        ruido=torch.normal(self.mu * torch.ones(self.m,self.n),self.var).to(self.DEVICE_dos)      
        data2 =torch.sqrt(self.alphas_cumprod[self.T-1])*self.data2.T + (torch.sqrt(1-self.alphas_cumprod[self.T-1])*ruido)
        return data2,ruido


    def forward_alpha_last_update(self,Time):                                   #proceso forward que se empleará para entrenar tiempo a tiempo cada máquina al completo
        ruido=torch.normal(self.mu * torch.ones(self.m,self.n),self.var).to(self.DEVICE_dos)
        data2 =torch.sqrt(self.alphas_cumprod[Time-1])*self.data2.T + (torch.sqrt(1-self.alphas_cumprod[Time-1])*ruido)
        return data2,ruido


    def reverse_sampling(self, W,ruido,model_dict):
            with torch.no_grad():
                x=torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)#sampling de una distribución normal
                predicted_noise=torch.ones(self.n).to(DEVICE)

                for i in reversed(range(self.T)):                               #conteo inverso desde T hasta 0
                      alpha = self.alphas[i]                                    #los alpha beta y alpha coump de el paso específico de la trayectoria reverse
                      alpha_hat = self.alphas_cumprod[i]
                      beta = self.betas[i]
                      model=model_dict[str(i)]                                  #seleccionamos el modelo del paso de tiempo, es un diccionario, entra como str
                      model.eval()                                              #se pone en modo evaluación
                      predicted_noise=model(x)                                  #en el primer paso partimos del ruido y generamos una predicción de x con el modelo, luego x se redefinirá
                      if i >= 1:                                                #en todos los pasos desde T hasta 1 se genera una variable aleatoria 
                          noise= torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)
                      else:                                                     #en 0 no se añade más ruido
                          noise = torch.zeros(self.n).to(DEVICE)
                      gamma=beta                                                #gamma mejor empleado para sampling desde distribución normal
                      #if i>0:
                      #  gamma=((1-self.alphas_cumprod[i-1])/(1-alpha_hat))*beta#mejor gamma para asignar nosotros el valor x_0
                      x =((1/np.sqrt(alpha))*(x-(((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)))+(np.sqrt(gamma)*noise)
                      model.train()

            return x
    def reverse_sampling_last_update(self, W,ruido,model_dict,Time):
            with torch.no_grad():
                x=torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)#sampling de una distribución normal
                x=ruido                                                         #al comentar esta linea lo hacemos tomar un ruido base aleatorio si no usa la muestra característica
                predicted_noise=torch.ones(self.n).to(DEVICE)

                for i in reversed(range(Time)):                                 #conteo inverso desde T hasta 0
                      alpha = self.alphas[i]                                    #los alpha, beta y alpha coump de el paso específico de la trayectoria reverse
                      alpha_hat = self.alphas_cumprod[i]
                      beta = self.betas[i]
                      model_aux=model_dict[str(i)]                              #seleccionamos el modelo del paso de tiempo, es un diccionario, entra como str
                      model.eval()                                              #se pone en modo evaluación
                      predicted_noise=model_aux(x)                              #en el primer paso partimos del ruido y generamos una predicción de x con el modelo, luego x se redefinirá

                      if i >= 1:                                                #en todos los pasos desde T hasta 1 se genera una variable aleatoria 
                          noise= torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)
                      else:                                                     #en 0 no se añade más ruid0
                          noise = torch.zeros(self.n).to(DEVICE)
                      #gamma=beta                                               #gamma mejor empleado para sampling desde distribución normal
                      if i>0:
                        gamma=((1-self.alphas_cumprod[i-1])/(1-alpha_hat))*beta #mejor gamma para asignar nosotros el valor x_0                        
                      else:
                        gamma=beta 
                      x = 1 / np.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + np.sqrt(gamma) * noise
                      model.train()
            return x

"""Lectura y carga del dataset"""
def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')

data_bien=torch.from_numpy(train_data).float()                                 #definimos nuestras variables de data originales ojo que son [sample,data]
test_bien=torch.from_numpy(test_data).float()

data_bien=((torch.transpose(data_bien,0,1)/255)*2-1)                         #pongo e dataset linealmente entre -0.5 y 0.5
test_bien=((torch.transpose(test_bien,0,1)/255)*2-1)

data_bien = data_bien[:,np.where((train_labels==1)|(train_labels==3))[0]]      #selecciono solo los samples de 1
test_bien = test_bien[:,np.where((test_labels==1)|(test_labels==3))[0]]

data_mean = torch.mean(data_bien)                                              #almaceno la media como variable
data_bien = (data_bien - data_mean)                                            #resto la media a mi dataset
test_bien = (test_bien - data_mean) # / torch.std(data_bien_,1).reshape(784,1)


print(np.where(train_labels==1)[0])
print(data_bien.size())
print(test_bien.size())

data_bien=data_bien[:,:12500]                                                   #acorto el tamaño del dataset total aprox 6100 train y 1200 test
test_bien=test_bien[:,:2100]
print(data_bien.size())

T=1000                                                                          #defino mi tiempo para todo el programa


x_util=torch.ones(data_bien.size(dim=1),data_bien.size(dim=0))                  #predefino las variables d1el entrenamiento
x_test=torch.ones(test_bien.size(dim=1),test_bien.size(dim=0))
y_util=torch.ones(data_bien.size(dim=1),data_bien.size(dim=0))
y_test=torch.ones(test_bien.size(dim=1),test_bien.size(dim=0))

x_util=x_util.to(DEVICE)                                                        #las defino en cuda
x_test=x_test.to(DEVICE)
y_util=y_util.to(DEVICE)
y_test=y_test.to(DEVICE)

train_data=1                                                                    #libero memoria
train_labels=1
test_data=1
test_labels=1
del test_data
del test_labels
del train_data
del train_labels
gc.collect()

"""Definiciones del entrenamiento"""

errores = 10                                                                    #variable de ruptura del while
learning_rate = 0.0001                                                          #usando adam no tendrá mayor rol
batch=50
                                                         
frame_num= 300                                                                  #cantidad de toma de datos totales
train_gap=1                                                                     #variable del numero de frames entre toma de datos
array_images=torch.ones(frame_num,784)

chuleta_mins=np.loadtxt('chuleta_min')                                          #carga de mínimos

pixel_bar=torch.cuda.mem_get_info(DEVICE)[1]                                    #crea una barra de carga con el total de la memoria de cuda
p_bar=tqdm(total=pixel_bar)
first=0
tol=6                                                                           #define la tolerancia de la varianza
almac_er=torch.ones(tol)                                                        #almacena la loss de las últimas tres épocas
almac=0
epoch_frame=1                                                                   #si se pone 1 solo se toma las variables originles empleado para sacar más de una toma de datos a diferentes valores de hidden var o epochs
salida_ploteable=torch.ones(T,epoch_frame,frame_num*train_gap)                  #inicializamos las variables de recogida de datos de loss
salida_ploteable_test=torch.ones(T,epoch_frame,frame_num*train_gap)
var_loss_ploteable=torch.ones(T,epoch_frame,frame_num*train_gap)

"""bucle de entrenamiento"""
for m in range(epoch_frame):
    D_in, H, D_out = 784, 1700+m*100, 784                                       #784 entradas, H neuronas en la capa oculta y 784 salidas.
    model_dict=torch.nn.ParameterDict()                                         #inicializamos el diccionario|probar a definir t como variable

    for i in {str(sub) for sub in range(T)}:                                    #definimos el modelo: autoencoder
        model_dict[i] = torch.nn.Sequential( 
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    user1=Difusion(torch.clone(data_bien).to(DEVICE) ,None,T)                   #dos usuarios de la clase difusión uno para el training y otro para test
    user2=Difusion(torch.clone(test_bien).to(DEVICE) ,None,T)
    l=[]                                                                        #l almacenará la loss
    enumerador=0
    loss_c=(nn.MSELoss(reduction='mean'))#/batch                                #defino mi función loss
 
    for g in range(T):  #T-901,T-900                                            #inicio del bucle de T
        i=str(g)                                                                #i será mi esntrada para el diccionario
        print('esto es g =',g)
        errores = 10
        epoch = 1                                                               #inicializo las epocas antes del while                                                       
        if g==4:
             learning_rate=learning_rate*2
        # if g==20:
        #     frame_num=frame_num/2
        model=model_dict[i]                                                     #doy el valor del diccionario con entrada i a una variable 
        model=model.to(DEVICE)                                                  #y lo mando a cuda
        optimizer = Adam(model.parameters(),lr=learning_rate)                   #inicializo adam en cada T
        model.train()                                                           #revisar
        cont=0                                                                  #inicializo la variable que dará entrada a la dimensión temporal de las variables de salida
        while errores>0 and epoch<=frame_num*train_gap:                         #bucle de entrenamiento que se mantiene por un valor de errores o por el número de épocas
            l=[]
            l_test=[]


            permut=torch.randperm(x_util.size(0))                               #hago una permuta aleatoria de training y test
            permut_=torch.randperm(x_test.size(0))
            forward_alpha_result, ruido_util = user1.forward_alpha_last_update(int(i)+1)            #disfusiono hasta el paso i
            forward_alpha_result_test, ruido_util_test = user2.forward_alpha_last_update(int(i)+1)
            
            x_util[:,:data_bien.size(dim=0)]=forward_alpha_result[permut,:]     #asocio las salidas con la variable útil en los índices random, se trata de el estado x_t
            forward_alpha_result=1                                              #borro de la memoria
            x_test[:,:data_bien.size(dim=0)]=forward_alpha_result_test[permut_,:]                   #aqui metemos los tests
            forward_alpha_result_test=1                                         #borro de la memoria

            y_util[:,:data_bien.size(dim=0)]=ruido_util[permut,:]               #asocio las salidas con la variable útil en los índices random, se trata del ruido empleado
            ruido_util=1                                                        #borro de la memoria
            y_test[:,:data_bien.size(dim=0)]=ruido_util_test[permut_,:]         #aqui metemos los tests
            ruido_util_test=1 #para descargar la RAM                            #borro de la memoria
        
            permut=1                                                            #borro de la memoria
            permut_=1
            
            for j in range(int(x_util.size(dim=0)/batch)):
                
                beta,alpha,alpha_hat,alpha_hat_anterior = user1.salida_factor(int(i))                #defino mi loss
                if g==0:
                  sigma=beta
                else:
                  sigma=((1-alpha_hat_anterior)/(1-alpha_hat))*beta
                loss=((beta**2)/(2*(sigma**2)*alpha*(1-alpha_hat)))*loss_c(model(x_util[range((j*batch),((j*batch)+batch)),:]),y_util[range((j*batch),((j*batch)+batch)),:])  #((beta**2)/(2*(sigma**2)*alpha*(1-alpha_hat)))*                                                          
                #loss=loss_c(model(x_util[range((j*batch),((j*batch)+batch)),:]),y_util[range((j*batch),((j*batch)+batch)),:])

                optimizer.zero_grad()                                           #paso backward
                loss.backward()
                optimizer.step()                                                #actualización de los pesos

            loss_=loss_c(model(x_util).detach(),y_util).cpu()                   #guardo la loss despues de ver el modelo toda la data
            loss_test=loss_c(model(x_test).detach(),y_test).cpu()          
            gc.collect()                                                        #limpio cache

            almac_er[almac]=loss_test.detach()                                  #almaceno la loss y hago la varianza de las 3 últimas medidas
            almac=almac+1
            print(cont)
            
            if almac==tol:

                almac=0
            if cont>=5:
                var_loss_ploteable[int(i),m,cont]=abs(torch.var(almac_er))
            salida_ploteable[int(i),m,cont]=loss_.detach() 
            salida_ploteable_test[int(i),m,cont]=loss_test.detach()
            print('el bueno',loss_)
            cont=cont+1

            #if cont>30 and int(i)>10 and cont<950:
            #    print(var_loss_ploteable[int(i),m,cont-1])
            #    if var_loss_ploteable[int(i),m,cont-1]<1e-11:
            #        errores=0
            #if chuleta_mins[g]==cont-1:                                        #analiza la entrada de chuleta_min
            #     errores=0
            if abs(loss_)<0.00000006:                                           #da una tolerancia a la loss
                errores=0
            epoch=epoch+1
      
            
        if first==0:                                                            #me da el uso de memoria de cuda
            pixel_bar=torch.cuda.mem_get_info(DEVICE)[0]
            p_bar.update(torch.cuda.mem_get_info(DEVICE)[1]-pixel_bar)
            prev=torch.cuda.mem_get_info(DEVICE)[0]
            first=1
        pixel_bar=torch.cuda.mem_get_info(DEVICE)[0]
        p_bar.update(prev-pixel_bar)
        prev=torch.cuda.mem_get_info(DEVICE)[0]


        gc.collect()
         
        model_dict[i]=model.cpu()                                               #guardo mi entreno  en una entrada del diccionario
        print(int(i))
        print('while: ',enumerador)
        enumerador=enumerador+1
    #del model
    #torch.cuda.empty_cache()

p_bar.close()                                                                   #cierro la barra de memoria de cuda


"""Representaciones gráficas y salida de variables"""
plt.figure()
for i in range(epoch_frame):
    plt.plot(salida_ploteable[0,i,:].detach().cpu().numpy())
    plt.plot(salida_ploteable_test[0,i,:].detach().cpu().numpy(),'--')
plt.ylabel('loss de todas las sample/batch')
plt.xlabel('epochs')
#plt.xscale('log')
plt.show()
plt.savefig('plot_loss')

from datetime import date
from datetime import datetime

#Día actual
today = date.today()

#Fecha actual
now = datetime.now()

print(today)
print(now)
#print(W)
today.strftime('%m/%d/%Y')
now.strftime('%m_%d_%Y_%H_%M_%S')
print(now)
nombre_test='salida_ploteable_test'+'_'+now.strftime('%m_%d_%Y_%H_%M_%S')
nombre_sample='salida_ploteable'+'_'+now.strftime('%m_%d_%Y_%H_%M_%S')
nombre_var='var_loss_ploteable'+'_'+now.strftime('%m_%d_%Y_%H_%M_%S')
torch.save(salida_ploteable_test.detach().cpu(),nombre_test) #lo pongo al fondo mejor por si falla
torch.save(salida_ploteable.detach().cpu(),nombre_sample) 
torch.save(var_loss_ploteable.detach().cpu(),nombre_var) 

nombre='dict'+'_'+now.strftime('%m_%d_%Y_%H_%M_%S')
torch.save(model_dict.cpu(),nombre) #para salvar el modelo
#torch.save(X.cpu(),'estados_difusos') #para salvar la muestra de data
