
# el objetivo es crear un proceso de difusion para datos 2D
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
#from torch import optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
#a.to(DEVICE)


#intentaremos crear una espiral
sample_num=20000 #6000 en el mejor caso
theta = np.linspace(0,3*np.pi,sample_num)
r = 5 + 1*theta
x=np.multiply(r,np.cos(theta)) #+ 0.2*np.random.randn(sample_num)
y=np.multiply(r,np.sin(theta)) #+ 0.2*np.random.randn(sample_num)
factor=15
data_bien=torch.tensor([x,y])/12
#print(data_bien)
plt.scatter(data_bien.numpy()[0,:],data_bien.numpy()[1,:])
# prueba de distribucion normal de torch: print(torch.normal(0 * torch.ones(2,100),1))



class Difusion:
    def __init__(self,data,W,paso_temp):
        self.DEVICE_dos =torch.device("cuda:0" if data.is_cuda==True else "cpu")
        self.n=data.size(dim=0) #numero de puntos que se quiere samplear
        self.m=data.size(dim=1)
        self.T = paso_temp   #pasos de tiempo
        self.diffusion_rate = 0.01   #para la difusión normal
        self.mu=0
        self.var=1
        self.data=data
        self.data2=data
        self.start=1e-4
        self.end=0.02
        self.betas = self.linear_beta_schedule() #llamada interior
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.from_numpy(np.cumprod(self.alphas, axis=0)).to(self.DEVICE_dos)
        self.alpha_hat=self.alphas_cumprod
    # Beta no lineal
    def linear_beta_schedule(self):
        return np.linspace(self.start, self.end, self.T)

    # Funcion para aplicar el modelo de difusion hacia adelante
    # Es el paso del forward step con el reparametrization trick
    def forward_diffusion(self):
        ruido=torch.normal(self.mu * torch.ones(self.n),self.var).to(self.DEVICE_dos)
        #ruido=0.1*ruido
        for t in range(self.T):

            self.data =(torch.sqrt(1-self.diffusion_rate)*self.data) + (torch.sqrt(self.diffusion_rate)*ruido)

        return self.data,ruido

    def forward_diffusion_all(self):
        all_data = torch.zeros((self.T+1,self.data.shape[0],self.data.shape[1]))
        all_data[0,:] = self.data[:]
        for t in range(self.T):

            self.data =(torch.sqrt(1-self.diffusion_rate)*self.data) + (torch.sqrt(self.diffusion_rate)*torch.normal(self.mu * torch.ones(2,self.n),self.var))
            all_data[t+1,:] = self.data[:]

        return all_data

    # Paso forward con alphas acumulativas
    def forward_alpha(self):
        ruido=torch.normal(self.mu * torch.ones(self.m,self.n),self.var).to(self.DEVICE_dos)#de cara a la nueva distribucion de data pongo 1 en vez de 2
        #ruido=(1/(1+torch.exp(-(ruido))))
        # ruido=0.1*ruido
        data2 =torch.sqrt(self.alphas_cumprod[self.T-1])*self.data2.T + (torch.sqrt(1-self.alphas_cumprod[self.T-1])*ruido)
        return data2,ruido


    #proceso forward que se empleará para entrenar tiempo a tiempo cada máquina al completo
    def forward_alpha_last_update(self,Time):
        ruido=torch.normal(self.mu * torch.ones(self.m,self.n),self.var).to(self.DEVICE_dos)#de cara a la nueva distribucion de data pongo 1 en vez de 2
        #ruido=(1/(1+torch.exp(-(ruido))))
        # ruido=0.1*ruido
        data2 =torch.sqrt(self.alphas_cumprod[Time-1])*self.data2.T + (torch.sqrt(1-self.alphas_cumprod[Time-1])*ruido)
        return data2,ruido
    def reverse_sampling(self, W,ruido,model_dict):
            with torch.no_grad():
                x=torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE) #sampling de una distribución normal
                x=ruido #al comentar esta linea lo hacemos tomar un ruido base aleatorio si no usa la muestra característica
                predicted_noise=torch.ones(self.n).to(DEVICE)

                for i in reversed(range(self.T)):
                        alpha = self.alphas[i] #los alpha y alpha coump de el paso específico de la trayectoria reverse
                        alpha_hat = self.alphas_cumprod[i]
                        beta = self.betas[i]

                        model=model_dict[str(i)]
                        model=model.to(DEVICE)
                        model.eval()
                        predicted_noise=model(x)

                        if i >= 1:
                            noise= torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)
                        else:
                            noise = torch.zeros(self.n).to(DEVICE)
                        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                        #x[(x.size(dim=0)-1)]=i+1 #aqui igualamos el termino del tiempo al paso por el que vamos
                        model.train()
            return x

#el reverse personalizable
    def reverse_sampling_last_update(self, W,ruido,model_dict,Time):
            with torch.no_grad():
                x=torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE) #sampling de una distribución normal
                #x=ruido #al comentar esta linea lo hacemos tomar un ruido base aleatorio si no usa la muestra característica
                predicted_noise=torch.ones(self.n).to(DEVICE)

                for i in reversed(range(Time)):
                        alpha = self.alphas[i] #los alpha y alpha coump de el paso específico de la trayectoria reverse
                        alpha_hat = self.alphas_cumprod[i]
                        beta = self.betas[i]

                        model_aux=model_dict[str(i)]
                        model_aux.eval()
                        predicted_noise=model_aux(x)

                        if i >= 1:
                            noise= torch.normal(self.mu * torch.ones(self.n),self.var).to(DEVICE)
                        else:
                            noise = torch.zeros(self.n).to(DEVICE)
                        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                        #x[(x.size(dim=0)-1)]=i+1 #aqui igualamos el termino del tiempo al paso por el que vamos
                        model_aux.train()

            return x



T=1000
ruido=torch.ones(T,data_bien.size(dim=1),data_bien.size(dim=0))
X=torch.ones(T,data_bien.size(dim=1),data_bien.size(dim=0))


for j in range(1,T+1): #j entra como el tiempo pero habra que restarle uno para usarlo como indice
    print('vamos por: ',j)
    # for i in range(data_bien.size(dim=1)):
        #print('vamos por: ',i)
    user1=Difusion(torch.clone(data_bien),None,j)
        #forward_result ,ruido_util_itera= user1.forward_diffusion()
    forward_alpha_result, ruido_util = user1.forward_alpha()
        #print(forward_result.size())
    X[j-1,:,:data_bien.size(dim=0)]=forward_alpha_result[:]
    #X[j-1,:,data_bien.size(dim=0)]=j
    ruido[j-1,:,:data_bien.size(dim=0)]=ruido_util[:]
    #ruido[j-1,:,data_bien.size(dim=0)]=j

#ahora X y ruido me almacenan los distintos pasos de tiempo en los distintos j con los distintos ruidos a cada paso de tiempo PERO para luego el reverse al ser forward alpha no cuadra muy bien¿?
print(ruido.size())
print(X.size())
ruido=torch.reshape(ruido,[T,data_bien.size(dim=1),data_bien.size(dim=0)])
X=torch.reshape(X,[T,data_bien.size(dim=1),data_bien.size(dim=0)])
print(ruido.size())
print(X.size())
#print(X)
Y=ruido
#permut=torch.randperm(X.size(0))
#print(permut.size())
#x_preshuffle=X
#X=X[permut,:,:] #al estar comentado no estamos mezclando los samples
#Y=Y[permut,:,:]




D_in, H, D_out = 2, 1000, 2 #784 entradas, 100 neuronas en la capa oculta y 784 salidas.
model_dict=torch.nn.ParameterDict()
#loss_dict=torch.nn.ParameterDict()
for i in {str(sub) for sub in range(T)}:
    model_dict[i] = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )



errores = 10                             #esta variable corta el bucle while
learning_rate = 0.008                     #learning rate que se le introducira a Adam
lr_=np.linspace(learning_rate,learning_rate/2,T)
cont_aux=1   
cont_print=0                             #Mide los intervalos entre toma de muestras alcanzará en valor de train_gap y se tomará una muestra del reverse
cont_aux_continuo=0                      #va a ser el que vaya dando entradas a array_images al tomar la muestra de reverse
frame_num= 30                           #cantidad de toma de datos totales
train_gap=1                              #variable del numero de frames entre toma de datos
array_images=torch.ones(frame_num,784)   #almacenará las imágenes del proceso reverse         PERTENECE A OTRA CONF
batch=100                                #batch size para el entrenamiento
image_gap=1                              #separacion de tiempo entre toma de datos
user1=Difusion(torch.clone(data_bien).to(DEVICE),None,T)
l=[]                                     #almacenará la loss final de cada epoch
x_util=torch.ones_like(X[0,:,:])         #en el entrenamiento se generarán los propios procesos forward entre cada epoch x_util y y_util almacenarán esas salidas para todas las samples
x_util=x_util.to(DEVICE)
y_util=torch.ones_like(Y[0,:,:])
y_util=y_util.to(DEVICE)
enumerador=0                             #mide los pasos de tiempo
loss_c=(nn.MSELoss(reduction='mean'))    #función loss

for g in range(T):
    learning_rate=lr_[g]
    i=str(g)
    epoch = 1
    cont_aux=1
    cont_aux_continuo=0
    model=model_dict[i].to(DEVICE)
    #if g>200:
    #     learning_rate=learning_rate/100
    optimizer = Adam(model.parameters(),lr=learning_rate)
    model.train()
    #loss=torch.to(DEVICE)
    print('et')
    while errores>0 and epoch<=frame_num*train_gap:
       
        error_dis=0
        l=[]
        forward_alpha_result, ruido_util = user1.forward_alpha_last_update(g+1)
        x_util[:,:data_bien.size(dim=0)]=forward_alpha_result[:].to(DEVICE)
        y_util[:,:data_bien.size(dim=0)]=ruido_util[:].to(DEVICE)

        permut=torch.randperm(x_util.size(0))
        x_util=x_util[permut,:]
        y_util=y_util[permut,:]
        
        for j in range(int(X.size(dim=1)/batch)):
            #ENTRENAMIENTO
            loss=loss_c(model(x_util[range((j*batch),((j*batch)+batch)),:]),y_util[range((j*batch),((j*batch)+batch)),:]) #1 porque se emplean únicamente las y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if cont_print==0:
            print(loss)
            cont_print=0
        else:
            cont_print=cont_print+1
        if abs(loss)<0.00000006:
            errores=0
        epoch=epoch+1

    print('el bueno',loss_c(model(x_util[:,:]),y_util[:,:]).detach() )
    model_dict[i]=model.cpu()

    if cont_aux==image_gap:
        samplenum=1
        #salida_sampled = user1.reverse_sampling_last_update(None,x_util[samplenum,:].to(DEVICE),model_dict.to(DEVICE),g)
        #salida_sampled = user1.reverse_sampling(ruido[:,1],X[:,1])


        #array_images[cont_aux_continuo,:784]=salida_sampled[:784]
        cont_aux_continuo=cont_aux_continuo+1
        cont_aux=1
    else:
        cont_aux=cont_aux+1

    #el modelo siguiente se inicializa con los parámetros del anterior
    #if int(i)<T-1:
    #    model_dict[str(int(i)+1)]=model.cpu()
    print(int(i))
    print('while: ',enumerador)
    enumerador=enumerador+1





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


nombre='dict'+'_'+now.strftime('%m_%d_%Y_%H_%M_%S')
torch.save(model_dict.cpu(),nombre) #para salvar el modelo
#torch.save(X.cpu(),'estados_difusos') #para salvar la muestra de data