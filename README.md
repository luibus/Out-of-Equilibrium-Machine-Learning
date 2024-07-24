The following files follow the work developed for the final degree project on Out-of-equilibrium Machine Learning

ABSTRACT:

In recent years, Diffusion Models have undergone a series of transformations and developments
that have turned them into very useful and competitive tools in the field of generative computing.
Consequently, this work studies the last developments of diffusion models under an unified frame,
and constructs a practical application from the conclusions drawn. Additionally, a link to phase
transitions is introduced.

The study will begin from the basis of non-equilibrium thermodynamics, where diffusion is understood as the conduction mechanism due to a change or gradient in a systems property under
a thermodynamical force. As well as particle or heat diffusion, we can study diffusion occurring
in the information of a system. Therefore, it would be convenient to study the evolution of probability through a diffusive process. To achieve this goal, a Markov Diffusion Chain will be used to transform an original data distribution t = 0 into a thermalized Gaussian distribution t = T.

Hence, from this probability, a second and reversed diffusion process can be constructed.
This second reverse diffusion process will be the goal to modelize through the simplest yet practical machine learning architecture, single hidden layer autoencoders. That would allow a traceable study of the model together with good results.

From this study it will be proved how the reverse process can transform samples from Gaussian
distributions back into new samples living in our original data distribution.

Training Algorithm (Ho et al. ddpm):

![image](https://github.com/user-attachments/assets/0f4fd138-f77b-4693-a2d1-8765fa409fce)


Forward and reverse diffusion (MNIST):

![image](https://github.com/user-attachments/assets/98d0fc1b-b672-4150-a099-c5755fe7843d)

Results (MNIST and Swiss-Roll):

![image](https://github.com/user-attachments/assets/345965e5-7cff-4d0e-8f68-3e69ede0e8c0)

![image](https://github.com/user-attachments/assets/568ff606-99a7-4240-b8a0-00474e1b12f0)

Reconstruction results (MNIST):

![image](https://github.com/user-attachments/assets/10a75d7b-71a4-4ad7-abea-14da7ae1a697)

![image](https://github.com/user-attachments/assets/ec4ec38e-2c4b-47a1-a92f-37c7cf5d7731)


USE:

A jupyter notebook is uploaded with the code used, also, a folder with the MNIST files and a last update training code file. 

The jupyther version acts as a guide of the variables used during trainning and the methods used for evaluation. Training is recommend to be done with the "prueba_epochs_ho.py" file.

As an example the "prueba_epochs_ho.py" can be runned on a folder together with the MNIST files. The trainning will run the epochs specified for as much machines as timesteps the diffusion is defined, one at a time. It is designed to accept CUDA for faster implementation and at the end of the run a python dictionary will be created as a model ready to be evaluated.

