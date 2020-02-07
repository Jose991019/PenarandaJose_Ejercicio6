import numpy as np
import matplotlib.pyplot as plt

x = np.array([4.6, 6.0, 2.0, 5.8])
sigma = np.array([2.0, 1.5, 5.0, 1.0])

def post(mu):
    return(np.sum(-0.5*(x-mu)**2/(sigma**2)))
mu = np.linspace(-0,10,1000)

L = []
for i in mu:
    L.append(post(i))
    
maximo = np.max(L)
mu0 = int(np.where(L == maximo)[0])
deriv2 = (L[mu0+1] - 2*L[mu0] + L[mu0-1])/(mu[mu0]-mu[mu0-1])**2
sigma2 = (-deriv2)**(-1/2)

prob = np.exp(L)
norm = np.trapz(prob, x = mu)

prob = prob/norm

N = 100000
lista = [np.random.random()*np.pi]
sigma_delta = 1.0

for i in range(1,N):
    propuesta  = lista[i-1] + np.random.normal(loc=0.0, scale=sigma_delta)
    r = min(1,np.exp(post(propuesta)-post(lista[i-1])))
    alpha = np.random.random()
    if(alpha<r):
        lista.append(propuesta)
    else:
        lista.append(lista[i-1])

plt.plot(mu,prob)
_ = plt.hist(lista, density=True, bins=40)
plt.title("{} = {:.3f} {} {:.3f}".format("$\mu_0$",mu[mu0],"$\pm$",sigma2))
plt.xlabel("{}".format("$\mu$"))
plt.ylabel("posterior")
plt.savefig("mean.png")
