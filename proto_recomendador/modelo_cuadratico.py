import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class modelo_cuadratico():
    """
    Modelo cuadratico para collaborative filtering segun el paper A Recommendation Model Based on Deep Neural Network del autor LIBO ZHANG
    """
    def __init__(self, a:int, b:int, n_users:int, n_items:int, device = 'auto'):
        """
        Attributes
        ----------
        a : int
            Latent factors para la matriz de usuarios U
        b : int
            Latent factors para la matriz de usuarios V
        n_users : int
            Cantidad de usuarios
        n_items : int
            Cantidad de items

        Methods
        -------
        entrenar(self, ratings, lr, epochs, track_every, early_stop_min_diff)
            Entrena el modelo en base a los ratings proporcionados
        """
        l = a+b
        self.a = a
        self.b = b
        self.W = np.random.rand(l,l)
        self.w = np.random.rand(l)
        self.U = np.random.rand(n_users, a)
        self.V = np.random.rand(n_items, b)
        self.p = calculate_linear_weighted_features(self.U,self.w[:a]) + calculate_weighted_interactions(self.U, self.W[:a,:a])
        self.q = calculate_linear_weighted_features(self.V,self.w[a:]) + calculate_weighted_interactions(self.V,self.W[a:,a:])
        self.z = np.random.rand(1)
        
        self.device = device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Using device:', self.device)
        
        self.p = torch.from_numpy(self.p).to(self.device)
        self.q = torch.from_numpy(self.q).to(self.device)
        self.z = torch.from_numpy(self.z).to(self.device)
        self.W = torch.from_numpy(self.W).to(self.device)
        self.w = torch.from_numpy(self.w).to(self.device)
        self.U = torch.from_numpy(self.U).to(self.device)
        self.V = torch.from_numpy(self.V).to(self.device)
    
    def entrenar(self, ratings:torch.Tensor, lr = 0.01, epochs = 10**6, track_every = 1000, early_stop_min_diff = 0.000001):
        
        a = self.a
        b = self.b

        indicator = init_indicator_matrix(ratings)
        ratings = ratings.to(self.device)
        indicator = indicator.to(self.device)
        
        loss_tracker = []
        L_anterior = 0
        L_nuevo = 0
        
        n_items = self.V.shape[0]
        n_users = self.U.shape[0]

        for i in range(epochs):
            L_anterior = L_nuevo
            # Forward Pass
            p_broad = torch.broadcast_to(self.p, (-1, n_items))
            q_broad = torch.broadcast_to(self.q, (-1, n_users)).transpose(0,1)
            z_broad = torch.broadcast_to(self.z, (n_users, n_items))
            R_uv = z_broad + p_broad + q_broad + calculate_weighted_matrix_multiplication(self.W,self.U,self.V)
            L_nuevo = ((ratings[ratings!=-1]-R_uv[ratings!=-1])**2).mean()/2 # Calculo función perdida sólo para los datos conocidos
            L_diferencia = (L_anterior - L_nuevo).item()

            mae = (torch.absolute(ratings[ratings!=-1]-R_uv[ratings!=-1])).mean()

            if (i%track_every==track_every-1):
                print("Epoch {} Loss: {}  MAE: {}".format(i, L_nuevo, mae), end = "\r")
            if (i%track_every==track_every-1):
                loss_tracker.append(L_nuevo.item())

            if (L_diferencia < early_stop_min_diff)&(i>100):
                print("\nEarly stopping en epoch {}".format(i))
                break

            delta_uv = indicator*(R_uv-ratings)
            # Update z
            self.z = self.z - lr*(delta_uv.mean())
            # Update p
            self.p = self.p - lr*(delta_uv.mean(axis=1).reshape(n_users,-1))
            # Update q
            self.q = self.q - lr*(delta_uv.mean(axis=0).reshape(n_items,-1))
            # Update U
            self.U = self.U - lr * (delta_uv@(self.W[:a,a:]@self.V.transpose(0,1)).transpose(0,1))/(n_items*self.b)
            # Update V
            self.V = self.V - lr * ((self.W[:a,a:]@self.U.transpose(0,1))@delta_uv).transpose(0,1)/(n_users*self.a)
            # Update W (O la parte de W que se va a usar)
            self.W[:a,a:] = self.W[:a,a:] - lr * ((delta_uv.transpose(0,1)@self.U).transpose(0,1)@self.V)/(n_users*n_items)

        acc = 1-torch.absolute(torch.round(R_uv[ratings!=-1])-ratings[ratings!=-1]).mean()
        print("Accuracy: {}".format(acc))
        plot_loss(loss_tracker, track_every)
        return R_uv
    
def calculate_weighted_matrix_multiplication(W:torch.Tensor, M_1: torch.Tensor, M_2: torch.Tensor):
    a = M_1.shape[1]
    WM_1 = W[:a,a:]@M_1.transpose(0,1)
    WM_1M_2 = WM_1.transpose(0,1)@M_2.transpose(0,1)
    return WM_1M_2

def calculate_linear_weighted_features(M:torch.Tensor, w: torch.Tensor):
    """ Calcula Suma(i=1, a)wi Uui"""
    if M.shape[1] != w.shape[0]:
        raise Exception("calculate_linear_weighted_features: las matrices no se pueden multiplicar. Chequear la dimension.")
    else:
        return (M@w.reshape(-1,1))

def calculate_weighted_interactions(M: torch.Tensor, W: torch.Tensor):
    """ Calcula las interacciones de cada fila de M ponderadas con W.
    Es una implementacion de la siguiente formula
    Suma(i=1, a-1)Suma(j=i+1, a) W_ij Uui Uuj
    """
    if M.shape[1]!=W.shape[0]:
        raise Exception("calculate_weighted_interactions: Las matrices M y W no se pueden multiplicar.")
    if W.shape[0]!=W.shape[1]:
        raise Exception("calculate_weighted_interactions: La matriz de pesos no es cuadrada.")

    M_interactions = calculate_interactions(M)
    upper_W = get_upper_triangle_matrix(W)
    M_weighted_interactions = (M_interactions@upper_W).reshape(-1,1)
    return M_weighted_interactions

def get_upper_triangle_matrix(M: torch.Tensor):
    """Devuelve los elementos de la triangular superior sin la diagonal. 
    Esos elementos se aplastar a un array 1D
    Si M es la matriz de 3x3
    1 2 3
    4 5 6
    7 8 9
    La salida es el array [2,3,6]
    """
    if (M.shape[0]!=M.shape[1]):
       raise Exception("get_upper_triangle_matrix: La matriz no es cuadrada")
    else:
        out_size = M.shape[0]
        return M[np.triu_indices(out_size, k = 1)]

def calculate_interactions(M: torch.Tensor):
    interaction = PolynomialFeatures(include_bias=False, interaction_only=True)
    M_interactions = interaction.fit_transform(M)
    M_interactions = M_interactions[:,M.shape[1]:] # Me quedo unicamente con las interacciones de tipo xy, xz, yz
    return M_interactions

def init_indicator_matrix(ratings: torch.Tensor):
    """ Recibe la matriz de interacciones y
    devuelve la matriz indicadora del mismo tamaño que tiene
    0 donde la matriz de interacciones tiene -1 y 1 en todos los demás lugares
    """
    indicator = torch.ones((ratings.shape[0], ratings.shape[1]))
    indicator[np.where(ratings == -1)] = 0
    return indicator

def plot_loss(loss_tracker: list, track_every: int):
    x = np.arange(0,len(loss_tracker)*track_every, track_every)[1:]
    y = loss_tracker[1:]

    plt.plot(x, y, linewidth=2)
    plt.grid(alpha=.4)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()