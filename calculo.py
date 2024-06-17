import numpy as np
import plotly.graph_objects as go

# Definindo a função f(di) e sua derivada para a equação da lente delgada
def func(di, f, do):
    di = np.where(di == 0, np.nan, di)  # Evita divisão por zero substituindo 0 por NaN
    return (1/f) - (1/do) - (1/di)

def func_derivative(di, f, do):
    di = np.where(di == 0, np.nan, di)  # Evita divisão por zero substituindo 0 por NaN
    return -1/(di**2)

# Implementação do método de Newton
def Newton(f, df, x0, epsilon, Iter, f_params, df_params):
    if abs(f(x0, *f_params)) <= epsilon:  
        return x0
    k = 1 
    while k <= Iter:
        x1 = x0 - f(x0, *f_params)/df(x0, *df_params)
        if abs(f(x1, *f_params)) <= epsilon:
            return x1 
        x0 = x1
        k = k + 1
    return x1

# Parâmetros do problema
focal_length = 10  # distância focal f
object_distance = 15  # distância do objeto do
initial_guess = 10  # chute inicial ajustado para di

# Encontrando a raiz usando o método de Newton
Raiz = Newton(func, func_derivative, initial_guess, 0.001, 40, (focal_length, object_distance), (focal_length, object_distance))
print(f"A raiz encontrada é: {Raiz:.6f}")

# Preparando dados para o gráfico
di_values = np.linspace(1, 50, 400)  # Ajustado para evitar valores muito pequenos
f_values = func(di_values, focal_length, object_distance)
X, Y = np.meshgrid(di_values, di_values)
Z = func(X, focal_length, object_distance)

# Criando o gráfico 3D da função usando plotly
fig = go.Figure()

# Adicionando a superfície da função
fig.add_trace(go.Surface(x=di_values, y=di_values, z=Z, colorscale='Viridis', opacity=0.8))

# Destacando a raiz encontrada pelo método de Newton
fig.add_trace(go.Scatter3d(x=[Raiz], y=[Raiz], z=[func(Raiz, focal_length, object_distance)], mode='markers',
                           marker=dict(color='red', size=5),
                           name=f'Raiz encontrada: di = {Raiz:.6f}'))

# Configurando o layout do gráfico
fig.update_layout(title='Gráfico 3D da função de lente delgada',
                  scene=dict(
                      xaxis_title='di',
                      yaxis_title='di',
                      zaxis_title='f(di)'
                  ))

# Mostrando o gráfico
fig.show()

