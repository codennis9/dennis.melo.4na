def regressao_linear(x, y):
    n = len(x)
    soma_x = sum(x)
    soma_y = sum(y)
    soma_x2 = sum([xi**2 for xi in x])
    soma_xy = sum([xi*yi for xi, yi in zip(x, y)])
    
    b1 = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x**2)
    b0 = (soma_y - b1 * soma_x) / n
    
    return b0, b1

def prever(x, b0, b1):
    return [b0 + b1 * xi for xi in x]

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

b0, b1 = regressao_linear(x, y)

print(f"Coeficiente b0: {b0}")
print(f"Coeficiente b1: {b1}")

y_previsto = prever(x, b0, b1)
print(f"Valores previstos para y: {y_previsto}")
