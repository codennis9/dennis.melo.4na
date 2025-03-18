print("Calculadora")

usuario = input("Digite o que deseja fazer: soma, tabuada ou divisão: ").lower()

def soma():
    print("Soma")
    user = int(input("Digite um número: "))
    user2 = int(input("Digite outro número: "))
    soma = user + user2
    print(f"{user} + {user2} = {soma}")

def tabuada():
    print("Tabuada")
    user = int(input("Digite um número: "))
    for tabuada in range(1, 11):
        valor = user * tabuada
        print(f"{user} x {tabuada} = {valor}")

def divisão():
    n1 = int(input("Digite um número: "))
    n2 = int(input("Digite outro número: "))
    divisão = n1 / n2
    print(f"{n1} / {n2} = {divisão}" 

if usuario == "soma":
    soma()
elif usuario == "tabuada":
    tabuada()
else:
    print("Opção inválida")

