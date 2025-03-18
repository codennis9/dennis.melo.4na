print("Calculadora")

usuario = input("Digite o que deseja fazer: Soma ou Tabuada: ").lower()

def soma():
    print("Soma")
    user = int(input("Digite um número: "))
    user2 = int(input("Digite outro número: "))
    soma = user + user2
    print(f"A soma de {user} + {user2} é igual a {soma}")

def tabuada():
    print("Tabuada")
    user = int(input("Digite um número: "))
    for tabuada in range(1, 11):
        valor = user * tabuada
        print(f"{user} x {tabuada} = {valor}")

if usuario == "soma":
    soma()
elif usuario == "tabuada":
    tabuada()
else:
    print("Opção inválida")

