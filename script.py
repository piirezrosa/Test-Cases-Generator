import csv
from collections import defaultdict


def calcular_matriz_transicao(arquivo_csv):
    """
    Lê um arquivo CSV com registros de estados de servidor e calcula
    a matriz de transição de estados (Matriz de Markov)

    Args:
        arquivo_csv (str): Caminho para o arquivo CSV

    Returns:
        dict: Matriz de transição no formato:
              {estado_origem: {estado_destino: probabilidade}}
    """
    # Estruturas para armazenar as transições
    transicoes = defaultdict(lambda: defaultdict(int))
    contagem_estados = defaultdict(int)

    # Leitura do arquivo CSV
    with open(arquivo_csv, 'r', encoding='utf-8') as arquivo:
        leitor = csv.DictReader(arquivo)
        estados = [linha['status'] for linha in leitor]

    # Contabiliza as transições entre estados consecutivos
    for i in range(len(estados) - 1):
        estado_atual = estados[i]
        proximo_estado = estados[i + 1]
        transicoes[estado_atual][proximo_estado] += 1
        contagem_estados[estado_atual] += 1

    # Calcula as probabilidades de transição
    matriz_transicao = {}
    for estado_origem, destinos in transicoes.items():
        total_transicoes = contagem_estados[estado_origem]
        probabilidades = {}

        for estado_destino, count in destinos.items():
            probabilidades[estado_destino] = count / total_transicoes

        matriz_transicao[estado_origem] = probabilidades

    return matriz_transicao


def imprimir_matriz_transicao(matriz):
    """Imprime a matriz de transição de forma formatada"""
    print("\nMatriz de Transição de Estados:")
    print("=" * 50)

    # Coleta todos os estados únicos
    todos_estados = set()
    for origem, destinos in matriz.items():
        todos_estados.add(origem)
        todos_estados.update(destinos.keys())

    # Cabeçalho da tabela
    cabecalho = ["De\\Para"] + sorted(todos_estados)
    print("{:<12}".format(cabecalho[0]), end="")
    for estado in cabecalho[1:]:
        print(" │ {:>10}".format(estado), end="")
    print("\n" + "-" * 12 + ("┼" + "-" * 11) * (len(cabecalho) - 1))

    # Linhas da matriz
    for origem in sorted(matriz.keys()):
        print("{:<12}".format(origem), end="")
        for destino in cabecalho[1:]:
            prob = matriz[origem].get(destino, 0.0)
            print(" │ {:>10.2%}".format(prob), end="")
        print()

    print("=" * 50)


def main():
    arquivo_csv = "server_status.csv"  # Substitua pelo seu arquivo

    # Calcula a matriz de transição
    matriz = calcular_matriz_transicao(arquivo_csv)

    # Imprime a matriz formatada
    imprimir_matriz_transicao(matriz)

    # Imprime também no formato de dicionário
    print("\nMatriz no formato dicionário:")
    for origem, destinos in matriz.items():
        print(f"\n{origem}:")
        for destino, prob in destinos.items():
            print(f"  → {destino}: {prob:.2%}")


if __name__ == "__main__":
    main()