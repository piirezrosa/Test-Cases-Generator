import numpy as np
import csv
import os
from collections import defaultdict


class MarkovStateTransition:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.states = set()

    def process_data(self, data):
        """Processa dados brutos para calcular transições entre estados"""
        if not data or len(data) < 2:
            raise ValueError("Dados insuficientes para calcular transições")

        previous_state = None

        for entry in data:
            current_state = entry['status']
            self.states.add(current_state)

            if previous_state is not None:
                self.transition_counts[previous_state][current_state] += 1
                self.state_counts[previous_state] += 1

            previous_state = current_state

    def calculate_transition_matrix(self):
        """Calcula a matriz de transição baseada nos dados processados"""
        transition_matrix = {}

        for from_state in self.states:
            transition_matrix[from_state] = {}
            total_transitions = self.state_counts.get(from_state, 0)

            for to_state in self.states:
                count = self.transition_counts[from_state].get(to_state, 0)
                probability = count / total_transitions if total_transitions > 0 else 0
                transition_matrix[from_state][to_state] = probability

        return transition_matrix


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        constants = np.array([float(num) for num in lines[-1].split()])
        coefficients = np.array([[float(num) for num in line.split()] for line in lines[:-1]])
        return coefficients, constants
    except Exception as e:
        print(f"Erro ao ler o arquivo TXT: {e}")
        raise


def read_csv_file(file_path):
    coefficients = []
    constants = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        blank_line_index = next(
            (i for i, row in enumerate(rows) if all(cell.strip() == '' for cell in row)), -1
        )
        if blank_line_index == -1:
            raise ValueError("Arquivo CSV inválido: falta linha em branco para separar coeficientes e constantes.")

        for row in rows[:blank_line_index]:
            coefficients.append([float(num) for num in row if num.strip()])

        for row in rows[blank_line_index + 1:]:
            constants.append(float(row[0]))

        return np.array(coefficients), np.array(constants)
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        raise


def read_server_status_csv(file_path):
    """Lê arquivo CSV com estados do servidor para análise de Markov"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Erro ao ler arquivo de estados do servidor: {e}")
        raise


def write_csv_file(file_path, solution):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Variável", "Valor"])
            for i, value in enumerate(solution):
                writer.writerow([f"x{i + 1}", f"{value:.4f}"])
        print(f"Solução gravada no arquivo: {file_path}")
    except Exception as e:
        print(f"Erro ao escrever o arquivo: {e}")
        raise


def write_transition_matrix(file_path, matrix):
    """Escreve a matriz de transição em formato CSV"""
    try:
        # Coleta todos os estados únicos
        states = sorted(matrix.keys())

        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Cabeçalho
            writer.writerow(['De/Para'] + states)

            # Linhas da matriz
            for from_state in states:
                row = [from_state]
                for to_state in states:
                    prob = matrix[from_state].get(to_state, 0.0)
                    row.append(f"{prob:.4f}")
                writer.writerow(row)

        print(f"Matriz de transição gravada em: {file_path}")
    except Exception as e:
        print(f"Erro ao escrever matriz de transição: {e}")
        raise


def generate_400x400_matrix():
    """Gera uma matriz 400x400 e vetor de constantes aleatórios."""
    coefficients = np.random.uniform(low=-10, high=10, size=(2000, 2000))
    constants = np.random.uniform(low=-10, high=10, size=2000)
    return coefficients, constants


def solve_linear_equations(input_file, output_file, use_generator=False):
    try:
        if use_generator:
            print("Gerando matriz 400x400 aleatória...")
            coefficients, constants = generate_400x400_matrix()
        else:
            file_extension = os.path.splitext(input_file)[1].lower()
            if file_extension == ".txt":
                coefficients, constants = read_txt_file(input_file)
            elif file_extension == ".csv":
                coefficients, constants = read_csv_file(input_file)
            else:
                raise ValueError("Formato de arquivo não suportado. Use .txt ou .csv.")

        print(f"Matriz de coeficientes (dimensão {coefficients.shape}):")
        print(coefficients[:2, :2], "...")  # Mostra apenas um preview
        print(f"\nVetor de constantes (dimensão {constants.shape}):")
        print(constants[:2], "...")  # Mostra apenas um preview

        if coefficients.shape[0] != coefficients.shape[1]:
            raise ValueError("A matriz de coeficientes deve ser quadrada.")
        if coefficients.shape[0] != constants.shape[0]:
            raise ValueError("O número de constantes deve ser igual ao número de equações.")

        solution = np.linalg.solve(coefficients, constants)
        write_csv_file(output_file, solution)

    except np.linalg.LinAlgError as e:
        print(f"Erro ao resolver o sistema: {e} (matriz singular ou mal condicionada)")
    except Exception as e:
        print(f"Erro no processamento: {e}")


def analyze_server_transitions(input_file, output_file):
    """Analisa estados do servidor e calcula matriz de transição"""
    try:
        # Carrega dados do servidor
        server_data = read_server_status_csv(input_file)
        print(f"Carregados {len(server_data)} registros de estados do servidor")

        # Processa os dados para matriz de transição
        state_processor = MarkovStateTransition()
        state_processor.process_data(server_data)

        # Calcula matriz de transição
        transition_matrix = state_processor.calculate_transition_matrix()

        # Escreve a matriz em formato CSV
        write_transition_matrix(output_file, transition_matrix)

        # Exibe resumo
        print("\nResumo da Matriz de Transição:")
        states = sorted(transition_matrix.keys())
        for from_state in states:
            for to_state in states:
                prob = transition_matrix[from_state].get(to_state, 0.0)
                if prob > 0:
                    print(f"  {from_state} → {to_state}: {prob:.2%}")

        return transition_matrix

    except Exception as e:
        print(f"Erro na análise de transições: {e}")
        raise


if __name__ == "__main__":
    # Menu principal
    print("Selecione o modo de operação:")
    print("1 - Resolver sistema de equações lineares")
    print("2 - Analisar matriz de transição de estados do servidor")
    choice = input("Opção: ").strip()

    if choice == '1':
        # Modo de resolução de equações lineares
        input_file = "matriz.txt"
        output_file = "resultado.csv"

        if not os.path.exists(input_file):
            with open(input_file, 'w') as f:
                f.write("""0 1 1 1 1 2
1 0 0 -1 0 -1
3 -1 0 3 2 0
2 -2 2 0 -1 2
1 0 -1 3 0 0
0 -1 0 2 -1 0
1 1 0 -2 -2 0""")

        use_generator = True  # Mantendo sua opção padrão
        solve_linear_equations(input_file, output_file, use_generator)

    elif choice == '2':
        # Modo de análise de matriz de transição
        input_file = input("Arquivo CSV com dados do servidor: ").strip()
        output_file = input("Arquivo de saída para matriz de transição: ").strip()

        if not input_file:
            input_file = "server_status.csv"
        if not output_file:
            output_file = "matriz_transicao.csv"

        analyze_server_transitions(input_file, output_file)

    else:
        print("Opção inválida. Selecione 1 ou 2.")