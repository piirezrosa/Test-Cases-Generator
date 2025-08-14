import numpy as np
import csv
import os

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

if __name__ == "__main__":
    # Configuração padrão (lê arquivo)
    input_file = "matriz.txt"
    output_file = "resultado.csv"

    # Se o arquivo não existir, cria um exemplo pequeno (6x6)
    if not os.path.exists(input_file):
        with open(input_file, 'w') as f:
            f.write("""0 1 1 1 1 2
                       1 0 0 -1 0 -1
                       3 -1 0 3 2 0
                       2 -2 2 0 -1 2
                       1 0 -1 3 0 0
                       0 -1 0 2 -1 0
                       1 1 0 -2 -2 0""")

    # Opção 1: Usar arquivo existente
    #solve_linear_equations(input_file, output_file, use_generator=False)

    # Opção 2: Gerar matriz 200x200 aleatória (descomente a linha abaixo)
    solve_linear_equations(input_file, output_file, use_generator=True)