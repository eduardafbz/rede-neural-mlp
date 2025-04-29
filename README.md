# Rede Neural Simples

Rede neural simples implementada em Python utilizando a biblioteca NumPy, com o objetivo de prever classificações de filmes com base em dados de média de avaliação e duração.  
Este projeto foi desenvolvido com fins **educacionais** como parte de atividades acadêmicas da faculdade, com foco no **treinamento de redes neurais artificiais** e no entendimento de seus fundamentos matemáticos.

### Tecnologias Utilizadas:
- [x] Linguagem: Python
- [x] Biblioteca NumPy (operações matriciais e funções matemáticas)
- [x] Manipulação de arquivos CSV

### Funcionalidades:
- Leitura de dataset contendo informações de filmes (média de avaliação e duração)
- Implementação de uma rede neural simples com:
  - Uma camada oculta
  - Função de ativação sigmoide
  - Propagação direta (forward) e retropropagação (backpropagation)
- Treinamento da rede para prever classificações binárias (0 ou 1)
- Cálculo de erro médio a cada 10 épocas
- Saída final da rede após treinamento

### Como executar este projeto:
1. Certifique-se de ter o Python 3 e a biblioteca NumPy instalada:
```bash
pip install numpy
```

2. Coloque o arquivo `movies_metadata.csv` no mesmo diretório do código (ou utilize os dados simulados já inclusos em caso de erro na leitura).

3. Execute o script Python:
```bash
python movies_mlp.py
```

### Estrutura esperada do CSV:
O arquivo `movies_metadata.csv` deve conter ao menos 3 colunas numéricas por linha. O script utiliza as três primeiras colunas para compor os dados de entrada da rede. Caso não seja encontrado ou ocorra erro na leitura, será utilizado um conjunto de dados padrão:

```plaintext
[5.0, 90]    # média dos usuários, duração (min)
[8.7, 115]
[4.7, 120]
[7.2, 130]
```

### Objetivo:
Este projeto foi criado como parte das atividades de aprendizado na faculdade para praticar:
- Implementação de redes neurais básicas
- Cálculo de gradientes e ajustes de pesos
- Noções fundamentais de machine learning

### Autor:
Desenvolvido por @eduardafbz como exercício acadêmico.
