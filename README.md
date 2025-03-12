# machine learning selecao de atributos
 Usado metodo de seleção de atributo
# Análise de Dados de Crédito com SVM e ExtraTreesClassifier

Este projeto utiliza o **Support Vector Machine (SVM)** e o **ExtraTreesClassifier** para treinar e avaliar um modelo de classificação com um conjunto de dados de crédito. O objetivo é prever a classe de um cliente com base em atributos financeiros e comportamentais.

## Descrição do Projeto

O código realiza as seguintes etapas principais:

1. **Leitura e Exploração dos Dados:**
   - O conjunto de dados `Credit.csv` é carregado e as principais estatísticas são exibidas para exploração inicial.

2. **Pré-processamento dos Dados:**
   - As variáveis categóricas são convertidas em valores numéricos utilizando a técnica de **Label Encoding** para permitir o uso em modelos de Machine Learning.

3. **Divisão do Conjunto de Dados:**
   - O conjunto de dados é dividido em duas partes: uma para treinamento (70%) e outra para teste (30%), garantindo consistência nos resultados usando o parâmetro `random_state=1`.

4. **Treinamento do Modelo com SVM:**
   - O modelo **Support Vector Classifier (SVC)** é treinado com os dados de treinamento e em seguida, é utilizado para fazer previsões com o conjunto de dados de teste.

5. **Avaliação de Desempenho:**
   - A **matriz de confusão** é gerada para avaliar o desempenho do modelo, indicando os verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
   - A **taxa de acerto** e a **taxa de erro** do modelo também são calculadas.

6. **Seleção de Características:**
   - O **ExtraTreesClassifier** é utilizado para calcular a importância das características, e um novo modelo é treinado utilizando apenas as características mais importantes, visando reduzir o overfitting e melhorar o desempenho.

## Dependências

O projeto depende das seguintes bibliotecas:

- `pandas` para manipulação de dados.
- `scikit-learn` para os modelos de Machine Learning e métricas de avaliação.
- `numpy` para manipulação de arrays numéricos.

## Resultados

O modelo de **SVM** inicialmente teve uma taxa de acerto de 71.6% e taxa de erro de 28.3%.
Após a redução da dimensionalidade com o ExtraTreesClassifier, o novo modelo **SVM2** obteve uma taxa de acerto diferente, 72,3%, levando em consideração que resultou em um modelo mais simples e eficiente.

