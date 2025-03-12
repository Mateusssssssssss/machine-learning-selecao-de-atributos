# Manipular os dados
import pandas as pd
# Divide o conjunto de dados em duas partes: uma para treinamento e outra para teste.
from sklearn.model_selection import train_test_split
#Converte rótulos categóricos (como texto) em valores numéricos
# para que possam ser usados em modelos de machine learning.
from sklearn.preprocessing import LabelEncoder
# confusion_matrix: Calcula a matriz de confusão, que mostra o desempenho de um classificador, 
# comparando as previsões com os valores reais.
#Avaliar o desempenho de modelos de classificação (ex.: True Positives, False Positives, etc.).
# acuracu_score: Calcula a acurácia do modelo, ou seja, a porcentagem de previsões corretas.
from sklearn.metrics import confusion_matrix, accuracy_score
#é utilizado para treinar um modelo de árvore de decisão. Ele é um classificador baseado em floresta aleatória (ensemble), 
# que pode calcular a importância das características (atributos) do conjunto de dados durante o treinamento.
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.svm import SVC
# Ler o dataset
dados = pd.read_csv('Credit.csv')
print(dados.head())
print(dados.describe())
print(dados.shape)

# formato da matriz
previsores = dados.iloc[:,0:20].values
classe = dados.iloc[:, 20].values

# Transformação dos atributos categóricos em atributos numéricos, 
# passando o índice de cada coluna categórica. Precisamos criar um objeto para cada atributo categórico, 
# pois na sequência vamos executar o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente,
# o que deixará o teste inconsistente.
# Codificação de variáveis categóricas para variáveis numéricas.
labelencoder1 = LabelEncoder()
previsores[:,0] = labelencoder1.fit_transform(previsores[:,0])

labelencoder2 = LabelEncoder()
previsores[:,2] = labelencoder2.fit_transform(previsores[:,2])

labelencoder3 = LabelEncoder()
previsores[:, 3] = labelencoder3.fit_transform(previsores[:, 3])

labelencoder4 = LabelEncoder()
previsores[:, 5] = labelencoder4.fit_transform(previsores[:, 5])

labelencoder5 = LabelEncoder()
previsores[:, 6] = labelencoder5.fit_transform(previsores[:, 6])

labelencoder6 = LabelEncoder()
previsores[:, 8] = labelencoder6.fit_transform(previsores[:, 8])

labelencoder7 = LabelEncoder()
previsores[:, 9] = labelencoder7.fit_transform(previsores[:, 9])

labelencoder8 = LabelEncoder()
previsores[:, 11] = labelencoder8.fit_transform(previsores[:, 11])

labelencoder9 = LabelEncoder()
previsores[:, 13] = labelencoder9.fit_transform(previsores[:, 13])

labelencoder10 = LabelEncoder()
previsores[:, 14] = labelencoder10.fit_transform(previsores[:, 14])

labelencoder11 = LabelEncoder()
previsores[:, 16] = labelencoder11.fit_transform(previsores[:, 16])

labelencoder12 = LabelEncoder()
previsores[:, 18] = labelencoder12.fit_transform(previsores[:, 18])

labelencoder13 = LabelEncoder()
previsores[:, 19] = labelencoder13.fit_transform(previsores[:, 19])


# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
# random_state=1 é uma maneira de garantir consistência nos resultados, o que é útil quando você está tentando comparar modelos ou experimentos.
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 1)
print(x_teste)  


# Criação e treinamento do modelo (geração da tabela de probabilidades)
#  svm = SVC() cria uma instância do Support Vector Classifier (SVC), que é um classificador baseado no Máquinas de Vetores de Suporte (SVM).
# O SVC é um classificador que separa as classes de dados com base em margens ideais.
svm = SVC()
#treina o modelo
svm.fit(x_treinamento, y_treinamento)

# Previsões utilizando os registros de teste
previsoes = svm.predict(x_teste)
print(previsoes)


#geração da matriz de confusão
#A matriz de confusão é uma ferramenta essencial para avaliar a performance de um modelo de classificação, 
# pois mostra não apenas os acertos do modelo (TP e TN), mas também os erros (FP e FN). 
# Isso ajuda a entender onde o modelo está errando e pode fornecer informações valiosas para ajustar o
# modelo ou o processo de treinamento
confusao = confusion_matrix(y_teste, previsoes)
print(confusao)

# calcula a taxa de acerto e a taxa de erro do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
#Taxa de acerto: 0.71666, Logo 72,6% de acerto
#Taxa de erro:0.2833, Logo 28,3% de erro
print(f'Taxa de acerto: {taxa_acerto}\nTaxa de erro:{taxa_erro}')

# Utilização do algoritimo ExtraTressClassifier para extrair as caracteristica mais importante
forest = ExtraTreesClassifier()
forest.fit(x_treinamento, y_treinamento)
# Descobri qual os atributsos mais importantes
#extrai a importância de cada característica.
importancia = forest.feature_importances_
print(importancia)
# Croação de nova base de dados utilizando somente  os atributos mais importantes
#contendo apenas as colunas de atributos mais importantes, selecionadas pelos índices [0, 1, 2, 3]
#o que pode ajudar a reduzir o overfitting e melhorar o desempenho do modelo em termos de taxa de acerto.
x_treinamento2 = x_treinamento[:,[0, 1, 2, 3]]
x_teste2 = x_teste[:,[0, 1, 2, 3]]
#Crialçao de outro modelo com a base de dados reduzidas, treinamento e obtenção das previsões e taxa de acerto.
#Resulta em um modelo mais simples e eficiente, que pode ter um desempenho melhor.
svm2 = SVC()
svm2.fit(x_treinamento2, y_treinamento)
previsoes2 = svm2.predict(x_teste2)
taxa_acerto2 = accuracy_score(y_teste, previsoes2)
print(f'Taxa de Acerto com SVC: {taxa_acerto2}')