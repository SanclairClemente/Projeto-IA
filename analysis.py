import pandas as pd

#Leitura e exibicao do data frame
pure_data = pd.read_csv("heart.csv")
#print(pure_data)

#var duplicated_rows recebe soma total de linhas duplicadas em pure_data (723)
duplicated_rows = pure_data.duplicated().sum()

#isnull considera cada atributo como True se não existir e False se existir
#enquanto .sum() percorre as colunas do dataFrame realizando a soma comutativa desses booleanos
#com 1(vazio) ou 0(presente) 
null_rows = pure_data.isnull().sum()

#Exibe coluna de atributos e quantia de cada atributo vazio no data frame: 0
#print(null_rows)   


# Lista de colunas categóricas
colunas_categoricas = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Verificar a distribuição de valores em cada variável 
distribuicao_categoricas = pure_data[colunas_categoricas].apply(pd.Series.value_counts)

# Exibir o resultado
#print("Distribuição de Variáveis Categóricas:")
#print(distribuicao_categoricas)
#Analise das variáveis categoricas
#fasting blood sugar,exercise induced angina e sex apresentam valores corretos(0-1)
#chest pain apresenta valores corretos(0-3)
#restecg apresenta valores corretos(0-2)
# thal apresenta valores incorretos(0-4) quando deveria apresentar valores de (0-3),
# number of major vessels(ca) apresenta valores entre (0-4) quando deveria apresentar
# valores no interavalo(0-3) e slope está correto(apresenta valores 0,1,2).


#Outliers

# Identificar as colunas numéricas
outliers_df = pd.DataFrame()

# Lista de colunas numéricas para análise de outliers
columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']  # Apenas colunas numéricas contínuas

for column in columns:
    # Calcular o IQR (Intervalo Interquartil)
    Q1 = pure_data[column].quantile(0.25)
    Q3 = pure_data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir os limites inferior e superior
    bot_limit = Q1 - 1.5 * IQR
    top_limit = Q3 + 1.5 * IQR
    
    # Identificar os outliers
    outliers = pure_data[(pure_data[column] < bot_limit) | (pure_data[column] > top_limit)]
    
    # Adicionar a coluna de outliers ao DataFrame (sem NaN)
    if not outliers.empty:
        outliers_df = pd.concat([outliers_df, outliers])

# Remover duplicatas para evitar registros repetidos
outliers_df = outliers_df.drop_duplicates()

# Exibir os outliers identificados
print("Outliers detectados:")
print(outliers_df)

#Distribuição da variável target
distribution_target = pure_data['target'].value_counts()
print(f"Distribuição da variável target:\n{distribution_target}")