import pandas as pd

pure_data = pd.read_csv("heart.csv")

preprocessed_data = pure_data.copy()


#Remove linhas duplicadas
preprocessed_data = preprocessed_data.drop_duplicates()

print(preprocessed_data)

# Remove valores incorretos em 'thal' e 'ca'
preprocessed_data = preprocessed_data[preprocessed_data['thal'].isin([0, 1, 2])]
preprocessed_data = preprocessed_data[preprocessed_data['ca'].isin([0, 1, 2, 3])]

print(preprocessed_data)

preprocessed_data = preprocessed_data[preprocessed_data['thalach'] != 71]

preprocessed_data.to_csv("clean_heart.csv",index=False)


#print(preprocessed_data)
# Verificar a distribuição de valores em cada variável 
#distribuicao_categoricas = preprocessed_data[colunas_categoricas].apply(pd.Series.value_counts)
#print(distribuicao_categoricas)