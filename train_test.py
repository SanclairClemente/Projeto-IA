from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

preprocessed_data = pd.read_csv("clean_heart.csv") 

# Separar features e labels
features = preprocessed_data.drop(columns=['target'])  # Todos os atributos, exceto a variável alvo
labels = preprocessed_data['target']  # Variável alvo

# Divisão dos conjuntos de dados (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

# Exibir tamanhos dos conjuntos
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

##Normalização
#  colunas numéricas para normalização
numeral_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Criar o normalizador Min-Max
scaler = MinMaxScaler()

# Aplicar normalização apenas nos dados de treino (evitando Data Leakage)
X_train[numeral_columns] = scaler.fit_transform(X_train[numeral_columns])

# Aplicar a transformação nos dados de teste (usando os parâmetros do treino!)
X_test[numeral_columns] = scaler.transform(X_test[numeral_columns])

print("Normalização aplicada com sucesso!")

#Balanceamento
# Criar o balanceador SMOTE
smote = SMOTE(random_state=42)

# Aplicar SMOTE apenas nos dados de treino
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verificar a nova distribuição de `target` no treino balanceado
print("Distribuição da target após SMOTE (treino):")
print(y_train_resampled.value_counts(normalize=True) * 100)