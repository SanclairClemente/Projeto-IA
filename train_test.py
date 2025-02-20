import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Carregar os dados pré-processados
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

## Normalização
# Colunas numéricas para normalização
numeral_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Criar o normalizador Min-Max
scaler = MinMaxScaler()

# Aplicar normalização apenas nos dados de treino (evitando Data Leakage)
X_train[numeral_columns] = scaler.fit_transform(X_train[numeral_columns])

# Aplicar a transformação nos dados de teste (usando os parâmetros do treino!)
X_test[numeral_columns] = scaler.transform(X_test[numeral_columns])

print("Normalização aplicada com sucesso!")

# Balanceamento
# Criar o balanceador SMOTE
smote = SMOTE(random_state=42)

# Aplicar SMOTE apenas nos dados de treino
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verificar a nova distribuição de `target` no treino balanceado
print("Distribuição da target após SMOTE (treino):")
print(y_train_resampled.value_counts(normalize=True) * 100)

# Lista de modelos a serem testados
models = {
    "Árvore de Decisão": DecisionTreeClassifier(),
    "Naïve Bayes": GaussianNB(),
    "K-Vizinhos Mais Próximos (K-NN)": KNeighborsClassifier()
}

# Número de folds (10) e número de execuções (5)
n_folds = 10
n_runs = 5

def run_experiment(X, y, model, n_folds=10, n_runs=5):
    accuracy_results = []
    f1_results = []
    time_results = []
    
    for _ in range(n_runs):
        # Criar o validador de 10-fold cross-validation com distribuição estratificada
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Medir o tempo de execução
        start_time = time.time()
        
        # Rodar a validação cruzada com previsões
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Acurácia balanceada
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        accuracy_results.append(balanced_accuracy)
        
        # F1-Score
        f1 = f1_score(y, y_pred)
        f1_results.append(f1)
        
        # Tempo de execução
        end_time = time.time()
        time_results.append(end_time - start_time)
    
    # Retornar as médias e desvios padrão das métricas
    return {
        "mean_accuracy": sum(accuracy_results) / len(accuracy_results),
        "mean_f1_score": sum(f1_results) / len(f1_results),
        "mean_time": sum(time_results) / len(time_results),
        "std_accuracy": (sum([(x - (sum(accuracy_results) / len(accuracy_results)))**2 for x in accuracy_results]) / len(accuracy_results))**0.5,
        "std_f1_score": (sum([(x - (sum(f1_results) / len(f1_results)))**2 for x in f1_results]) / len(f1_results))**0.5,
        "std_time": (sum([(x - (sum(time_results) / len(time_results)))**2 for x in time_results]) / len(time_results))**0.5
    }

# Executando os experimentos para cada modelo
for name, model in models.items():
    print(f"\nResultados para o modelo: {name}")
    
    # Rodar o experimento com dados balanceados (X_train_resampled e y_train_resampled)
    results = run_experiment(X_train_resampled, y_train_resampled, model)
    
    # Exibindo resultados
    print(f"Acurácia média: {results['mean_accuracy']:.2f}")
    print(f"Desvio padrão da acurácia: {results['std_accuracy']:.2f}")
    print(f"F1-Score médio: {results['mean_f1_score']:.2f}")
    print(f"Desvio padrão do F1-Score: {results['std_f1_score']:.2f}")
    print(f"Tempo médio de execução: {results['mean_time']:.2f} segundos")
    print(f"Desvio padrão do tempo de execução: {results['std_time']:.2f} segundos")
