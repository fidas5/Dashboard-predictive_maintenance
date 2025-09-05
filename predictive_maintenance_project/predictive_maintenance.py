
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("predictive_maintenance.csv")

# Renommer les colonnes pour faciliter l'accès (supprimer les caractères spéciaux)
df.columns = [col.replace("[K]", "_K").replace("[rpm]", "_rpm").replace("[Nm]", "_Nm").replace("[min]", "_min").replace(" ", "_") for col in df.columns]

# Debug: Afficher les noms de colonnes après renommage
print("Noms des colonnes après renommage:", df.columns)

# Séparer les caractéristiques (X) et la cible (y)
X = df.drop(["UDI", "Product_ID", "Target", "Failure_Type"], axis=1)
y = df["Target"]

# Identifier les colonnes numériques et catégorielles
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Créer les pipelines de prétraitement
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Créer un préprocesseur qui applique les transformations aux colonnes appropriées
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Prétraitement initial terminé. Données divisées en ensembles d'entraînement et de test.")
print(f"Forme de X_train: {X_train.shape}")
print(f"Forme de X_test: {X_test.shape}")
print(f"Forme de y_train: {y_train.shape}")
print(f"Forme de y_test: {y_test.shape}")

# Créer le pipeline complet avec le préprocesseur et le modèle
model = Pipeline(steps=[("preprocessor", preprocessor),
                      ("classifier", RandomForestClassifier(random_state=42))])

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
print("\nMatrice de confusion sauvegardée dans confusion_matrix.png")

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
print("Courbe ROC sauvegardée dans roc_curve.png")

# Nuages de points de vibration vs température
# Utiliser le dataframe original df pour les visualisations car X ne contient pas toutes les colonnes
plt.figure(figsize=(10, 7))
sns.scatterplot(x="Rotational_speed__rpm", y="Process_temperature__K", hue="Target", data=df, palette="coolwarm", alpha=0.6)
plt.title("Vitesse de Rotation vs Température du Processus (avec Cible)")
plt.xlabel("Vitesse de Rotation [rpm]")
plt.ylabel("Température du Processus [K]")
plt.savefig("rotational_speed_vs_process_temperature.png")
print("Nuage de points Vitesse de Rotation vs Température du Processus sauvegardé dans rotational_speed_vs_process_temperature.png")

plt.figure(figsize=(10, 7))
sns.scatterplot(x="Torque__Nm", y="Process_temperature__K", hue="Target", data=df, palette="coolwarm", alpha=0.6)
plt.title("Couple vs Température du Processus (avec Cible)")
plt.xlabel("Couple [Nm]")
plt.ylabel("Température du Processus [K]")
plt.savefig("torque_vs_process_temperature.png")
print("Nuage de points Couple vs Température du Processus sauvegardé dans torque_vs_process_temperature.png")


