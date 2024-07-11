# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Charger le dataset
dataset_path = r"C:\Users\Rahma\Desktop\creditcard.csv"  
df = pd.read_csv(dataset_path)

# Diviser le dataset en ensemble d'apprentissage et ensemble de test
X = df.drop("Class", axis=1)  
y = df["Class"]  

# Exemple d'utilisation de la pondération des classes
decision_tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
decision_tree_model.fit(X_train, y_train)

# Afficher l'arbre de décision avec une taille de figure plus grande et une police plus grande
plt.figure(figsize=(16, 12))  # Ajustez la taille de la figure selon vos préférences
plot_tree(decision_tree_model, feature_names=X.columns, class_names=["Non-fraude", "Fraude"], filled=True, rounded=True, fontsize=6, max_depth=3)
plt.show()


# Prédiction sur l'ensemble de test
y_pred = decision_tree_model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Afficher les résultats
print(f"Précision : {accuracy}")
print("\nRapport de Classification :")
print(report)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(conf_matrix)

# Calculer la sensibilité (rappel) et la spécificité
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

print(f"Sensibilité (Rappel) : {sensitivity}")
print(f"Spécificité : {specificity}")

# Prédiction des probabilités pour la classe positive (fraude) sur l'ensemble de test
y_scores = decision_tree_model.predict_proba(X_test)[:, 1]

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()

# Courbe PR
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe PR (Aire sous la courbe = {:.2f})'.format(avg_precision))
plt.show()
