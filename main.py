import os
from src.data_preprocessing import *
from src.train_models import *
from src.evaluate import evaluate_model
from src.utils import plot_model_comparison, plot_confusion_matrices

def main():
    # 1. Load and prepare data
    data_path = 'data/customer_retail.csv'   # adjust path if needed
    if not os.path.exists(data_path):
        data_path = 'customer_retail.csv'    # fallback to current directory
    print("Loading data...")
    df_raw = load_and_clean_data(data_path)
    customer_df = create_customer_features(df_raw)
    customer_df, _ = encode_country(customer_df)
    X, y = prepare_features(customer_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Scale for LR and KNN
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 3. Train models
    print("\nLogistic Regression")
    lr_model, lr_params = train_logistic_regression(X_train_scaled, y_train)
    print("Best params:", lr_params)
    y_pred_lr, acc_lr = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")

    print("\nDecision Tree")
    dt_model, dt_params = train_decision_tree(X_train, y_train)
    print("Best params:", dt_params)
    y_pred_dt, acc_dt = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

    print("\nKNN")
    knn_model, knn_params = train_knn(X_train_scaled, y_train)
    print("Best params:", knn_params)
    y_pred_knn, acc_knn = evaluate_model(knn_model, X_test_scaled, y_test, "KNN")

    # 4. Compare
    model_names = ['Logistic Regression', 'Decision Tree', 'KNN']
    accuracies = [acc_lr, acc_dt, acc_knn]
    plot_model_comparison(model_names, accuracies)
    plot_confusion_matrices(y_test, [y_pred_lr, y_pred_dt, y_pred_knn], model_names)

    # 5. Best model
    best_idx = accuracies.index(max(accuracies))
    print(f"Best model: {model_names[best_idx]} with accuracy {accuracies[best_idx]:.4f}")

if __name__ == "__main__":
    main()