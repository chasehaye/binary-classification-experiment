import os
import numpy as np
import pandas as pd
from data_preprocessor import preprocess_and_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

data_dir = "data_storage"
os.makedirs("models", exist_ok=True)
output_path = "models/experiment_results.csv"
default_file = "GSPC_clean_data.csv"

print("************************************************")
print("--- Machine Learning Data Loader ---")
file_prompt = input(f"Enter the CLEANED filename to load (default: {default_file}): ").strip()
target_file = file_prompt if file_prompt else default_file
if not target_file.lower().endswith('.csv'):
    target_file += '.csv'

input_path = os.path.join(data_dir, target_file)

X_train, X_test, y_train, y_test = preprocess_and_split(input_path)




print("************************************************")
print(f"MODEL TRAINING")
print("************************************************")



def build_model(input_dim, hidden_layers, activation, lr, dropout, optimizer_type):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Dropout(dropout * (0.7 ** i)))

    model.add(layers.Dense(1, activation="sigmoid"))


    if optimizer_type == "adam":
        opt = Adam(learning_rate=lr)
    elif optimizer_type == "rmsprop":
        opt = RMSprop(learning_rate=lr)
    elif optimizer_type == "nadam":
        opt = Nadam(learning_rate=lr)
    elif optimizer_type == "adamw":
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    else:
        opt = Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model


architectures = [[64, 32], [128, 64, 32], [256, 128, 64, 32]]
activations = ["relu", "elu", "swish"]
optimizers = ["adam", "rmsprop", "nadam", "adamw"]
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [16, 32, 64]
dropouts = [0.2, 0.3, 0.4]
seeds = [1, 2, 3]

results = []

print("\n************************************************")
print(f"STARTING FULL EXPERIMENT GRID")
print(f"Total Unique Configs: {len(architectures)*len(activations)*len(optimizers)*len(learning_rates)*len(batch_sizes)*len(dropouts)}")
print("************************************************\n")


for lr in learning_rates:
    for batch in batch_sizes:
        for dropout in dropouts:
            for arch in architectures:
                for act in activations:
                    for opt in optimizers:
                        
                        run_aucs = []
                        
                        for seed in seeds:
                            tf.keras.backend.clear_session()
                            tf.random.set_seed(seed)
                            np.random.seed(seed)

                            model = build_model(
                                input_dim=X_train.shape[1],
                                hidden_layers=arch,
                                activation=act,
                                lr=lr,
                                dropout=dropout,
                                optimizer_type=opt
                            )

                            early_stop = callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=5,
                                restore_best_weights=True
                            )

                            model.fit(
                                X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=30,
                                batch_size=batch,
                                verbose=0,
                                shuffle=True,
                                callbacks=[early_stop]
                            )

                            _, auc = model.evaluate(X_test, y_test, verbose=0)
                            run_aucs.append(auc)

                        mean_auc = np.mean(run_aucs)
                        std_auc = np.std(run_aucs)

                        results.append({
                            "lr": lr,
                            "batch": batch,
                            "dropout": dropout,
                            "arch": str(arch),
                            "activation": act,
                            "optimizer": opt,
                            "auc_mean": mean_auc,
                            "auc_std": std_auc
                        })

                        print(f"LR:{lr} | Batch:{batch} | Drop:{dropout} | Arch:{arch} | Act:{act} | Opt:{opt} | AUC:{mean_auc:.4f}")
                        
                        if len(results) % 5 == 0:
                            pd.DataFrame(results).to_csv(output_path, index=False)

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(by="auc_mean", ascending=False)
    results_df.to_csv(output_path, index=False)
    print("\nSaved final experiment results to:", output_path)
    print("\n==========================")
    print("TOP 10 RESULTS")
    print("==========================")
    print(results_df.head(10))

    best = results_df.iloc[0]
    print("\n==========================")
    print("BEST CONFIGURATION")
    print("==========================")
    print(best)
else:
    print("No results to display.")