# Importing standard libraries for data handling, numerical operations, and system utilities
import pandas as pd                      # For data manipulation and analysis
import numpy as np                       # For numerical computing
import torch                             # Main PyTorch library
import torch.nn as nn                    # For neural network components
import torch.optim as optim              # For optimization algorithms
from sklearn.model_selection import train_test_split        # For splitting dataset into training and test sets
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # For preprocessing
from sklearn.compose import ColumnTransformer             # For handling mixed types (categorical + numerical)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  # Evaluation metrics
import datetime                           # For timestamps
import os                                 # File operations
import warnings                           # To suppress warnings
import time                               # To measure training time
from concurrent.futures import ThreadPoolExecutor          # For concurrent (parallel) execution
import threading                          # For thread-safe operations
import matplotlib.pyplot as plt           # For plotting accuracy graph

# Disable warnings to keep the output clean
warnings.filterwarnings('ignore')

# Thread-safe lock to prevent simultaneous writes to Excel file
write_lock = threading.Lock()

# ----------------------------- Custom Activation Functions -----------------------------

# Custom Fractional Sigmoid Activation Function
class FractionalSigmoid(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Fractional coefficient

    def forward(self, t):
        alpha = self.alpha
        t_pos = torch.clamp(t, min=1e-7)  # Prevent division by zero or negative roots
        exp_neg_t = torch.exp(-t)
        return 1 + (1 - alpha * exp_neg_t - alpha * torch.pow(t_pos, 1 - alpha) * exp_neg_t)

# Custom Fractional Tanh Activation Function using FracSigmoid
class FractionalTanh(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        self.frac_sigmoid = FractionalSigmoid(alpha)

    def forward(self, t):
        return 2 * self.frac_sigmoid(2 * t) - 1  # Derivation from tanh(x) = 2σ(2x) - 1

# Custom autograd-based Fractional ReLU
class FractionalReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)  # Save input for use in backward pass
        ctx.alpha = alpha             # Save alpha for gradient calculation
        return torch.clamp(input, min=0)  # Standard ReLU operation

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        grad_input[input > 0] *= input[input > 0].pow(1 - alpha)  # Fractional gradient
        return grad_input, None

# Wrapper for FractionalReLU using nn.Module
class FractionalReLU(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return FractionalReLUFunction.apply(x, self.alpha)

# ----------------------------- Data Loading & Preprocessing -----------------------------

def load_and_preprocess(csv_file):
    # Load CSV with fallback encoding
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='latin-1')

    # Convert all columns to numeric if possible
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

    # Drop columns with >50% missing values and then drop remaining rows with NaNs
    df.dropna(axis=1, thresh=int(0.5 * len(df)), inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("No data left after cleaning")

    # Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Filter test labels to known classes
    mask = y_test.isin(le.classes_)
    y_test = y_test[mask]
    X_test = X_test[mask]
    y_test_enc = le.transform(y_test)

    # Separate numerical and categorical features
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Build preprocessor pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

    # Apply transformations and convert to float32
    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    return X_train_proc, X_test_proc, y_train_enc, y_test_enc, len(le.classes_)

# ----------------------------- Neural Network Model -----------------------------

# Simple 3-layer feedforward NN with customizable activation
class SimpleNN(nn.Module):
    def __init__(self, input_dim, activation, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            activation,
            nn.Linear(64, 32),
            activation,
        )
        # Output: 1 for binary, num_classes for multi-class
        self.output_layer = nn.Linear(32, 1 if num_classes == 2 else num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.net(x)
        return self.output_layer(x)

# ----------------------------- Training & Evaluation -----------------------------

# Train model with early stopping
def train_model(X_train, y_train, activation, num_classes, epochs=100, lr=0.001):
    model = SimpleNN(X_train.shape[1], activation, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train)

    # Set target tensor and loss function
    if num_classes == 2:
        y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
    else:
        y_train_t = torch.from_numpy(y_train).long()
        criterion = nn.CrossEntropyLoss()

    model.train()
    best_loss = float('inf')
    counter = 0

    # Early stopping after 10 consecutive non-improvements
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            if counter >= 10:
                break

    return model

# Evaluate model using multiple metrics
def evaluate(model, X_test, y_test, num_classes):
    model.eval()
    X_test_t = torch.from_numpy(X_test)
    with torch.no_grad():
        outputs = model(X_test_t)

    if num_classes == 2:
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_true = y_test
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = float('nan')
    else:
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true = y_test
        try:
            auc = roc_auc_score(y_true, probs, multi_class='ovr')
        except:
            auc = float('nan')

    return {
        'accuracy': np.mean(preds == y_true),
        'precision': precision_score(y_true, preds, average='weighted', zero_division=0),
        'recall': recall_score(y_true, preds, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, preds, average='weighted', zero_division=0),
        'roc_auc': auc
    }

# ----------------------------- Results Logging -----------------------------

# Save results to Excel safely across threads
def save_results(results):
    file = "database.xlsx"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create DataFrame for new results
    df_new = pd.DataFrame([{
        'timestamp': now,
        'filename': results['filename'],
        'activation': results['activation'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'roc_auc': results['roc_auc'],
        'training_time': str(results['time']),
        'classes': results['classes'],
        'alpha': results.get('alpha', None)
    }])

    with write_lock:
        # Load or initialize the Excel file
        if os.path.exists(file):
            try:
                df_existing = pd.read_excel(file, engine='openpyxl')
            except:
                df_existing = pd.DataFrame()
        else:
            df_existing = pd.DataFrame()

        # Append new results and save
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.to_excel(file, index=False, engine='openpyxl')

# ----------------------------- Parallel Training -----------------------------

# Run 30 trainings per activation function in a loop
def train_multiple(name, activation, alpha, X_train, y_train, X_test, y_test, n_classes, csv_file):
    print(f"\n=== Starting: {name} ===")
    for i in range(1, 31):
        print(f"[{name}] Run {i}/30")
        start = time.time()
        try:
            model = train_model(X_train, y_train, activation, n_classes)
            metrics = evaluate(model, X_test, y_test, n_classes)
            elapsed = time.time() - start

            save_results({
                'filename': os.path.basename(csv_file),
                'activation': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'time': elapsed,
                'classes': n_classes,
                'alpha': alpha
            })
        except Exception as e:
            print(f"Error on {name} run {i}: {e}")

# ----------------------------- Summary & Visualization -----------------------------

# Summarize all results and plot accuracy
def summarize_and_plot():
    df = pd.read_excel("database.xlsx", engine='openpyxl')
    grouped = df.groupby('activation')
    summary = grouped[['accuracy', 'precision', 'recall']].agg(['min', 'max', 'mean'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index().to_excel("summary.xlsx", index=False)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('activation'):
        plt.plot(group['accuracy'].values, label=name)
    plt.xlabel("Run Index")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Activation Function (30 Runs Each)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()

# ----------------------------- Main Script -----------------------------

if __name__ == "__main__":
    csv_file = "Anemia.csv"  # Change this to your actual dataset path

    # Load and preprocess dataset
    try:
        X_train, X_test, y_train, y_test, n_classes = load_and_preprocess(csv_file)
    except Exception as e:
        print(f"Failed to load/process data: {e}")
        exit(1)

    # Define all activation functions (standard and fractional)
    activations = [
        ('Sigmoid', nn.Sigmoid(), None),
        ('ReLU', nn.ReLU(), None),
        ('Tanh', nn.Tanh(), None),
        ('FracSigmoid', FractionalSigmoid(alpha=0.5), 0.5),
        ('FracReLU', FractionalReLU(alpha=0.3), 0.3),
        ('FracTanh', FractionalTanh(alpha=0.6), 0.6),
    ]

    # Launch all training tasks in parallel threads
    with ThreadPoolExecutor(max_workers=len(activations)) as executor:
        futures = [
            executor.submit(train_multiple, name, act, alpha, X_train, y_train, X_test, y_test, n_classes, csv_file)
            for name, act, alpha in activations
        ]
        for f in futures:
            f.result()  # Wait for each thread to complete

    # Summarize results and plot accuracy graph
    summarize_and_plot()
    print("\n✅ All training complete. Summary saved to 'summary.xlsx', plot saved to 'accuracy_plot.png'")
