# Fractional-Order-Activation-Function
Enhancing Neural Networks for Medical Diagnosis Using Fractional Order Activation Functions
---------------------------------------------------------------------------------------------------------------------------------------------------------
Download the Fractional-Order-Activation-Function.zip
---------------------------------------------------------------------------------------------------------------------------------------------------------

This Python application trains a basic feedforward neural network 30 times for each of the six activation functions including three custom fractional-order functions—and logs the outcomes. 
Parallel execution, excel result saving, and a final summary report featuring a performance plot are all supported.

---------------------------------------------------------------------------------------------------------------------------------------------------------

Features

- Both custom (FracSigmoid, FracReLU, and FracTanh) and standard (Sigmoid, ReLU, and Tanh) activation functions are supported.
- To improve learning behavior, fractional-order derivatives are used with custom activations.
- Manages 
	Multithreaded execution for training efficiency
	Binary and multi-class classification
	Early stopping during training
	Data loading
	Encoding
	Scaling
- Results are saved to the database.xlsx and an overview in summary.xlsx
- Accuracy_plot.png shows trends in accuracy

---------------------------------------------------------------------------------------------------------------------------------------------------------

Activation Functions Tested

| Name         | Type       | Description                                  |
|--------------|------------|----------------------------------------------|
| Sigmoid      | Standard   | Traditional sigmoid activation               |
| ReLU         | Standard   | Rectified Linear Unit                        |
| Tanh         | Standard   | Hyperbolic tangent                           |
| FracSigmoid  | Fractional | Modified sigmoid using fractional calculus   |
| FracReLU     | Fractional | ReLU with fractional-order gradient          |
| FracTanh     | Fractional | Derived from FracSigmoid: `2σ(2x) - 1`       |

---------------------------------------------------------------------------------------------------------------------------------------------------------

Requirements

Install dependencies 

pip install pandas numpy scikit-learn torch openpyxl matplotlib

---------------------------------------------------------------------------------------------------------------------------------------------------------

How to Run

1. Place your dataset (e.g., Anemia.csv) in the project directory.
2. Modify the csv_file path in the __main__ block if needed.
3. Run the script:

---------------------------------------------------------------------------------------------------------------------------------------------------------

After execution:

- Each activation function's training result (30 runs each) will be stored in a database.xlsx.
- Recall, accuracy, and precision will all be summarized and stored in summary.xlsx. 
- Accuracy_plot.png will display the accuracy trends.

---------------------------------------------------------------------------------------------------------------------------------------------------------

Output Example

- summary.xlsx contains:
  - min, max, and mean of accuracy, precision, and recall per activation function.

- database.xlsx contains 
  - stores detailed performance metrics from each training run, including accuracy, precision, recall, and the activation function used.

- accuracy_plot.png shows:
  - Accuracy per run for each activation function over 30 trials.

---------------------------------------------------------------------------------------------------------------------------------------------------------

Customization Tips

- Change the loop in train_multiple() to alter the number of runs.
- To use a different dataset, substitute your dataset for Anemia.csv and, if necessary, modify the preprocessing.
- Modify the alpha values for every custom activation in the activations list to adjust fractional parameters.

---------------------------------------------------------------------------------------------------------------------------------------------------------

