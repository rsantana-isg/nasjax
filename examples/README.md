# NASJAX Examples

This directory contains example scripts demonstrating how to use NASJAX for neural architecture search and evolution.

## Classification Examples

### MLP Classification (`evolve_mlp_classification.py`)

Demonstrates evolving Multi-Layer Perceptron architectures for classification using the **Iris dataset** from scikit-learn.

**Features:**
- Simple 3-class classification problem (Iris species)
- Low complexity networks (max 3 layers, 30 neurons per layer)
- Evolutionary algorithm with tournament selection and mutation
- Fitness based on classification accuracy
- Small dataset ideal for quick experimentation

**Usage:**
```bash
python examples/evolve_mlp_classification.py
```

**Expected Results:**
- Achieves 93-98% accuracy on test set
- Typically converges in 4-5 generations
- Evolution time: ~2-5 minutes on CPU

**Dataset:**
- 150 samples (105 train, 45 test)
- 4 features (sepal/petal dimensions)
- 3 classes (Iris setosa, versicolor, virginica)

---

### Image Classification (`evolve_cnn_classification.py`)

Demonstrates evolving networks for image classification using the **Digits dataset** from scikit-learn.

**Features:**
- 10-class digit recognition (0-9)
- 8x8 grayscale images (flattened to 64 features for MLPs)
- Low complexity networks (max 3 layers, 64 neurons per layer)
- Evolutionary algorithm with elitism
- Fitness based on classification accuracy
- Larger dataset for more challenging problem

**Usage:**
```bash
python examples/evolve_cnn_classification.py
```

**Expected Results:**
- Achieves 95-98% accuracy on test set
- May achieve high accuracy in first generation
- Evolution time: ~5-10 minutes on CPU

**Dataset:**
- 1,797 samples (1,347 train, 450 test)
- 64 features (8x8 images flattened)
- 10 classes (digits 0-9)

**Note:** This example currently uses MLPs with flattened images. Future versions of NASJAX will include full CNN support with convolutional layers for more efficient image processing.

---

## Evolutionary Algorithm Parameters

Both examples use similar evolutionary algorithm configurations:

| Parameter | MLP Example | CNN/Image Example | Description |
|-----------|-------------|-------------------|-------------|
| Population Size | 8 | 6 | Number of individuals per generation |
| Generations | 5 | 4 | Number of evolutionary iterations |
| Max Layers | 3 | 3 | Maximum hidden layers |
| Max Neurons | 30 | 64 | Maximum neurons per layer |
| Mutation Prob | 0.8 | 0.8 | Probability of mutation |
| Training Epochs | 50 | 30 | Epochs per fitness evaluation |
| Batch Size | 16 | 32 | Training batch size |

These parameters are tuned for relatively **low complexity** networks that can be trained quickly while still achieving good performance.

---

## Common Patterns

Both examples follow a similar structure:

1. **Data Loading & Preprocessing**
   - Load dataset from scikit-learn
   - Split into train/test sets
   - Standardize features
   - Convert to JAX arrays

2. **Fitness Evaluation**
   - Build network from descriptor
   - Train on training data
   - Evaluate on test data
   - Return fitness (1 - accuracy)

3. **Evolution Loop**
   - Initialize random population
   - For each generation:
     - Evaluate all individuals
     - Select best (elitism)
     - Create offspring via mutation
     - Report progress

4. **Results Display**
   - Show best architecture found
   - Display accuracy progression
   - Print final configuration

---

## Customization

You can customize the evolution process by modifying:

### Network Complexity
```python
max_num_layers=5,    # Allow deeper networks
max_num_neurons=100, # Allow wider layers
```

### Evolution Parameters
```python
population_size=20,  # Larger population for more diversity
n_generations=10,    # More generations for better results
mutation_prob=0.6,   # Lower mutation for exploitation
```

### Training Settings
```python
n_epochs=100,        # More training per evaluation
learning_rate=0.01,  # Adjust learning rate
optimizer="adamw",   # Try different optimizers
```

### Dataset Selection
Use other scikit-learn datasets:
```python
from sklearn.datasets import load_wine, load_breast_cancer, load_digits

# For classification:
data = load_wine()        # 13 features, 3 classes
data = load_breast_cancer()  # 30 features, 2 classes
```

---

## Other Examples in This Directory

- `minimal_example.py` - Basic demonstration of NASJAX components
- `simple.py` - Simple MLP example from the original DEATF project (reference implementation)
- `cnn_class.py` - CNN classification example from the original DEATF project (reference)
- Other examples from the original project for reference

---

## Requirements

All examples require:
- JAX >= 0.4.20
- Equinox >= 0.11.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

Install with:
```bash
pip install -e .
```

---

## Tips for Best Results

1. **Start Small**: Begin with small populations and few generations to understand behavior
2. **Monitor Progress**: Watch accuracy progression to see if evolution is working
3. **Adjust Complexity**: Increase max_layers/neurons if accuracy plateaus
4. **Balance Training**: More epochs = better fitness but slower evolution
5. **Use Standardization**: Always standardize/normalize input features
6. **Set Seeds**: Use random seeds for reproducibility

---

## Future Enhancements

Upcoming NASJAX features:
- True CNN support with convolutional layers
- RNN/LSTM evolution for sequence tasks
- Multi-objective optimization (accuracy + size)
- Crossover operators for architecture search
- Parallel population evaluation
- Advanced selection strategies
- Hyperparameter co-evolution

---

## Contributing

To add new examples:
1. Follow the existing structure
2. Include comprehensive docstrings
3. Use relatively simple datasets
4. Keep complexity low for quick testing
5. Add documentation to this README

---

For more information, see the main [NASJAX README](../README.md).
