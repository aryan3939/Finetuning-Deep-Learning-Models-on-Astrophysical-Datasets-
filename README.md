# Galaxy Image Classification using ResNet and LoRA

This project focuses on classifying galaxy images into two main categories using deep learning techniques. It leverages the ResNet architecture for feature extraction and explores both full fine-tuning and efficient fine-tuning using Low-Rank Adaptation (LoRA). The dataset used is from the Galaxy Zoo challenge, which provides images and corresponding labels for galaxy classification.

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

### Dataset

The dataset used is from the Galaxy Zoo challenge, which includes:

- Training images: Located in `images_training_rev1`
- Test images: Located in `images_test_rev1`
- Labels: Provided in `training_solutions_rev1.csv`

The CSV file contains probabilities for various classes, and for this project, we focus on the top two classes: 'Class6.2' and 'Class1.2'.

### Data Preprocessing

- The GalaxyID in the CSV is modified to include the '.jpg' extension.
- The dataset is filtered to include only the top two classes.
- Labels are mapped to 0 and 1 for binary classification.
- Images are resized to 224x224, converted to tensors, and normalized using standard ImageNet means and standard deviations.

### Model Architecture

- **Base Model:** Pre-trained ResNet-50 from Hugging Face's transformers library.
- **Fine-Tuning:** The entire model is fine-tuned on the training data.
- **LoRA Adaptation:** A custom LoRA layer is added to the ResNet architecture for efficient fine-tuning, focusing on low-rank updates to the weights.

### Training

- **Fine-Tuning:**
  - Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-4.
  - Scheduler: Cosine annealing scheduler.
  - Epochs: 7
  - Best validation accuracy achieved: 88.17%

- **LoRA Fine-Tuning:**
  - A custom LoRA layer is integrated into the ResNet model.
  - Optimizer: Adam with learning rate 1e-3.
  - Scheduler: StepLR with step size 10 and gamma 0.1.
  - Epochs: 10
  - Best validation accuracy achieved: 74.12%

### Evaluation

- The models are evaluated on the test set using accuracy as the metric.
- For the direct evaluation of the pre-trained ResNet, the accuracy is 68.94%.
- After fine-tuning, the accuracy improves to 88.52%.
- Using LoRA, the accuracy is 73.93%.

### Results

| Method                           | Test Accuracy |
|----------------------------------|---------------|
| Pre-trained ResNet (Direct)      | 68.94%        |
| Fine-Tuned ResNet                | 88.52%        |
| LoRA Adapted ResNet              | 73.93%        |

The fine-tuned ResNet model significantly outperforms the directly evaluated pre-trained model, while the LoRA adaptation provides a balance between efficiency and performance.

### Usage

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from the Galaxy Zoo challenge and place it in the appropriate directories.
4. Run the provided Jupyter notebook to train and evaluate the models.

**Note:** The code in this repository was originally developed and run on Kaggle. If running locally, ensure that the dataset paths are correctly set and that you have sufficient computational resources, especially for training the models.

**Hardware Requirements:**

- A GPU is recommended for training the models efficiently. The code is compatible with CUDA-enabled GPUs.

### Requirements

The project requires the following Python libraries:

- torch
- torchvision
- transformers
- peft
- loralib
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Pillow

You can install them using:
```bash
pip install torch torchvision transformers peft loralib pandas numpy matplotlib seaborn scikit-learn Pillow
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.