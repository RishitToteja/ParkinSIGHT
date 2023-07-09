# Project Description

This project is focused on the identification of Parkinson’s Disease (PD) at an early stage and differentiating PD from healthy (Control) and the SWEDD. To achieve this goal, we've broken down our areas of research into the following steps.

## Flow of Work

1. **Dataset:** We utilized the PPMI dataset, which comprises SPECT scans from individuals categorized into three classes: PD (Parkinson's disease), Control, and SWEDD (Scans Without Evidence of Dopaminergic Deficit). To construct the image data used in our analysis, we calculated the average of the 42nd, 43rd, and 44th slices extracted from the SPECT scans.
2. **Data preprocessing:** We performed the necessary preprocessing steps such as image resizing, normalization, and noise reduction to enhance the quality and consistency of the SPECT scans.
3. **Feature extraction:** We extracted meaningful features from the SPECT scans that helped us differentiate between healthy and Parkinson's affected brains. 
Common features include the use of X,Y coordinates of the segmented curve, intensity-based statistics, texture descriptors, and shape-based measurements. 
4. **Feature selection:** We employed feature selection techniques to identify the most informative features for the classification task, reducing dimensionality and improving the efficiency of machine learning algorithms.
5. **Algorithm selection:** We experimented with different machine learning algorithms such as Support Vector Machines (SVM), Logistic Regression, Random Forests, and XGBoost. 
We compared their performance in terms of accuracy, precision, recall, and F1-score to determine the most suitable algorithm for Parkinson's detection.
6. **Training and validation:** We split the dataset into training and validation sets. We utilized a portion (e.g. 80%) of the data for training and the remaining portion for validation and testing and implemented cross-validation techniques like stratified k-fold validation to ensure robustness and minimize overfitting.
7. **Performance evaluation:** We assessed the performance of the machine learning model using appropriate evaluation metrics, including accuracy, recall, specificity, F1-score, and area under the Receiver Operating Characteristic (ROC) curve to ensure effective detection of Parkinson's disease.
8. **Hyperparameter tuning:** We fine-tuned the model's hyperparameters using grid search to optimize the model's performance.

## ORIGINAL DATASET
<div align="center">
<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/original_dataset.png" width="548" height="657">
</div>

## SEGMENTED DATASET
<div align="center">
<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/segmented_dataset.png" width="548" height="657">
</div>

## Active Contour Segmentation

We used a Gaussian filter to reduce noise and enhance the quality of the data.

Active contour models, commonly referred to as snakes, are energy-based deformable models that iteratively evolve to fit the contours of objects within an image. These models are driven by internal and external energy terms, which can be customized based on the specific segmentation task. By minimizing the overall energy, the snake adapts its shape to closely match the boundaries of the targeted structure, even in the presence of noise or intensity variations.

### Substantia Nigra Segmentation

The substantia nigra is a critical region affected by PD. Active contour segmentation allows precise delineation of the substantia nigra from structural or functional brain images.


### Feature Extraction from Contours

Active contour segmentation enables the extraction of relevant biomarkers from medical images. These biomarkers can include shape descriptors, intensity statistics, or spatial distribution features of segmented regions. By applying machine learning algorithms to these biomarkers, This can be used to develop predictive models for early PD detection.


## Feature Matrix Formation

We extracted information from the Active Contour Segmentation curve by uniformly sampling six points on each side (left and right) of the curves (as shown in the image below). These points are then flattened into separate x and y coordinates, resulting in a 24-dimensional feature representation for each image.

## Autoencoder Training

Autoencoder consisted of an encoder network and a decoder network. The encoder compressed the input data into a lower-dimensional representation, while the decoder reconstructed the original input from the compressed representation. The compact representation obtained from the bottleneck layer of the autoencoder served as the reduced-dimensional representation of the input data, which was used for subsequent analysis.

Through the application of overfitting, we successfully validated the autoencoder with a dimensionality of eight, thus enabled the reconstruction of our feature matrix from the ensuing input.

The hypothesis in our project was that without applying dimensionality reduction techniques, the performance of our Parkinson's disease detection model using machine learning would be lower compared to when using techniques such as autoencoders and PCA.

The rationale behind this hypothesis was that high-dimensional data, like brain SPECT images, often contain redundant or irrelevant information. Without dimensionality reduction, the model may struggle to effectively distinguish between relevant and irrelevant features, which leads to decreased performance.

By incorporating techniques such as autoencoders and PCA, the hypothesis assumed that the model could overcome the challenges posed by high-dimensional data. Autoencoders could learn compact representations of the input data through unsupervised learning, capturing the most essential features. Additionally, PCA could further reduce the dimensionality by identifying the principal components that explain the majority of the data's variance.

By utilizing these dimensionality reduction techniques, the model was expected to focus on the most informative aspects of the data and enhance its ability to discriminate between healthy and Parkinson's affected brains. As a result, the hypothesis suggested that the model's performance, measured in terms of accuracy, sensitivity, specificity, or other evaluation metrics, was higher when utilizing techniques such as autoencoders and PCA were used as compared to not applying any dimensionality reduction techniques.


<div align="center">
<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/Block%20Diagram%201.png">
</div>
</br>
<div align="center">
<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/Block%20Diagram%202.png">
</div>

## Machine Learning Algorithms

- **SVM-Linear:** SVM with linear kernel is a powerful machine learning algorithm used for binary classification. It finds the optimal linear hyperplane that separates data points of different classes, making it effective for linearly separable datasets. SVM-Linear aims to maximize the margin between the classes, leading to robust and interpretable decision boundaries.

- **SVM-RBF:** SVM with Radial Basis Function (RBF) kernel is a versatile algorithm suitable for both linearly and nonlinearly separable datasets. It projects data into a higher-dimensional space to find non-linear decision boundaries. The RBF kernel allows SVM to capture complex relationships and has adjustable parameters for regularization and kernel width, making it flexible and effective for a wide range of applications.

- **Random Forest:** Random Forest is an ensemble learning algorithm that combines multiple decision trees to create a robust and accurate model. Each tree in the forest is trained on a random subset of the data and features. Random Forest leverages the power of averaging and features randomness to reduce overfitting and improve generalization. It is capable of handling high-dimensional data and provides feature importance analysis.

- **XGBoost:** XGBoost (Extreme Gradient Boosting) is a boosting algorithm known for its exceptional performance in various machine learning tasks. It sequentially builds an ensemble of weak prediction models, typically decision trees, to minimize a loss function. XGBoost employs gradient boosting, which optimises the model by adding new trees that focus on correcting the mistakes made by previous trees. It handles complex relationships and is highly efficient, often achieving state-of-the-art results.

- **Logistic Regression:** Logistic Regression is a popular statistical learning algorithm used for binary classification problems. Despite its name, it is actually a regression model that uses the logistic function (sigmoid) to map predicted values to probabilities. Logistic Regression models the relationship between the input features and the probability of belonging to a specific class. It is interpretable, computationally efficient, and performs well when the decision boundary is linear or can be approximated by a linear function.

## Final Results
## Original Feature Matrix (no. of features = 24)
### Early PD vs Control

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/Original%201.png">

### Early PD vs SWEDD

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/Original%202.png">

### Principal Component Analysis
### Early PD vs Control

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/PCA%201.png">

### Early PD vs SWEDD

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/PCA%202.png">

### Autoencoder 
### Early PD vs Control

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/AE%201.png">

### Early PD vs SWEDD

<img src="https://github.com/RishitToteja/PDVisionAI/blob/main/images/AE%202.png">


