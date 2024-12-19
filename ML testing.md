### Types of testing: 
- responsible ai testing, post production testing, shift left testing, Prompting testing, Memory Testing ,Context testing, 

### Examples of ML: 
- Spam filters in emails,  Promotion recomendation engines by user preferences,  Predective maintainance for industrial equipments.

### Strengths and Limitations: 
- Flexiblity: Can get results even when data is not feeded to the ML model, because it already learned from the previous data and it analyzes with predections.
- Scalablity: Can give broader results than the conditional results from an expret based traditional systems.
- Interpretability: ML models have  less interpretability and we don't know the outputs (As it's a prediction by machine). In expert based systems, Reactions are defined.

### ML Model Life cycle: 
<img width="862" alt="image" src="https://github.com/user-attachments/assets/847194be-7d5c-4518-9164-538faa4b1d3d">

- Devs use Training data and testers have validation data.
- Validation data have 2 problems: Underfitting(Need more data training on seen data) and Overfitting(If a high accuracy model is trained on more related but un-useful data than needed(ex: giving house colour,designs,facing directions. model can't predict price accurately) efficiency goes down, ). 
- GoodFit: after UnderFitting, when model starts giving better results. Itis said a goodfit.
- Overfitting: Caused by:
      - over training from unrelated data (called noise & randomeness).
      - Can be caused by more model complexity.
      - Can be caused by non-regularization. To be corrected by Regularization techniques like - Lasso(L1) ,Ridge(L2).
      - Cross validation not implemented.   
- Labeled data vs Unlabeled data: When ML model have only input data and don't have output data called Un-labelled data ,When ML model have both Input and Output data called labelled data.

### Testing UnSUpervised data models(BlackBox functional testing):
1st step is Training on Training set. Then Training on Validation set.
Here, We have only un-labeled data. In **training data set**, model learns the needed patterns and structure in data. In **validation data set** ,input data is seprated into clusters of matching data(Training data set) .

### Silhouette Scores(For Whitebox testing): 
This score is from -1 to 1. This checks the new untrained data score of model. If data is matching matching to the similar data cluster, it is higher score. If score is not matching to the nearby cluster, it is low score and shows data far away from matching cluster.
Can ask developers ,the methods and utilities where to pass data with some python methods get Silhouette score for different test data. 
