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
  
## Supervised Learning: 
It is type of machine learning algorithm that learns from labeled data. Here, ML model needs a trainer to train model on the labeled data(with correct answer or tagged data). 
* Advantages: Lears from previous experiences, solves real time computational problems
* Disadvantages: Classifying big data can be challenging, Needs lots of computation time and resourses, requires labelled data set, needs training mentor. 
   
Types of supervised learning: 

#### Regression supervised learning: 
For contineous data prediction like house prices, stock prices, customer behaviour. Some common types are: 
* Linear regression
* Polynomial Regression
* Support Vector Machine Regression
* Decision Tree Regression
* Random Forest Regression

#### Classification supervised learning: 
Solves a classification problem where the output variable is a categtory like any colour, any disease, any other condition, email spam. It learns from input data and by using probablity distribution over output groups. Some common types are : 
* Linear Regression : For Predicting a continuous value like house prices.  Frameworks used for linear regression: Scikit Learn.
* Logistic regression: Used for Binary classification tasks like : mail Spam Detection ,loan approvals by learning probablity of binary outcome. Used for both. Framework used : Scikit Learn, 
* support vector machines(SVMs): effecrive in high dimentional soaces. Used for both regression and classification.  
* Decision Trees
* Random Forests
* Naive Bayes
* K-Nearest Neighbors (KNN)
* Neural Networks :Highly capable model capable of handling various types of data and tasks including image and speech recognition by learning complex patterns from data. Library used : Tensor Flow 
* Gradient Boosting Machines(GBM): 


### Ways to evaluate supervised learning models: 

#### For Regression: 
* Mean Squared Error (MSE):
* Root Mean Squared Error (RMSE):
* Mean Absolute Error (MAE):
* R-Squared (Coefficient of Determination):

#### For Classification: 
* Accuracy:
* Precison :
* Recall:
* F1 Score:
* Confusion Matrix:  
  

## UnSupervised Learning: 
Here learning happens from UnLabeled or Uncategorized data. Goal is to discover the pattern and categories in the Unlabeled data without explicit guidance. No training is given to the model. So, machines are supposed to find the hidden pattern ,actions or structures to categirize data.  

Types of unsupervised learning: 

#### Clustering: 
Grouping similar data points together. It is a way to move silimar data points in nearer to the same clusters and away from the other non-similar data clusters. techniques and methods are used to group data points into clusters based on their similarities: 

* Exclusive (partitioning)
* Agglomerative
* Overlapping
* Probabilistic

Types of clustering: 
* Hierarchial Clustering
* K-means clustering
* Principal Component analysis(PCA)
* Singular Value Decomposition
* Independent Component Analysis(ICA)
* Autoencoders
* Gaussian Mixture Models (GMMs)
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

#### Association : 
Here, relationship between different data in clusters are found. Such as ,people who buy x item may also buy y item. Common types of association Unsupervised learning include: 
* Apriori algorithm
* Eclat Algorithm
* FP-Growth Algorithm



### Testing UnSUpervised data models(BlackBox functional testing):
#### Cross validation testing method: 
1st step is Training on Training set. Then Training on Validation set. Here, can train the dev build with many inout data and see if build is able to segrigate the input data into different clusters and test it by asking questions like: where is my sales highest in which city ,product, questions on type of customers to target marketing. 

Here, We have only un-labeled data. In **training data set**, model learns the needed patterns and structure in data. In **validation data set** ,input data(Training data set) is seprated into clusters of matching data .

### Silhouette Scores(For Whitebox testing): 
This score is from -1 to 1. This checks the new untrained data score of model. If data is matching to the similar data cluster and not matching to the other nearby clusrters, it is higher score. If score is matching to the nearby cluster, it is low score and shows data far away from matching cluster.
Can ask developers ,the methods and utilities where to pass data with some python methods to get the Silhouette score for different test data. 

If the silhoutte scores are high with training data and it suddenly score dips after more training it reached training Overfitting. So, should stop there. 

### Calinski-Harabasz score: 
The Calinski-Harabasz score measures the ratio between the variance between clusters and the variance within clusters. Ranges from 0 to infinity. Higher score is better clustering.

### Adjusted Rand index: 
measures the similarity between two clusterings. It ranges from -1 to 1. higher scores indicating more similar clusterings.

### F1 score: 
This is weighted avg. of precision and recall ,which are 2 matrics used in supervised learning to evaluate classification model. F1 score can also be used for Unsupervised learning.  

### Applications of Supervised Learning: 
* Span filtering, Image classification, Medical DIagnosys, Fraud Detection,
### Applications of unsupervised leaerning: 
* anomaly detection: Can identify unusual(total new) patterns or deviations from normal behaviour in data, enabling fraud detection, system failure etc.
* Scientific discovery: Can find hidden relation,pattern in data.
* Recommendation systems : on products, moview, songs etc
* Customer segmentation: Can cluster/group similar customers together.
* Image, audio, video segmentation

### Disadvantages of unSupervised Learning: 
* Difficult to measure accuracy or effectiveness due to lack of labeled data.
* Lesser accuracy.
* Noisy data can be difficult to cluster.
* Number of classes not known

### Disavantages of supervised learning: 
* Cannot process very large and more complex data from supervised learning.


## Hybrid Learning: 
Combination of Supervised and Unsupervised learning. eg: ChatGPT. 


## Reinforcement Learning: 
Here model learns from it's surrounding ,actions taken by other users, here model's decision depends on the current state not the history. This needs trial and error for learning. 

Types of Reinforcement learning:
* 


Famous Frameworks: 
* Scikit Learn: Mainly for Linear regresion and Logistic regression.
* Tensor Flow: Mainly for Neural Networks, Reinforcement Learning.
* Keras: High Level Neural Network API ,runs on top of TensorFlow for better UI.  
* PyTorch:For DL models. Used mainly in research.
* XGBoost:For Gradient Boosting algorithm. 
* Light BGM: A Gradient Boosting Framework for large scale data. 
* CatBoost: A Gradient Boosting Framework, specialized in categorial data. 
* OpenAI Gym: For Developing and comparing Reinfrcement Learning Algorithms. 
* Stable Baselines:



## QA Questions  
1. Which type of learning team is using from supervised, Unsupervised, Hybrid(Supervised+Unsupervised), Reinforcement ?
2. Which Algorithm Team is using for the type of Learning? As there are multiple for every type of learning mentioned above.
3. Which Framework(Library) team is using to implement algorithms?
4. Can  you give me utilities/methods where i can generate overfitting and Underfitting graphs for Linear regression.
5. Can you give me utilities/methods where i can generate scores for unsupervised learning ? 
