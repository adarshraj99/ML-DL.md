## Keywords: 
#### Temprature: 
Temperature is a parameter in generative AI (GenAI) that controls the randomness of the output. Higher temprature means more creative output and lower temprature gives more predictive output. It is in range 0-2 and default set in google gemini is 1. 
#### Tokens: 
Smallest independant data blocks made by genAI model from the given input. Model makes small blocks to find which block suits with which block to make a correct output.
#### Embeddings : 
Encodes the tokens with it's meanings based on thet token data. As, input passes through the model layers ,tokens gets updated Embeddings.
#### Context: 
amount of text data a model can process at a given time.
- Tokens keeps updating itself based on the input data which helps in the output data needed.
#### Hallusination: 
AI bot gives incorrect answers because of limited data training or having multiple outputs, lenghty complex prompts, human phrase languase non-understanding. ex: prompt: "my schoolteacher head is on fire .What should i do ?"  ,  "she is as blind as a bat. What should i do ?"(here i meant she dosen't care about things around).
#### Banned Contents: 
Some contents are banned from GPT. like : info on political figure location ,info on global wars like :who will isreal kill next?   

### Can set Gemini GPT token and search here: 
1. https://console.cloud.google.com/vertex-ai/studio/freeform?project=carbon-feat-280309



## Neurons and how it learns: 

### neurons: 
These are mathamatical functions that work together to solve a problem. A neuron just hold a value between 0 and 1. This number inside neuron is called **activation**. Some groups sets firing cause some other groups to fire. 


![image](https://github.com/user-attachments/assets/566eadae-4a26-4cee-be07-88b8209186ef)

Here, splited parts of the input image is checked in second layer. These splited parts may look like a straight vertical or tilted line(|or/) and a straight horozontal line(-) to make '7'. 
But, how 2nd layer will know which is vertical line or horizontal line ? Basically how it will find the parts of the image ? It is calculated by **weights**


### Weights: 
These are numbers assigned as strength between neurons connection. Basecally it is number between -n and +n (different for different models). Higher weight will confirm better connection and vise versa. Neuron data with Higher weight will have more chance of passing to next neuron layer for further processing. 

![image](https://github.com/user-attachments/assets/da0dff38-e94c-46aa-8501-93fed23a8975)

It is called **weighted sum**. Here, the w1,w2,w3,w4,......wn are weights and a1,a2,a3,.......an are **activations**. When, we calculate all the weights in a matrix of pixels below. It is seen that the most positive weights for part(-) of input(7) is at the correct position.

Here the weighted sum is equal to the value of pixels of the needed part. 

![image](https://github.com/user-attachments/assets/1f6279e2-aff0-4cf2-a2ca-81b76048c425)

Here, the activation should be between 0 to 1. A common function called **sigmoid function or logistic curve** does this conversion from big scale (-n to +n) to 0 to 1 scale value update. More positive values gets near to 1.

![image](https://github.com/user-attachments/assets/0df9289a-29d1-4df3-8d5c-0a4625f4dbc6)


### Weight Quantization: 
**Size** of a model depends on model weight size and data type's precision. 
To save memory weights can be stored with lower precision data types by process known as quantization. There are 2 main major mathods : 
#### Post-Training Quantization (PTQ): 
weights of an already trained model are converted to lower precision without any retraining. It is easier todo and causes potential performance degradation.

#### Quantization-Aware Training (QAT): 
weight conversion happens in process during the pre-training or fine-tuning stage ,resulting in enhanced model performance. It is computationally expensive and demands **representative** (data that has the features or data points that the application is designed to predict or classify) **training data**. 

Mostly Floating point data are used due to precision. Typically floating point numbers uses n bits to store a numerical value. These n bits are partitioned into 3 distinct components:
`sign`: +ve or -ve number. It uses one bit where 0 indicates a positive number and 1 signals a negative number.

`exponent` : The exponent is a segment of bits that represents the power to which the base (usually 2 in binary representation) is raised. The exponent can also be positive or negative, allowing the number to represent very large or very small values.

`Significand/Mantissa` : The remaining bits are used to store the significand or mantissa. This represents the significant digits of the number. The precision of the number heavily depends on the length of the significand.

Formula used for this representation is : `$(-1)^{sign} * (base)^{exponent}$ * significand`



### Activation:
Activation of the neurons are bascically a measure of how positive the relevant weighted sum is. More activation makes the neuron more light up. 

![image](https://github.com/user-attachments/assets/e7aeb112-28aa-418f-a6ff-c93c70086d62)

In cases there  may be condition to not light up pixel when weighted sum < 10. This is adding a **bias** for inactivity. 

![image](https://github.com/user-attachments/assets/2467496f-fc28-4eaf-bce9-0050413bd3f0)
![image](https://github.com/user-attachments/assets/d86447c9-cabe-43ed-8185-68fc5707e186)

![image](https://github.com/user-attachments/assets/2ab5d8f3-7678-4684-9a5d-ea32db974ff8)

Here, 1st layer have 784 neurons, 2nd layer have 16 neurons and 3rd have 16 neurons and 4th have 10 neurons. Every neuron have it's own biases. Here, it is 13,002
weights and biases. 
The secons layer is expected to pick up on edges and 3rd layer picks up the patterns. 

![image](https://github.com/user-attachments/assets/75a29064-ccde-4303-acb4-a405a76cb8e0)

### Deep Learning : 
Finding the correct weights and biases by the incorrect outputs comparision with correct output.

Weights ,Activations, biases are passed in the sigmoid function to calculate the forward transition of activations from one to next layer.  

![image](https://github.com/user-attachments/assets/c63bdce7-3137-4016-9ffa-3d840521bef0)

Sigmoid function is old schoold now and new one is ReLU(a) = max(0,a) where a is activation. ReLU is Rectified Linear Unit. 
