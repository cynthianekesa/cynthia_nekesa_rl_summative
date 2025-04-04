# Botanical Explorer: A journey into the unknown to identify an indigenous rare medicinal plant
## 1. **Introduction: RL Embedded in Pre-Capstone Proposal**
There has been extreme biodiversity loss taking place in `KALONU AREA` over the past 5 years.

Cynthia, a botanist and student at ALU has been doing research for the past 4 months in her Pre-capstone module and realized how the loss of biodiversity has led to extreme loss of indigenous medicinal plants(native to Africa).

On the other hand, 1 day ago, Cynthia saw a breaking news story on her television about the rise of chronic diseases like cancer, HIV, and diabetes in KALONU AREA. She quickly remembers the indigenous knowledge she got from her late grandmother on a rare plant called `cure` that could be the answer to all these human disasters. She is the only hope for humanity at this point in time, as she is the only one who knows where `cure` is and its location. Let's see how RL can be applied to illustrate how Cynthia identifies this rare plant and saves humanity.


This project focuses on implementing and comparing two reinforcement learning (RL) techniques; (DQN) and Proximal Policy Optimization (PPO), in a custom gym environment. The goal is to train an agent to navigate a grid-based world`(wild tropical indigenous forest called Kalo)`, avoiding obstacles`(wild animals and poisonous plants)` and reaching a target`(the ‘cure’ indigenous medicinal plant that is going to save the world from chronic diseases like cancer, HIV and diabetes)` efficiently. The DQN approach leverages Q-learning with deep neural networks, while PPO optimizes the policy directly using gradient ascent. 

---

## 2. **Custom Gym Environment**
**Scratch env**

The visualization shows the agent (red), obstacles (grey), and the target (green).


![kalo environment](https://github.com/user-attachments/assets/3ab8961c-a977-41a3-ba58-38ff08f8747e)



**Static File**

INITIAL STATE


![kalo movement](https://github.com/user-attachments/assets/2b3b4492-7239-4f89-b677-a1728ca6e4f5)



FINAL STATE


![final state of cynthia in kalo](https://github.com/user-attachments/assets/0eb0eed8-f926-4cba-878c-0010b2ada290)




**GIFS**

Visualization of the agent and states without training involved:



![kalo_animation](https://github.com/user-attachments/assets/29fdbc01-c7bb-4ac3-9f43-895bbff3b19e)



![kalo_animation (1)](https://github.com/user-attachments/assets/c58792e8-5b7a-4072-8fd5-038d7a61ef49)


---

## 3. **DQN Training**

### Training Performance

![dqn-training-performance](https://github.com/user-attachments/assets/395925d9-6f6e-4398-9767-af3e8e3f2af9)


**Graph 1: Cumulative Reward During Training.**
- **X-axis**: Episodes (ranging from 0 to 10,000).
- **Y-axis**: Cumulative Reward (ranging from 0 to 40,000).
- **Data Representation**: A blue line represents the cumulative reward earned by the agent over the episodes.
- **Observations**:
  - The cumulative reward increases steadily over time, indicating that the agent is learning and improving its performance.
  - There are noticeable plateaus (e.g., around 2,000 episodes), suggesting periods where the agent's learning progress temporarily stagnates before resuming improvement.
  - The overall trend is upward, showing consistent learning.

**Graph 2: Loss Curve During Training**
- **X-axis**: Steps (ranging from 0 to 10,000).
- **Y-axis**: Loss (ranging from 0 to 2.5).
- **Data Representation**: A red line represents the loss values during the training process.
- **Observations**:
  - The loss fluctuates significantly throughout the training process, with sharp peaks and valleys.
  - There are occasional spikes in the loss (e.g., near the end of the training), which could indicate challenges in optimizing the agent's policy or instability in the learning process.
  - Despite the fluctuations, the loss appears to stabilize somewhat over time, though it does not completely smooth out.

### Agent Performance

**Metrics**
- Success Rate → % of times the agent(Cynthia) reached the "Cure"
- Average Reward → Measures agent efficiency
- Average Steps Taken → Lower steps = better navigation

![dqn agent performace](https://github.com/user-attachments/assets/a683021b-c38e-4106-8a3c-012e6fa733ef)


**Graphs**

![dqn agent graphs](https://github.com/user-attachments/assets/a34100ea-8159-4fae-8db7-11023bb23e4a)


**Graph 1: Total Reward per Episode**
- **Y-axis**: Total Reward (scale ranges from approximately 8.8 to 9.8)
- **X-axis**: Episode (ranges from 0 to about 17.5)
- **Data**: Blue dots connected by lines showing very stable reward values
- **Pattern**: The reward consistently stays around 9.3 throughout all episodes, with minimal variation

**Graph 2: Steps Taken per Episode**

- **Y-axis**: Steps Taken (scale ranges from approximately 7.6 to 8.4)
- **X-axis**: Episode (ranges from 0 to about 17.5, matching the top graph)
- **Data**: Red dots connected by lines showing very stable step counts
- **Pattern**: The steps taken per episode remains extremely consistent at around 8.0 across all episodes


- **Observations**:
  - Based on the DQN agent graphs, the agent has reached a remarkably stable performance level. Both the reward (consistently around 9.3) and steps taken (around 8.0) per episode show almost no variation across all training episodes.
  - Since there's no visible learning curve or improvement over time, the agent appears to have found its strategy immediately and maintained it throughout training.



**Hyperparameter Insights:**
- Hyperparameters experimented with:

```bash
 {"learning_rate": 2e-5, "batch_size": 16, "weight_decay": 0.01},
    {"learning_rate": 5e-6, "batch_size": 8, "weight_decay": 0.02},
    {"learning_rate": 1e-6, "batch_size": 32, "weight_decay": 0.005},
    {"learning_rate": 3e-5, "batch_size": 16, "weight_decay": 0.1},
    {"learning_rate": 1e-5, "batch_size": 16, "weight_decay": 0.001},
    {"learning_rate": 1e-6, "batch_size": 8, "weight_decay": 0.01},
    {"learning_rate": 1e-5, "batch_size": 16, "weight_decay": 0.02},
]
```

- Best accuracy was the last hyperparameter pair with an accuracy of `7.2154860496521`from `6.804567`
- More detailed visual evaluations of other parameters can be seen in the notebook.





- A lower learning rate (e.g., 2e-5) resulted in more stable training and better convergence.
- A batch size of 32 provided a good balance between training speed and memory usage.
- A higher weight decay (e.g 0.1) helps regularize the model and prevent overfitting, while a lower decay(e.g 0.001) is useful if the model is underfitting.
- Training for 4 epochs yielded the best performance without overfitting.
  

---

## 4. **PPO Training**

**Quantitative Metrics:**
- **BLEU Score**: Used to evaluate the quality of generated responses by comparing them to reference answers.
- **F1-Score**: Measured the chatbot's ability to correctly classify and respond to user intents.
- **Perplexity**: Assessed the model's confidence in generating responses.
  
![image](https://github.com/user-attachments/assets/d3ad7224-8ef2-4177-a387-d7ceded3db74)

  
**Metric	Score	Interpretation:**
- *BLEU_ 0.433_ Generated responses have a 43.3% overlap with the reference answers in terms of n-grams (e.g., 1-gram, 2-gram). This shows a decent overlap with reference, but responses could be more precise.*
- *ROUGE_ -1	0.643_ 64.3% of the words in the reference text are also present in the model’s output.	Strong overlap of key terms, indicating good relevance.*
- *ROUGE_ -L	0.571_	 The model is capturing 57.1% of the key phrases or sequences from the reference. This shows good alignment of phrases, but fluency and coherence could be improved.*
- *Perplexity_	10.076_	Low perplexity, indicating the model is confident in its predictions.*


---

## 5. **Deployment**
The chatbot was deployed using **Gradio**, a simple and intuitive web interface framework. 

### Installation
To install the required packages, run:

```bash
pip install -r requirements.txt
```

### Running

```bash
python app.py
```

The gradio interface will launch hence ready for interaction


---

## 7. **Demo Video**
The demo video includes:
- **Introduction**: Brief overview of the project and its goals.
- **User Interaction**: Demonstration of the chatbot answering climate change-related questions.
- **Key Insights**: Discussion of the chatbot's performance, challenges faced, and future improvements.

[https://drive.google.com/file/d/1c43Q6M_mUV70Mepm43XGcwHUFuVLEdn1/view?usp=drive_link]


---

## 8. **Conclusion**
The Climate Change Chatbot successfully leverages a pre-trained Transformer model to provide accurate and relevant responses to user queries. Fine-tuning the model on a domain-specific dataset and deploying it using Gradio ensures that the chatbot is both functional and accessible. The project demonstrates the potential of Transformer models in building domain-specific conversational agents and highlights the importance of careful dataset preparation and hyperparameter tuning.

**Future Work:**
- Expand the dataset to include more diverse conversational pairs.
- Experiment with other Transformer models, such as T5 or ALBERT.
- Improve the chatbot's ability to handle out-of-domain queries more gracefully.
  

---

## 9. **Contribution**
Make a pull request before contributing.


---

## 10. **License**
No licenses were installed for this project.


---

