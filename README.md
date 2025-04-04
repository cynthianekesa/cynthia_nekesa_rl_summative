# Botanical Explorer: A journey into the unknown to identify an indigenous rare medicinal plant
## 1. **Introduction: RL Embedded in Pre-Capstone Proposal**
There has been extreme biodiversity loss taking place in `KALONU AREA` over the past 5 years.

Cynthia, a botanist and student at ALU has been doing research for the past 4 months in her Pre-capstone module and realized how the loss of biodiversity has led to extreme loss of indigenous medicinal plants(native to Africa).

On the other hand, 1 day ago, Cynthia saw a breaking news story on her television about the rise of chronic diseases like cancer, HIV, and diabetes in KALONU AREA. She quickly remembers the indigenous knowledge she got from her late grandmother on a rare plant called `cure` that could be the answer to all these human disasters. She is the only hope for humanity at this point in time, as she is the only one who knows where `cure` is and its location. Let's see how RL can be applied to illustrate how Cynthia identifies this rare plant and saves humanity.


This project focuses on implementing and comparing two reinforcement learning (RL) techniques; (DQN) and Proximal Policy Optimization (PPO), in a custom gym environment. The goal is to train an agent to navigate a grid-based world`(wild tropical indigenous forest called Kalo)`, avoiding obstacles`(wild animals and poisonous plants)` and reaching a target`(the ‘cure’ indigenous medicinal plant that is going to save the world from chronic diseases like cancer, HIV and diabetes)` efficiently. The DQN approach leverages Q-learning with deep neural networks, while PPO optimizes the policy directly using gradient ascent. 

---

## 2. **Deployment**
The project was deployed using Google Colab, but Python scripts were also provided in the `simpleenv_withgym` folder to run locally, given enough CPU.

The `complexenv_withopengl` folder is not for use in this assignment as it was complex, needed alot of resources, time and attention. I however documented it for use in future improvements of this work.


### Installation
To set up the environment and run the project, follow these steps:

1. Clone the repository

```bash
git clone https://github.com/cynthianekesa/cynthia_nekesa_rl_summative.git
cd cynthia_nekesa_rl_summative
cd simpleenv_withgym
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the training script to train the DQN agent and PG agent:

```bash
cd training
python dqn_training
```

```bash
cd training
python pg_training
```

4. Simulate the trained agents :

```bash
Play around with main.py to run experiments and rendering.py to visualize the agents in action
```

---

## 3. **Custom Gym Environment**
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

## 4. **DQN Training**

### Training Performance

![dqn-training-performance](https://github.com/user-attachments/assets/395925d9-6f6e-4398-9767-af3e8e3f2af9)


![dqn-training-performance(1)](https://github.com/user-attachments/assets/765e2d6f-1bb7-4af4-8d18-bb86b6e86794)



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


---

## 5. **PPO Training**

### Training Performance

![dqn-training-performance](https://github.com/user-attachments/assets/395925d9-6f6e-4398-9767-af3e8e3f2af9)


![dqn-training-performance(1)](https://github.com/user-attachments/assets/765e2d6f-1bb7-4af4-8d18-bb86b6e86794)



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


---

## 6. **Documentation**

- **Demo video**: Folder contains simulation videos and presentation video. [https://drive.google.com/drive/folders/1awBBAhWED3L2p0Itbj3rvtyLyCoPiLii?usp=drive_link]
- **Report**: [https://docs.google.com/document/d/1Xm_uA0weyQzq_YIOEvccIexpPZ1UVzlct73tX35SkMU/edit?usp=sharing]


---

## 7. **Conclusion**
PPO outperformed DQN in terms of convergence speed and stability, attributed to its policy clipping and actor-critic architecture. However, DQN was simpler to implement and required less hyperparameter tuning. For future work, integrating prioritized experience replay into DQN or testing other policy gradient methods like SAC could further improve performance. Also, given additional time and resources, I would use an observational space that is intuitive, like the presence of different terrains and different medicinal plants, where the botanist(agent) identifies plants and their medicinal information is displayed, hence getting rewards for those actions. I actually began with this particular solution, but lack of time and resources made me backtrack to the solution presented.

**Future Work:**
- Building a more intuitive observation space(drone to locate the medicinal plant or the solution described above and tried out in complexenv_withopengl but failed).
- Experiment with other policy gradient methods and network architectures for DQN.
- Implement the environment such that it handles more edge cases and termination conditions.
  

---

## 8. **Contribution**
Make a pull request before contributing.


---

## 9. **License**
No licenses were installed for this project.


---

