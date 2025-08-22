# Generative Adversarial Imitation Learning (GAIL) — PyTorch from Scratch

This repository contains a **from-scratch implementation of Generative Adversarial Imitation Learning (GAIL)** in PyTorch.  
GAIL extends the concept of **Generative Adversarial Networks (GANs)** to reinforcement learning, where an agent learns to imitate expert behavior without explicit access to the reward function.

It has been trained and tested on **MuJoCo control environments** using expert demonstrations, achieving stable policy imitation through adversarial training between a generator (policy) and discriminator.

---

## EVAL-Return
<img src="return-result.jpeg">


---

## Key Features
- Implemented fully from scratch in PyTorch
- **Expert data collection pipeline** for generating demonstrations
- **Discriminator network** trained to distinguish expert vs agent trajectories
- **Policy network** trained adversarially to fool the discriminator
- Compatible with **MuJoCo environments**
- Training, evaluation, and video-recording scripts included

---

## 📂 Repository Structure
