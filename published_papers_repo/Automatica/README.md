# Direct Data-Driven Discounted Infinite Horizon LQR with Robustness Guarantees

![Automatica](https://img.shields.io/badge/Published%20in-Automatica-blue)
![Python](https://img.shields.io/badge/Made%20With-Python-blue)

This repository contains the official implementation of our paper:

> **Direct Data-Driven Discounted Infinite Horizon Linear Quadratic Regulator with Robustness Guarantees**  
> *Ramin Esmzad, Hamidreza Modares*  
> Published in Automatica, Elsevier, 2025  
> DOI: [10.1016/j.automatica.2025.112197](https://doi.org/10.1016/j.automatica.2025.112197)

## 📖 Overview
This repository provides a Python implementation of a **one-shot data-driven LQR design** with robustness guarantees for stochastic linear systems. Unlike traditional model-based and iterative methods, our approach ensures **robust stability and suboptimality gap analysis** while avoiding data-hungriness. 

The key contributions of this work include:
- A **direct, non-iterative** learning approach for **discounted infinite-horizon LQR**.
- The **prior robustness-guaranteed** one-shot LQR learning method.
- A novel **data-driven Lyapunov inequality formulation** for stochastic closed-loop systems.
- A practical case study on **active car suspension control**.

## 📦 Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
```

## 🚀 Usage
Run the main script to execute the LQR learning algorithm:

```bash
python automatica.py
```

## 📊 Results
Our method is validated on an **active car suspension simulation problem**, demonstrating superior robustness compared to existing methods. 

<p align="center">
  <img src="results/suspension_performance.png" width="500" />
</p>

## 📜 Citation
If you find this repository useful, please cite our paper:

```bibtex
@article{ESMZAD2025112197,
title = {Direct data-driven discounted infinite horizon linear quadratic regulator with robustness guarantees},
journal = {Automatica},
volume = {175},
pages = {112197},
year = {2025},
issn = {0005-1098},
doi = {https://doi.org/10.1016/j.automatica.2025.112197},
url = {https://www.sciencedirect.com/science/article/pii/S0005109825000883},
author = {Ramin Esmzad and Hamidreza Modares},
keywords = {Direct data-driven control, Linear quadratic regulator, Robustness, Suboptimality gap},
abstract = {This paper presents a one-shot learning approach with performance and robustness guarantees for the linear quadratic regulator (LQR) control of stochastic linear systems. Even though data-based LQR control has been widely considered, existing results suffer either from data hungriness due to the inherently iterative nature of the optimization formulation (e.g., value learning or policy gradient reinforcement learning algorithms) or from a lack of robustness guarantees in one-shot non-iterative algorithms. To avoid data hungriness while ensuing robustness guarantees, an adaptive dynamic programming formalization of the LQR is presented that relies on solving a Bellman inequality. The control gain and the value function are directly learned by using a control-oriented approach that characterizes the closed-loop system using data and a decision variable from which the control is obtained. This closed-loop characterization is noise-dependent. The effect of the closed-loop system noise on the Bellman inequality is considered to ensure both robust stability and suboptimal performance despite ignoring the measurement noise. To ensure robust stability, it is shown that this system characterization leads to a closed-loop system with multiplicative and additive noise, enabling the application of distributional robust control techniques. The analysis of the suboptimality gap reveals that robustness can be achieved by construction without the need for regularization or parameter tuning. The simulation results on the active car suspension problem demonstrate the superiority of the proposed method in terms of robustness and performance gap compared to existing methods.}
}
```

## 🛠 Contributions
Feel free to contribute by opening issues or submitting pull requests!

## 📩 Contact
For any inquiries or collaborations, reach out to:
- **Email**: [esmzadra@msu.edu](mailto:esmzadra@msu.edu)
- **LinkedIn**: [linkedin.com/in/raminesmzad](https://www.linkedin.com/in/raminesmzad/)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
🔬 **Bridging Control Theory & Data-Driven Methods!** 🚀

