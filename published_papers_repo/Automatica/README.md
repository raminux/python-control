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
@article{esmzad2025lqr,
  author  = {Ramin Esmzad and Hamidreza Modares},
  title   = {Direct Data-Driven Discounted Infinite Horizon Linear Quadratic Regulator with Robustness Guarantees},
  journal = {Automatica},
  volume  = {175},
  pages   = {112197},
  year    = {2025},
  publisher = {Elsevier},
  doi     = {10.1016/j.automatica.2025.112197}
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

