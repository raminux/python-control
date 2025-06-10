# Direct Data Driven Control Using Noisy Measurements 📊

**Ramin Esmzad**, **Gokul S. Sankar**, **Teawon Han**, **Hamidreza Modares**  
*Submitted May 9, 2025 • arXiv:2505.06407 [eess.SY, cs.LG, cs.RO, math.OC]*
https://arxiv.org/pdf/2505.06407




## 🚀 Overview
This repository presents a **direct, data-driven control framework** for solving Linear Quadratic Regulator (LQR) problems using solely noisy input-output measurements, without requiring the identification of the underlying dynamics. Our method guarantees **mean-square stability (MSS)** and **optimal performance**, employing convex optimization and robust controller synthesis via linear matrix inequalities (LMIs) and semidefinite programming (SDP). The approach is validated through simulations on benchmarks like the rotary inverted pendulum and active suspension system, showing improvements over existing methods. 

## 📚 Abstract
> We present a novel direct data-driven control framework for solving the linear quadratic regulator (LQR) under disturbances and noisy state measurements. Our approach guarantees mean-square stability (MSS) and optimal performance by leveraging convex optimization techniques that incorporate noise statistics directly into the controller synthesis. … Extensive simulations on benchmark systems … demonstrate the superior robustness and accuracy of our method compared to existing data-driven LQR approaches. 

## Cite this work 📄

You can cite this work as:

```bibtex
@misc{esmzad2025dddnm,
  title        = {{Direct Data Driven Control Using Noisy Measurements}},
  author       = {Ramin Esmzad and Gokul S. Sankar and Teawon Han and Hamidreza Modares},
  year         = {2025},
  eprint       = {2505.06407},
  archivePrefix= {arXiv},
  primaryClass = {eess.SY},
  url          = {https://arxiv.org/abs/2505.06407},
}


## 📞 Contact

For questions, feedback, or collaborations, feel free to reach out:

- **Ramin Esmzad**  
  ✉️ Email: [esmzadra@msu.edu](mailto:esmzadra@msu.edu)  
  🔗 LinkedIn: [linkedin.com/in/ramin-esmzad](https://linkedin.com/in/ramin-esmzad)  



## 🛠️ Installation
```bash
git clone <this-repo-url>
cd dddnm  # or your repo directory name
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
python runme.py

