# Assignment 1: Adversarial Attacks

Implemented attacks:
- Problem 1: Targeted FGSM
- Problem 2: Untargeted FGSM
- Problem 3: PGD (Targeted / Untargeted)

Datasets:
- MNIST
- CIFAR-10

## Project Structure

```text
assignment1/
├── assignment1.pdf
├── models.py
├── train.py
├── test.py
├── requirements.txt
├── problem1/
│   ├── attacks.py
│   └── test.py
├── problem2/
│   ├── attacks.py
│   └── test.py
└── problem3/
    ├── attacks.py
    └── test.py
```


## Requirements

- Python 3.10 recommended
- `pip` or `conda`

## How To Run

```bash
git clone https://github.com/imjaegyun/Reliable-and-Trustworthy-AI.git
cd Reliable-and-Trustworthy-AI/assignment1
chmod +x setup_and_run.sh
./setup_and_run.sh
```
