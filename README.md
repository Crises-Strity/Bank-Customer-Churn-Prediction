# Project 1

This repository was generated from the Jupyter notebook `Project 1.ipynb`.

## What&apos;s inside
- `notebooks/Project 1.ipynb` – the original notebook
- `src/project_1/main.py` – Python script exported from the notebook&apos;s code cells
- `requirements.txt` – best-effort dependencies parsed from imports found in the notebook
- `.gitignore` – standard Python & Jupyter ignores

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the script (if applicable)
python -m project_1.main
```

> Note: `requirements.txt` is inferred from imports and may not be complete. 
> If something is missing, install it and then pin with:
>
> ```bash
> pip freeze > requirements.txt
> ```

## Project Structure
```
project-1/
├─ notebooks/
│  └─ Project 1.ipynb
├─ src/
│  └─ project_1/
│     └─ main.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## How to publish to GitHub
1. Create a new GitHub repository (empty).
2. From this folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit from notebook export"
   git branch -M main
   git remote add origin https://github.com/&lt;your-username&gt;/project-1.git
   git push -u origin main
   ```
