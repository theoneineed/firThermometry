# firThermometry

This repository contains the codebase for processing and analyzing fluorescence data acquired from a spectrophotometer for **far-infrared (FIR) optical thermometry**.

The project is intended to support the analysis of fluorescence measurements and the extraction of temperature-dependent information relevant to FIR optical thermometry.

## Reproducibility

The following instructions describe how to set up the Python environment and install the required dependencies.

### 1. Clone the repository

From your terminal, navigate to the directory where you would like to store the project and clone the repository:

```bash
git clone <repository-url>
cd firThermometry
```

### 2. Create a virtual environment

From the project root directory, create a Python virtual environment:

```bash
python -m venv .venv
```

If `python` is not available on your system, you may need to use:

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

Choose the command corresponding to your operating system and shell:

| Operating System | Terminal / Shell     | Activation Command           |
| ---------------- | -------------------- | ---------------------------- |
| macOS / Linux    | bash / zsh           | `source .venv/bin/activate`  |
| Windows          | Command Prompt (cmd) | `.venv\Scripts\activate.bat` |
| Windows          | PowerShell           | `.venv\Scripts\Activate.ps1` |

Once activated, your terminal should indicate that the `.venv` environment is active.

### 4. Install dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 5. Deactivate the environment

When you are finished working with the project, deactivate the virtual environment with:

```bash
deactivate
```

## Project Structure

The repository may contain the following types of files and directories:

* **Python scripts** — Data processing and analysis code.
* **Jupyter notebooks** — Interactive analysis, visualization, and exploration.
* **`requirements.txt`** — Python dependencies required to run the project.
* **Data / output directories** — Experimental data and generated results, where applicable.

> **Note:** Experimental data, generated figures, and other large or non-source files may be excluded from version control. Refer to the repository's `.gitignore` file for details.

## Citation

If you use this codebase or any of its associated analysis methods in your research, please cite the author's work appropriately.

For questions regarding the methodology, analysis, or use of this code, please refer to the associated publication or contact the author.
