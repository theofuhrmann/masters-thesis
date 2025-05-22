# Code for my master's thesis

## Installation

To set up the environment, follow these steps:

1.  Create a conda environment with Python 3.10:

    ```bash
    conda create --name thesis python=3.10
    conda activate thesis
    ```
2.  Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
    
3.  Copy the `.env.example` file to `.env`:

    ```bash
    cp .env.example .env
    ```

4.  Fill in the values in the `.env` file with your actual configuration.