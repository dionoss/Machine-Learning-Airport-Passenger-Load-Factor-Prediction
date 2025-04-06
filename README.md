# Machine Learning Airport Passenger Load Factor Prediction

This repository contains code and resources for the research project:  
**"Predicting Future Passenger Load Factor (PLF) of Commercial Flights Using Machine Learning."**

## Overview

Passenger Load Factor (PLF) represents how efficiently airlines fill their seats, directly affecting revenue and operational decisions. This project employs Machine Learning (ML) techniques, primarily CatBoost and K-Nearest Neighbors (KNN), to predict future PLF values based on historical flight data.

## Repository Structure

```
├── Catboost_trainer.py        # Train CatBoost ML model
├── Catboost_runner.py         # Run predictions using trained CatBoost model
├── KNN_Trainer.py             # Train KNN ML model
├── KNN_Runner.py              # Run predictions using trained KNN model
├── dc.py                      # Data cleaning and preprocessing utilities for my specific case
├── requirements.txt           # Python dependencies
├── LICENSE                    # GPL-3.0 License file
├── .gitignore                 # Git ignore file
```

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/dionoss/Machine-Learning-Airport-Passenger-Load-Factor-Prediction.git
cd Machine-Learning-Airport-Passenger-Load-Factor-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
- Place your dataset in the root directory.
- Adjust you data cleaning requirements in `dc.py` to match your dataset.

### 4. Training the Models
- **CatBoost**
  ```bash
  python Catboost_trainer.py
  ```

- **K-Nearest Neighbors**
  ```bash
  python KNN_Trainer.py
  ```

### 5. Run Predictions
- **CatBoost Predictions**
  ```bash
  python Catboost_runner.py
  ```

- **KNN Predictions**
  ```bash
  python KNN_Runner.py
  ```

## Dependencies
See `requirements.txt` for a detailed list of Python packages required.

## License
Distributed under the GNU General Public License v3.0. See `LICENSE` file for details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss.

## Citation
If you use this repository in academic research or commercial projects, please cite appropriately:

```bibtex
@article{
  author = {Dion Avé},
  title = {Predicting Future Passenger Load Factor of Commercial Flights Using Machine Learning},
  year = {2022},
}
```

