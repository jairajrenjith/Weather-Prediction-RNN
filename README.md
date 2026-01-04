# Weather-Prediction-RNN

Weather-Prediction-RNN is a time-series forecasting project that uses a Recurrent Neural Network (RNN) built with PyTorch to predict the next day’s mean temperature based on historical weather data.

This project demonstrates the complete machine learning workflow for sequential data, including data preprocessing, model training, evaluation, and visualization.

---

## Project Objective

To build and train a Recurrent Neural Network (RNN) that learns patterns from historical weather data and predicts future weather conditions, specifically the next day mean temperature.

---

## 📊 Dataset

- **File:** `london_weather.csv`
- **Type:** Historical time-series weather data
- **Features Used:**
  - Mean temperature
  - Minimum temperature
  - Maximum temperature
  - Precipitation
  - Cloud cover
  - Sunshine

Missing values are handled and the dataset is sorted chronologically to preserve time-series order.

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Model Architecture

- Model Type: Recurrent Neural Network (RNN)
- Input: Weather data from the previous 7 days
- Hidden Layer Size: 64 units
- Output: Next day mean temperature
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam Optimizer

The model processes sequential data to capture temporal dependencies in weather patterns.

---

## Workflow

1. Load and preprocess historical weather data  
2. Normalize numerical features using Min-Max scaling  
3. Convert data into time-series sequences  
4. Split data into training and testing sets  
5. Train the RNN model  
6. Evaluate performance using MSE and MAE  
7. Visualize predictions against actual values  
8. Perform next-day temperature prediction  

---

## Evaluation Metrics

The model is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Predicted values are inverse-scaled to display results in actual temperature units (°C).

---

## Visualizations

- Training loss curve
- Actual vs Predicted mean temperature plot (°C)

---

## Future Prediction

The trained model predicts the next day’s mean temperature using the most recent 7 days of historical weather data.

---

## Project Structure

```
Weather-Prediction-RNN/
│
├── weather_prediction_rnn.ipynb
├── london_weather.csv
└── README.md
```

---

## How to Run the Project

### 1. Clone Repository

  ```bash
  git clone <Repo Link>
  cd Weather-Prediction-RNN
  ```

### 2. Create a Virtual Environment (Recommended)

  ```bash
  python -m venv venv
  ```

Activate the virtual environment:

**a. On Windows**
  ```bash
  venv\Scripts\activate
  ```

**b. On Linux / macOS**
  ```bash
  source venv/bin/activate
  ```


### 3. Install Required Dependencies

  ```bash
  pip install torch pandas numpy matplotlib scikit-learn jupyter
  ```


### Run Notebook

  ```bash
  jupyter notebook weather_prediction_rnn.ipynb
  ```

---

## Results

- The RNN successfully learns temporal weather patterns  
- Predictions closely follow actual temperature trends  
- Evaluation metrics confirm reasonable forecasting performance  

---

## Future Improvements

- Replace RNN with LSTM or GRU
- Predict multiple weather parameters
- Multi-day forecasting
- Deploy as a web app or API

---

## License

This project is open-source and intended for educational purposes.

---

By Jairaj R.
