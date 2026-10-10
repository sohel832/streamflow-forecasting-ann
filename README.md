# Application of Artificial Neural Networks in Streamflow Forecasting

**Course:** UE24CS352A – Machine Learning Mini-Project
**Team:** Shreyas Bellad  	PES2UG25CS824 
Sohel Mahibubasa Chapparband  	PES2UG25CS826    

**Section:** F
**Problem No.:** 11

## 1. Problem Statement
Predict the daily streamflow of the Leaf River near Collins, Mississippi, using only past rainfall and past streamflow. This is a regression problem. Two models are built and compared:

1. Feed-Forward Neural Network (FFNN)
2. Long Short-Term Memory network (LSTM)

The work is based on the reference study *"Application of Artificial Neural Network in Streamflow Forecasting"* (Xiao and You, Stanford).

## 2. Dataset
| Item | Details |
|---|---|
| Streamflow | USGS daily mean discharge (ft³/s), site `02472000`, parameter `00060` |
| Rainfall | NOAA GHCN-Daily `PRCP`, station `USC00221852` (Collins, MS), converted from tenths of mm to mm |
| Period | 1 Jan 2003 – 31 Dec 2012 |
| Records after cleaning | 3,607 days (45 days with missing rainfall removed) |
| Train / Validation | 2003–2007 / 2008–2012 (chronological, no shuffling) |

The notebook **downloads the data automatically** from the USGS and NOAA websites, so an internet connection is required on the first run. No dataset files need to be uploaded to this repository.

## 3. Approach
1. Download and load the raw data, then keep the study period only.
2. Convert rainfall units, merge both series by date and remove invalid or missing values.
3. Create lag features (FFNN: 3 rainfall lags + 4 streamflow lags; LSTM: sequences of 3 days × 2 features).
4. Split chronologically and apply Min-Max scaling (scalers fitted on training data only).
5. Train both models with RMSE as the loss, then predict and convert back to the original scale.
6. Evaluate with RMSE overall, on peak flow (Q > 1500) and on non-peak flow (Q ≤ 1500).

## 4. Model Configuration
| | FFNN | LSTM |
|---|---|---|
| Architecture | Dense(12, sigmoid) → Dense(1, linear) | LSTM(32, tanh) → Dense(1, linear) |
| Parameters | 109 | 4,513 |
| Optimizer | Adam, lr = 0.001 | Adam, lr = 0.002 |
| Epochs / Batch | 200 / 32 | 200 / 32 |
| Loss / Metric | RMSE | RMSE |

## 5. Results (Validation set, 2008–2012)
| Model | Validation RMSE | Peak RMSE | Non-peak RMSE | Training RMSE |
|---|---|---|---|---|
| FFNN | 733.52 | 2062.24 | 157.66 | 773.49 |
| LSTM | 698.10 | 1961.20 | 152.67 | 635.33 |

All RMSE values are in ft³/s. Both models are far more accurate on non-peak days than on peak days. In this implementation the LSTM is slightly better than the FFNN. Results may vary slightly between runs.

## 6. Repository Structure
```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── streamflow_forecasting.ipynb       # Complete code: data, models, evaluation
└── docs/
    └── ML_Mini_Project_Report.pdf     # Project write-up
```
The folders `raw_data/`, `processed_data/` and `results/` are created automatically when the notebook runs.

## 7. Setup and How to Run
**Requirements:** Python 3.9 – 3.12 and an internet connection.

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter and open the notebook
jupyter notebook streamflow_forecasting.ipynb
```
Then choose **Kernel → Restart & Run All**. Training takes a few minutes on a CPU.

## 8. Tools and Libraries
Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, matplotlib, seaborn, requests.

## 9. Team Contribution
| Member | Contribution |
|---|---|
| [Shreyas] | [e.g. data collection and preprocessing, FFNN] |
| [Sohel] | [e.g. LSTM, evaluation, report] |

## 10. References
1. Xiao, M. and You, S. *Application of Artificial Neural Network in Streamflow Forecasting.* Stanford University.
2. Hsu, K., Gupta, H. V. and Sorooshian, S. (1995). Artificial neural network modeling of the rainfall-runoff process. *Water Resources Research*, 31(10).
3. Kratzert, F. et al. (2018). Rainfall–runoff modelling using long short-term memory (LSTM) networks. *Hydrol. Earth Syst. Sci.*, 22(11).
4. USGS Water Services (site 02472000); NOAA NCEI GHCN-Daily (station USC00221852).
