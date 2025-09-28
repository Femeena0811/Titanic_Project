
# 🚢 Titanic Survival Predictor

An interactive Streamlit web app that predicts the survival chances of Titanic passengers using a trained machine learning model. Users can explore the dataset, visualize key patterns, and input custom passenger details to receive predictions.



## ✨ Features

* 🏠 Home Page – Overview of the dataset and app usage.
* 📊 Data Exploration – Browse, filter, and analyze the Titanic dataset interactively.
* 📈 Visualization – Pre-built charts showing survival trends across age, gender, class, and more.
* 🔮 Survival Prediction – Input passenger details to predict survival probability.
* ⚙️ Model Performance – View model performance metrics, feature importance, and a confusion matrix.



## 🛠 Requirements

Install the required Python libraries:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

Make sure the following files exist in the same folder as `app.py`:

* `model.pkl` – trained ML model file
* `label_encoder.pkl` – label encoders for categorical variables
* `Titanic-Dataset.csv` (optional) – dataset file. If missing, the app will generate sample data.



## 🚀 How to Run

1. Clone or download this repository.
2. Place `app.py`, `model.pkl`, and `label_encoder.pkl` in the same directory.
3. Install the dependencies (see above).
4. Run the app with:

```bash
streamlit run app.py
```

5. Your browser will open the app (default at `http://localhost:8501`).


## 📂 Project Structure

```
├── app.py               # Streamlit application script
├── model.pkl            # Pre-trained ML model
├── label_encoder.pkl    # Label encoders
├── Titanic-Dataset.csv  # Dataset (optional)
└── README.md
```



## 🧠 Model Overview

* Algorithm: Random Forest Classifier (default in uploaded model)
* Input Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone, Title
* Target: Survival (0 = Did Not Survive, 1 = Survived)


## 📊 Example Predictions

| Pclass | Sex    | Age | Fare | Embarked | Title | Prediction | Survival Prob. |
| ------ | ------ | --- | ---- | -------- | ----- | ---------- | -------------- |
| 1      | female | 29  | 75   | C        | Mrs   | Survived   | 87%            |
| 3      | male   | 35  | 10   | S        | Mr    | Died       | 12%            |



## 💡 Notes

* If the dataset file is not found, the app generates random sample data to demonstrate functionality.
* Model metrics shown on the **Model Performance** page are placeholders unless updated with real evaluation results.


## ❤️ Credits

Built with [Streamlit](https://streamlit.io/), [Pandas](https://pandas.pydata.org/), and [Scikit-learn](https://scikit-learn.org/).

