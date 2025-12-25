import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template
import pygad
import skfuzzy as fuzz
from skfuzzy import control as ctrl

dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)

x = dataset.iloc[:, : 10].values
y = dataset.iloc[:, 10].values

# Train/validation split for GA-based threshold tuning
train_x, val_x, train_y, val_y = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
scaler.fit(train_x)

val_x_scaled = scaler.transform(val_x)

model = tf.keras.models.load_model(os.path.join('filesuse', 'project_model1.h5'))

# Pre-compute validation probabilities for GA fitness
val_probs = model.predict(val_x_scaled).flatten()


def _evaluate_threshold(threshold: float) -> float:
    """Compute F1 score for a given decision threshold."""
    thr = float(np.clip(threshold, 0.0, 1.0))
    preds = (val_probs >= thr).astype(int)
    return f1_score(val_y, preds)


def _fitness_func(ga_instance, solution, _solution_idx):
    # ga_instance is unused; required by PyGAD signature
    return _evaluate_threshold(solution[0])


# Simple GA to optimize decision threshold
ga = pygad.GA(
    num_generations=20,
    sol_per_pop=20,
    num_parents_mating=5,
    num_genes=1,
    init_range_low=0.0,
    init_range_high=1.0,
    mutation_percent_genes=30,
    mutation_type="random",
    mutation_by_replacement=True,
    random_mutation_min_val=0.0,
    random_mutation_max_val=1.0,
    fitness_func=_fitness_func,
    allow_duplicate_genes=False,
)
ga.run()
best_solution, _, _ = ga.best_solution()
BEST_THRESHOLD = float(np.clip(best_solution[0], 0.0, 1.0))


# Fuzzy logic setup to map probability to risk label
probability = ctrl.Antecedent(np.linspace(0, 1, 101), "probability")
risk = ctrl.Consequent(np.linspace(0, 100, 101), "risk")

probability["low"] = fuzz.trimf(probability.universe, [0, 0, 0.4])
probability["medium"] = fuzz.trimf(probability.universe, [0.2, 0.5, 0.8])
probability["high"] = fuzz.trimf(probability.universe, [0.6, 1.0, 1.0])

risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
risk["medium"] = fuzz.trimf(risk.universe, [25, 50, 75])
risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

risk_rules = [
    ctrl.Rule(probability["low"], risk["low"]),
    ctrl.Rule(probability["medium"], risk["medium"]),
    ctrl.Rule(probability["high"], risk["high"]),
]

risk_control = ctrl.ControlSystem(risk_rules)


def fuzzy_risk(prob_value: float):
    sim = ctrl.ControlSystemSimulation(risk_control)
    sim.input["probability"] = float(np.clip(prob_value, 0.0, 1.0))
    sim.compute()
    score = float(sim.output["risk"])
    if score < 35:
        label = "LOW"
    elif score < 70:
        label = "MEDIUM"
    else:
        label = "HIGH"
    return score, label

app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')
def home():
    return render_template('home.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 


@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
        dob = pd.to_datetime(request.form.get("dob"))
        v5 = int(request.form.get("category"))
        v6 = float(request.form.get("card_number"))
        v8 = float(request.form.get("trans_amount"))
        v9 = int(request.form.get("state"))
        v10 = int(request.form.get("zip"))
        
        # Input validation
        errors = []
        if trans_datetime <= dob:
            errors.append("Transaction date must be after date of birth")
        if dob > pd.Timestamp.now():
            errors.append("Date of birth cannot be in the future")
        if v6 < 1000000000000000 or v6 > 9999999999999999:
            errors.append("Invalid card number (must be 16 digits)")
        if v8 <= 0:
            errors.append("Transaction amount must be positive")
        if v9 < 0 or v9 > 100:
            errors.append("Invalid state code")
        if v10 < 10000 or v10 > 999999:
            errors.append("Invalid ZIP code")
        
        if errors:
            return render_template(
                'result.html',
                OUTPUT="INVALID INPUT",
                PROBABILITY="N/A",
                THRESHOLD="N/A",
                RISK_LABEL="ERROR",
                RISK_SCORE="N/A",
                ERRORS=errors
            )
        
        v1 = trans_datetime.hour
        v2 = trans_datetime.day
        v3 = trans_datetime.month
        v4 = trans_datetime.year
        v7 = np.round((trans_datetime - dob).days / 365.25)
        
        x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
        proba = float(model.predict(scaler.transform([x_test]))[0][0])
        result = "VALID TRANSACTION" if proba <= BEST_THRESHOLD else "FRAUD TRANSACTION"
        risk_score, risk_label = fuzzy_risk(proba)
        return render_template(
            'result.html',
            OUTPUT=result,
            PROBABILITY=round(proba, 4),
            THRESHOLD=round(BEST_THRESHOLD, 4),
            RISK_LABEL=risk_label,
            RISK_SCORE=round(risk_score, 2),
        )
    except Exception as e:
        return render_template(
            'result.html',
            OUTPUT="ERROR",
            PROBABILITY="N/A",
            THRESHOLD="N/A",
            RISK_LABEL="ERROR",
            RISK_SCORE="N/A",
            ERRORS=[f"Invalid input format: {str(e)}"]
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)


