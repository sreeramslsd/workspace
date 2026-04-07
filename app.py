from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

le_skill = LabelEncoder()
le_dept = LabelEncoder()

@app.route("/")
def home():
    return """
    <h1>🎯 Predictive Resource Capacity & Workforce Risk Platform v2.0</h1>
    <p><a href="/predict">API: /predict</a> | <a href="/predict?multiplier=1.3">Scenario: 30% more workload</a></p>
    <p>Open <a href="/static/index.html">Dashboard</a></p>
    """

@app.route("/static/<path:filename>")
def send_static(filename):
    return send_from_directory("static", filename)

@app.route("/predict", methods=["GET"])
def predict():
    mult_str = request.args.get("multiplier", "1.1")
    try:
        multiplier = float(mult_str)
    except:
        multiplier = 1.1

    try:
        data = pd.read_csv("employees_extended.csv")
    except FileNotFoundError:
        return jsonify({"error": "employees_extended.csv not found"}), 400

    required_cols = ["name", "skill", "department", "capacity_hours", "allocated_hours", "experience_years", "performance_score"]
    for col in required_cols:
        if col not in data.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    data["skill_encoded"] = le_skill.fit_transform(data["skill"])
    data["dept_encoded"] = le_dept.fit_transform(data["department"])

    feature_cols = ["capacity_hours", "allocated_hours", "experience_years", "performance_score", "skill_encoded", "dept_encoded"]
    X = data[feature_cols]
    y = data["allocated_hours"] * multiplier

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions_hours = model.predict(X)
    predictions_pct = (predictions_hours / data["capacity_hours"]) * 100
    data["prediction"] = predictions_pct.round(1)

    def detect_risk(row):
        pred = row["prediction"]
        perf = row["performance_score"]
        exp = row["experience_years"]
        if pred > 130 or (pred > 110 and perf < 3.5):
            return "High Risk"
        elif pred > 100:
            return "Overloaded"
        elif pred < 50:
            return "Underutilized"
        elif pred < 70 and exp > 5:
            return "Skill Mismatch"
        else:
            return "Normal"

    data["risk"] = data.apply(detect_risk, axis=1)

    response = data[["name", "skill", "department", "prediction", "risk", "performance_score"]].to_dict(orient="records")
    return jsonify(response)

@app.route("/skills", methods=["GET"])
def get_skill_gaps():
    try:
        data = pd.read_csv("employees_extended.csv")
    except FileNotFoundError:
        return jsonify({"error": "employees_extended.csv not found"}), 400

    project_required_skills = {"Python", "SQL", "Java", "JavaScript", "C#"}
    gaps = []
    for _, row in data.iterrows():
        emp_skills = set(str(row["skill"]).split(","))
        missing = list(project_required_skills - emp_skills)
        gaps.append({
            "name": row["name"],
            "has_skills": list(emp_skills),
            "missing_skills": missing,
            "count": len(missing)
        })
    return jsonify(gaps)

if __name__ == "__main__":
    print("🚀 Starting Predictive Resource Capacity & Workforce Risk Platform on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)