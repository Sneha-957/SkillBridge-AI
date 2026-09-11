from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

skills_data = pd.read_csv("data/skills.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    name = request.form["name"]
    education = request.form["education"]
    skills = request.form["skills"]
    job_role = request.form["job_role"]

    # Convert student's skills into a clean list
    student_skills = [
        skill.strip().lower()
        for skill in skills.split(",")
        if skill.strip()
    ]

    # Find the selected job role in the dataset
    role_data = skills_data[
        skills_data["job_role"].str.lower() == job_role.strip().lower()
    ]

    if role_data.empty:
        return "Job role not found in our dataset."

    # Get required skills for the selected role
    required_skills = [
        skill.strip().lower()
        for skill in role_data.iloc[0]["required_skills"].split(",")
        if skill.strip()
    ]

    # Find missing skills
    skill_gap = [
        skill for skill in required_skills
        if skill not in student_skills
    ]

    # Calculate skill match percentage
    matched_skills = [
        skill for skill in required_skills
        if skill in student_skills
    ]

    match_percentage = round(
        (len(matched_skills) / len(required_skills)) * 100
    )

    return f"""
    <h1>Skill Gap Analysis</h1>

    <p><b>Name:</b> {name}</p>
    <p><b>Education:</b> {education}</p>
    <p><b>Desired Job Role:</b> {job_role}</p>

    <h2>Your Skills</h2>
    <p>{", ".join(student_skills)}</p>

    <h2>Required Skills</h2>
    <p>{", ".join(required_skills)}</p>

    <h2>Missing Skills</h2>
    <p>{", ".join(skill_gap) if skill_gap else "No major skill gap found!"}</p>

    <h2>Skill Match</h2>
    <p>{match_percentage}%</p>
    """

if __name__ == "__main__":
    app.run(debug=True)