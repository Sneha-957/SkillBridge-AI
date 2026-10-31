
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = "skillbridge_ai_secret_key"

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "skills.csv")


# Learning resources for missing skills
LEARNING_RESOURCES = {
    "python": {
        "roadmap": "Learn Python basics, variables, loops, functions, OOP and file handling.",
        "resource": "https://www.youtube.com/results?search_query=python+full+course",
        "documentation": "https://docs.python.org/3/",
        "practice": "https://www.hackerrank.com/domains/python"
    },

    "java": {
        "roadmap": "Learn Java syntax, OOP concepts, classes, objects and exception handling.",
        "resource": "https://www.youtube.com/results?search_query=java+full+course",
        "documentation": "https://docs.oracle.com/en/java/",
        "practice": "https://www.hackerrank.com/domains/java"
    },

    "sql": {
        "roadmap": "Learn SQL queries, joins, subqueries, grouping and database management.",
        "resource": "https://www.youtube.com/results?search_query=sql+full+course",
        "documentation": "https://www.w3schools.com/sql/",
        "practice": "https://www.hackerrank.com/domains/sql"
    },

    "html": {
        "roadmap": "Learn HTML tags, forms, tables, semantic elements and page structure.",
        "resource": "https://www.youtube.com/results?search_query=html+full+course",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/HTML",
        "practice": "https://www.w3schools.com/html/"
    },

    "css": {
        "roadmap": "Learn CSS selectors, colors, layouts, Flexbox, Grid and responsive design.",
        "resource": "https://www.youtube.com/results?search_query=css+full+course",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/CSS",
        "practice": "https://www.w3schools.com/css/"
    },

    "javascript": {
        "roadmap": "Learn JavaScript variables, functions, DOM manipulation and events.",
        "resource": "https://www.youtube.com/results?search_query=javascript+full+course",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "practice": "https://www.hackerrank.com/domains/javascript"
    },

    "flask": {
        "roadmap": "Learn Flask routing, templates, forms and database integration.",
        "resource": "https://www.youtube.com/results?search_query=flask+python+tutorial",
        "documentation": "https://flask.palletsprojects.com/",
        "practice": "https://www.w3schools.com/python/python_flask.asp"
    },

    "machine learning": {
        "roadmap": "Learn supervised learning, unsupervised learning, model training and evaluation.",
        "resource": "https://www.youtube.com/results?search_query=machine+learning+course",
        "documentation": "https://scikit-learn.org/stable/",
        "practice": "https://www.kaggle.com/learn"
    },

    "git": {
        "roadmap": "Learn Git commands, repositories, commits, branches and GitHub.",
        "resource": "https://www.youtube.com/results?search_query=git+github+tutorial",
        "documentation": "https://git-scm.com/doc",
        "practice": "https://skills.github.com/"
    }
}

def clean_skill(skill):
    """Normalize skill names."""
    return re.sub(r"\s+", " ", str(skill).strip().lower())


def parse_skills(value):
    """Convert comma-separated skills into a list."""
    if pd.isna(value):
        return []

    return [
        clean_skill(skill)
        for skill in re.split(r"[,;|]", str(value))
        if clean_skill(skill)
    ]


def load_skill_data():
    """Load job role and required skills from CSV."""

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "skills.csv not found inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    df.columns = [str(col).strip().lower() for col in df.columns]

    role_column = next(
        (c for c in ["job_role", "role", "job", "career"]
         if c in df.columns),
        None
    )

    skills_column = next(
        (c for c in ["required_skills", "skills", "skill"]
         if c in df.columns),
        None
    )

    if role_column is None or skills_column is None:
        raise ValueError(
            "CSV must contain job role and required skills columns."
        )

    jobs = {}

    for _, row in df.iterrows():
        role = str(row[role_column]).strip()

        if not role or role.lower() == "nan":
            continue

        jobs[role] = parse_skills(row[skills_column])

    return jobs


def calculate_similarity(student_skills, required_skills):
    """Calculate TF-IDF cosine similarity."""

    if not student_skills or not required_skills:
        return 0.0

    student_text = " ".join(student_skills)
    required_text = " ".join(required_skills)

    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform(
            [student_text, required_text]
        )

        score = cosine_similarity(
            vectors[0:1], vectors[1:2]
        )[0][0]

        return round(float(score) * 100, 2)

    except ValueError:
        return 0.0


def analyze_skills(student_skills, required_skills):
    """Compare student skills with job requirements."""

    matched_skills = [
        skill for skill in required_skills
        if skill in student_skills
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in student_skills
    ]

    total = len(required_skills)

    match_percentage = (
        round((len(matched_skills) / total) * 100, 2)
        if total else 0
    )

    similarity_score = calculate_similarity(
        student_skills, required_skills
    )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage,
        "similarity_score": similarity_score
    }


def generate_roadmap(missing_skills):
    """Generate a learning roadmap for missing skills."""

    roadmap = []

    for index, skill in enumerate(missing_skills, start=1):

        details = LEARNING_RESOURCES.get(
            skill.lower(),
            {
                "roadmap": f"Study the fundamentals of {skill}, practice examples and build a small project.",
                "resource": "https://www.youtube.com/",
                "documentation": "https://www.w3schools.com/",
                "practice": "https://www.hackerrank.com/"
            }
        )

        roadmap.append({
            "step": index,
            "skill": skill.title(),
            "description": details["roadmap"],
            "resource": details["resource"],
            "documentation": details["documentation"],
            "practice": details["practice"]
        })

    return roadmap


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Student profile page
@app.route("/profile", methods=["GET", "POST"])
def profile():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        education = request.form.get("education", "").strip()

        if not name:
            flash("Please enter your name.")
            return redirect(url_for("profile"))

        return render_template(
            "profile.html",
            name=name,
            email=email,
            education=education
        )

    return render_template("profile.html")


# Skill gap analysis
@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    if request.method == "POST":

        name = request.form.get("name", "Student").strip()
        email = request.form.get("email", "").strip()
        education = request.form.get("education", "").strip()

        selected_role = request.form.get(
            "job_role", ""
        ).strip()

        entered_skills = request.form.get(
            "skills", ""
        )

        student_skills = parse_skills(entered_skills)

        if not student_skills:
            flash("Please enter at least one skill.")
            return redirect(url_for("profile"))

        try:
            jobs = load_skill_data()

        except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
            return f"Dataset Error: {error}", 500

        if selected_role not in jobs:
            return "Selected job role was not found in the dataset.", 400

        required_skills = jobs[selected_role]

        results = analyze_skills(
            student_skills,
            required_skills
        )

        roadmap = generate_roadmap(
            results["missing_skills"]
        )

        return render_template(
            "result.html",
            name=name,
            email=email,
            education=education,
            job_role=selected_role,
            student_skills=student_skills,
            required_skills=required_skills,
            matched_skills=results["matched_skills"],
            missing_skills=results["missing_skills"],
            match_percentage=results["match_percentage"],
            similarity_score=results["similarity_score"],
            roadmap=roadmap
        )

    return redirect(url_for("profile"))


# Display available job roles
@app.route("/roles")
def roles():

    try:
        jobs = load_skill_data()
        return {
            "available_roles": list(jobs.keys())
        }

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        return {"error": str(error)}, 500


from xml.sax.saxutils import escape


@app.route("/download-report", methods=["POST"])
def download_report():

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(
        "SkillBridge AI - Skill Gap Analysis Report",
        styles["Title"]
    ))

    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Student Information</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    fields = [
        ("Name", request.form.get("name", "")),
        ("Email", request.form.get("email", "")),
        ("Education", request.form.get("education", "")),
        ("Target Job Role", request.form.get("job_role", "")),
        ("Skill Match", request.form.get("match_percentage", "") + "%"),
        ("TF-IDF Similarity", request.form.get("similarity_score", "") + "%")
    ]

    for label, value in fields:
        content.append(Paragraph(
            f"<b>{escape(label)}:</b> {escape(str(value))}",
            styles["Normal"]
        ))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 15))
    content.append(Paragraph("Matched Skills", styles["Heading2"]))

    matched_skills = request.form.getlist("matched_skills")

    for skill in matched_skills:
        content.append(Paragraph(
            "• " + escape(skill), styles["Normal"]
        ))

    content.append(Spacer(1, 15))
    content.append(Paragraph("Skills to Improve", styles["Heading2"]))

    missing_skills = request.form.getlist("missing_skills")

    for skill in missing_skills:
        content.append(Paragraph(
            "• " + escape(skill), styles["Normal"]
        ))

    content.append(Spacer(1, 15))
    content.append(Paragraph("Learning Roadmap", styles["Heading2"]))

    roadmap_skills = request.form.getlist("roadmap_skills")
    roadmap_descriptions = request.form.getlist("roadmap_descriptions")

    for index, skill in enumerate(roadmap_skills):
        content.append(Paragraph(
            f"<b>{index + 1}. {escape(skill)}</b>",
            styles["Normal"]
        ))

        if index < len(roadmap_descriptions):
            content.append(Paragraph(
                escape(roadmap_descriptions[index]),
                styles["Normal"]
            ))

        content.append(Spacer(1, 10))

    doc.build(content)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="SkillBridge_AI_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)
