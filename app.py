
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import re
from io import BytesIO
from xml.sax.saxutils import escape

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "skills.csv")


# --------------------------------
# LEARNING RESOURCES
# --------------------------------

LEARNING_RESOURCES = {

    "python": {
        "roadmap": "Learn Python basics, variables, data types, loops, functions, OOP and file handling. Practice small Python programs.",
        "resource": "https://www.youtube.com/results?search_query=Python+programming+full+course+Hindi",
        "documentation": "https://docs.python.org/3/tutorial/",
        "practice": "https://www.hackerrank.com/domains/python"
    },

    "java": {
        "roadmap": "Learn Java syntax, variables, loops, arrays, classes, objects, inheritance and exception handling.",
        "resource": "https://www.youtube.com/results?search_query=Java+programming+full+course+Hindi",
        "documentation": "https://docs.oracle.com/javase/tutorial/",
        "practice": "https://www.hackerrank.com/domains/java"
    },

    "sql": {
        "roadmap": "Learn SQL commands, database concepts, SELECT queries, joins, grouping, subqueries and database management.",
        "resource": "https://www.youtube.com/results?search_query=SQL+complete+course+Hindi",
        "documentation": "https://www.w3schools.com/sql/",
        "practice": "https://www.hackerrank.com/domains/sql"
    },

    "html": {
        "roadmap": "Learn HTML structure, elements, headings, forms, tables, links and semantic tags.",
        "resource": "https://www.youtube.com/results?search_query=HTML+full+course+Hindi",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/HTML",
        "practice": "https://www.w3schools.com/html/"
    },

    "css": {
        "roadmap": "Learn CSS selectors, colors, backgrounds, box model, flexbox, grid and responsive web design.",
        "resource": "https://www.youtube.com/results?search_query=CSS+full+course+Hindi",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/CSS",
        "practice": "https://www.w3schools.com/css/"
    },

    "javascript": {
        "roadmap": "Learn JavaScript variables, functions, arrays, objects, DOM manipulation and events.",
        "resource": "https://www.youtube.com/results?search_query=JavaScript+full+course+Hindi",
        "documentation": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "practice": "https://www.hackerrank.com/domains/javascript"
    },

    "flask": {
        "roadmap": "Learn Flask installation, routing, templates, forms, HTTP methods and database integration.",
        "resource": "https://www.youtube.com/results?search_query=Flask+framework+tutorial+Hindi",
        "documentation": "https://flask.palletsprojects.com/en/stable/",
        "practice": "https://www.w3schools.com/python/python_flask.asp"
    },

    "machine learning": {
        "roadmap": "Learn supervised and unsupervised learning, regression, classification, clustering and model evaluation.",
        "resource": "https://www.youtube.com/results?search_query=Machine+Learning+complete+course+Hindi",
        "documentation": "https://scikit-learn.org/stable/user_guide.html",
        "practice": "https://www.kaggle.com/learn"
    },

    "git": {
        "roadmap": "Learn Git basics, repositories, commits, branches, merging and GitHub collaboration.",
        "resource": "https://www.youtube.com/results?search_query=Git+and+GitHub+tutorial+Hindi",
        "documentation": "https://git-scm.com/doc",
        "practice": "https://learngitbranching.js.org/"
    },

    "oop": {
        "roadmap": "Learn Object-Oriented Programming concepts including classes, objects, inheritance, polymorphism, encapsulation and abstraction.",
        "resource": "https://www.youtube.com/results?search_query=OOP+Object+Oriented+Programming+Hindi+tutorial",
        "documentation": "https://docs.oracle.com/javase/tutorial/java/concepts/",
        "practice": "https://www.hackerrank.com/domains/java"
    },

    "object oriented programming": {
        "roadmap": "Learn classes, objects, inheritance, polymorphism, encapsulation and abstraction with practical examples.",
        "resource": "https://www.youtube.com/results?search_query=OOP+Object+Oriented+Programming+Hindi+tutorial",
        "documentation": "https://docs.oracle.com/javase/tutorial/java/concepts/",
        "practice": "https://www.hackerrank.com/domains/java"
    }
}


# --------------------------------
# SKILL PROCESSING
# --------------------------------

def clean_skill(skill):
    """Normalize skill names for matching."""

    skill = str(skill).strip().lower()
    skill = re.sub(r"[^a-z0-9+#. ]", " ", skill)
    skill = re.sub(r"\s+", " ", skill).strip()

    aliases = {
        "object oriented programming": "oop",
        "object-oriented programming": "oop",
        "oops": "oop",
        "js": "javascript",
        "ml": "machine learning"
    }

    return aliases.get(skill, skill)


def parse_skills(skills):
    """Convert comma, semicolon or pipe-separated skills into a list."""

    if pd.isna(skills):
        return []

    if isinstance(skills, list):
        raw_skills = skills
    else:
        raw_skills = re.split(r"[,;|]", str(skills))

    return [
        clean_skill(skill)
        for skill in raw_skills
        if str(skill).strip()
    ]


# --------------------------------
# LOAD DATASET
# --------------------------------

def load_skill_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    role_column = next(
        (column for column in ["job_role", "role", "job", "career"]
         if column in df.columns),
        None
    )

    skill_column = next(
        (column for column in ["required_skills", "skills", "skill"]
         if column in df.columns),
        None
    )

    if role_column is None or skill_column is None:
        raise ValueError(
            "CSV must contain job_role and required_skills columns."
        )

    return df, role_column, skill_column


# --------------------------------
# GET JOB ROLES
# --------------------------------

def get_job_roles():

    df, role_column, skill_column = load_skill_data()

    return sorted(
        df[role_column].dropna().astype(str).unique().tolist()
    )


# --------------------------------
# SKILL ANALYSIS
# --------------------------------

def calculate_similarity(user_skills, required_skills):

    if not user_skills or not required_skills:
        return 0

    user_text = " ".join(user_skills)
    required_text = " ".join(required_skills)

    if not user_text.strip() or not required_text.strip():
        return 0

    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(
            [user_text, required_text]
        )

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(score * 100, 2)

    except ValueError:
        return 0


def analyze_skills(user_skills, required_skills):

    user_clean = set(
        clean_skill(skill) for skill in user_skills
    )

    required_clean = set(
        clean_skill(skill) for skill in required_skills
    )

    matched_skills = [
        skill for skill in required_skills
        if clean_skill(skill) in user_clean
    ]

    missing_skills = [
        skill for skill in required_skills
        if clean_skill(skill) not in user_clean
    ]

    if required_clean:
        match_percentage = round(
            (len(user_clean.intersection(required_clean))
             / len(required_clean)) * 100,
            2
        )
    else:
        match_percentage = 0

    similarity_score = calculate_similarity(
        user_skills,
        required_skills
    )

    return (
        matched_skills,
        missing_skills,
        match_percentage,
        similarity_score
    )


# --------------------------------
# GENERATE LEARNING ROADMAP
# --------------------------------

def generate_roadmap(missing_skills):

    roadmap = []

    for index, skill in enumerate(missing_skills, start=1):

        normalized_skill = clean_skill(skill)

        details = LEARNING_RESOURCES.get(
            normalized_skill,
            {
                "roadmap": (
                    f"Study the fundamentals of {skill}, "
                    "practice examples and build a small project."
                ),
                "resource": (
                    "https://www.youtube.com/results?search_query="
                    + skill.replace(" ", "+")
                    + "+tutorial"
                ),
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


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------
# PROFILE PAGE
# --------------------------------

@app.route("/profile", methods=["GET"])
def profile():

    try:
        roles = get_job_roles()
    except Exception as error:
        print("Dataset Error:", error)
        roles = []

    return render_template(
        "profile.html",
        roles=roles,
        job_roles=roles
    )


# --------------------------------
# ANALYZE SKILLS
# --------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    education = request.form.get("education", "").strip()
    job_role = request.form.get("job_role", "").strip()
    user_skills_input = request.form.get("skills", "")

    user_skills = parse_skills(user_skills_input)

    try:

        df, role_column, skill_column = load_skill_data()

        role_data = df[
            df[role_column].astype(str).str.strip().str.lower()
            == job_role.lower()
        ]

        if role_data.empty:
            raise ValueError(
                f"Job role '{job_role}' was not found in the dataset."
            )

        required_skills = parse_skills(
            role_data.iloc[0][skill_column]
        )

        matched_skills, missing_skills, match_percentage, similarity_score = (
            analyze_skills(user_skills, required_skills)
        )

        roadmap = generate_roadmap(missing_skills)

        return render_template(
            "result.html",
            name=name,
            email=email,
            education=education,
            job_role=job_role,
            match_percentage=match_percentage,
            similarity_score=similarity_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            roadmap=roadmap
        )

    except Exception as error:

        print("Analysis Error:", error)

        try:
            roles = get_job_roles()
        except Exception:
            roles = []

        return render_template(
            "profile.html",
            roles=roles,
            job_roles=roles,
            error=str(error)
        ), 400


# --------------------------------
# ROLES API
# --------------------------------

@app.route("/roles")
def roles():

    try:
        return jsonify(get_job_roles())

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# --------------------------------
# DOWNLOAD PDF REPORT
# --------------------------------

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

    content.append(Paragraph(
        "Student Information",
        styles["Heading2"]
    ))

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

    content.append(Paragraph(
        "Matched Skills",
        styles["Heading2"]
    ))

    matched_skills = request.form.getlist("matched_skills")

    for skill in matched_skills:
        content.append(Paragraph(
            "&#8226; " + escape(skill),
            styles["Normal"]
        ))

    content.append(Spacer(1, 15))

    content.append(Paragraph(
        "Skills to Improve",
        styles["Heading2"]
    ))

    missing_skills = request.form.getlist("missing_skills")

    for skill in missing_skills:
        content.append(Paragraph(
            "&#8226; " + escape(skill),
            styles["Normal"]
        ))

    content.append(Spacer(1, 15))

    content.append(Paragraph(
        "Learning Roadmap",
        styles["Heading2"]
    ))

    roadmap_skills = request.form.getlist("roadmap_skills")
    roadmap_descriptions = request.form.getlist(
        "roadmap_descriptions"
    )

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


# --------------------------------
# RUN APPLICATION
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
