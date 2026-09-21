from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load job role and required skills dataset
skills_data = pd.read_csv("data/skills.csv")


# Personalized Learning Roadmap
roadmap_data = {
    "flask": [
        "Learn Flask basics",
        "Create Flask routes",
        "Build Flask forms",
        "Connect Flask with SQLite"
    ],
    "git": [
        "Learn Git basics",
        "Practice add, commit and push",
        "Learn branches",
        "Learn GitHub collaboration"
    ],
    "oop": [
        "Understand classes and objects",
        "Learn inheritance",
        "Learn encapsulation",
        "Practice OOP programs in Python"
    ]
}


# Learning Resources
resource_data = {
    "flask": [
        {
            "title": "Flask Official Documentation",
            "url": "https://flask.palletsprojects.com/"
        },
        {
            "title": "Flask Tutorial",
            "url": "https://flask.palletsprojects.com/en/stable/tutorial/"
        }
    ],
    "git": [
        {
            "title": "Git Official Documentation",
            "url": "https://git-scm.com/doc"
        },
        {
            "title": "GitHub Skills",
            "url": "https://skills.github.com/"
        }
    ],
    "oop": [
        {
            "title": "Python Classes and Objects",
            "url": "https://docs.python.org/3/tutorial/classes.html"
        },
        {
            "title": "Python OOP Tutorial",
            "url": "https://www.geeksforgeeks.org/python/python-oops-concepts/"
        }
    ]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    # Get student information from form
    name = request.form["name"]
    education = request.form["education"]
    skills = request.form["skills"]
    job_role = request.form["job_role"]

    # Convert student's skills into a list
    student_skills = [
        skill.strip().lower()
        for skill in skills.split(",")
        if skill.strip()
    ]

    # Find selected job role in dataset
    role_data = skills_data[
        skills_data["job_role"].str.lower()
        == job_role.strip().lower()
    ]

    # If job role does not exist
    if role_data.empty:
        return "Job role not found in our dataset."

    # Get required skills for selected job role
    required_skills = [
        skill.strip().lower()
        for skill in role_data.iloc[0]["required_skills"].split(",")
        if skill.strip()
    ]

    # Find matched skills
    matched_skills = [
        skill
        for skill in required_skills
        if skill in student_skills
    ]

    # Find missing skills
    skill_gap = [
        skill
        for skill in required_skills
        if skill not in student_skills
    ]

    # Calculate skill match percentage
    match_percentage = round(
        (len(matched_skills) / len(required_skills)) * 100
    )

    # Generate personalized learning roadmap
    learning_roadmap = {}

    for skill in skill_gap:
        if skill in roadmap_data:
            learning_roadmap[skill] = roadmap_data[skill]

    # Get learning resources for missing skills
    learning_resources = {}

    for skill in skill_gap:
        if skill in resource_data:
            learning_resources[skill] = resource_data[skill]

    # Result page
    return f"""
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Skill Gap Analysis - SkillBridge AI</title>

    <style>

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #222;
        }}

        .navbar {{
            background: white;
            padding: 20px 60px;
            font-size: 25px;
            font-weight: bold;
        }}

        .container {{
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
        }}

        .title {{
            text-align: center;
            margin-bottom: 35px;
        }}

        .title h1 {{
            font-size: 38px;
            margin-bottom: 10px;
        }}

        .title p {{
            color: #666;
        }}

        .card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}

        .card h2 {{
            margin-bottom: 15px;
        }}

        .info p {{
            margin: 8px 0;
        }}

        .match {{
            text-align: center;
        }}

        .percentage {{
            font-size: 50px;
            font-weight: bold;
            margin: 15px 0;
        }}

        .skills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .skill {{
            padding: 10px 15px;
            border-radius: 20px;
            background: #eee;
        }}

        .missing {{
            background: #ffe5e5;
        }}

        .matched {{
            background: #e5f7e5;
        }}

        .roadmap-item {{
            margin-bottom: 25px;
        }}

        .roadmap-item h3 {{
            margin-bottom: 12px;
        }}

        .roadmap-item ol {{
            padding-left: 25px;
            line-height: 2;
        }}

        .resource {{
            margin-bottom: 20px;
        }}

        .resource h3 {{
            margin-bottom: 10px;
        }}

        .resource a {{
            display: inline-block;
            padding: 10px 18px;
            background: #222;
            color: white;
            text-decoration: none;
            border-radius: 7px;
            margin-top: 5px;
        }}

        .resource a:hover {{
            opacity: 0.8;
        }}

        .btn {{
            display: inline-block;
            margin-top: 15px;
            padding: 12px 25px;
            background: #222;
            color: white;
            text-decoration: none;
            border-radius: 8px;
        }}

    </style>

</head>


<body>

    <nav class="navbar">
        SkillBridge AI
    </nav>


    <div class="container">


        <div class="title">

            <h1>Skill Gap Analysis</h1>

            <p>
                Your personalized career readiness report
            </p>

        </div>


        <!-- Student Profile -->

        <div class="card info">

            <h2>Student Profile</h2>

            <p>
                <b>Name:</b> {name}
            </p>

            <p>
                <b>Education:</b> {education}
            </p>

            <p>
                <b>Desired Job Role:</b> {job_role}
            </p>

        </div>


        <!-- Skill Match -->

        <div class="card match">

            <h2>Skill Match</h2>

            <div class="percentage">
                {match_percentage}%
            </div>

            <p>
                Your current skills match
                {match_percentage}% of the required skills.
            </p>

        </div>


        <!-- Matched Skills -->

        <div class="card">

            <h2>✅ Matched Skills</h2>

            <div class="skills">

                {
                    "".join(
                        f'<span class="skill matched">{skill}</span>'
                        for skill in matched_skills
                    )
                    if matched_skills
                    else '<span class="skill">No matched skills</span>'
                }

            </div>

        </div>


        <!-- Missing Skills -->

        <div class="card">

            <h2>❌ Missing Skills</h2>

            <div class="skills">

                {
                    "".join(
                        f'<span class="skill missing">{skill}</span>'
                        for skill in skill_gap
                    )
                    if skill_gap
                    else '<span class="skill matched">No major skill gap found!</span>'
                }

            </div>

        </div>


        <!-- Personalized Learning Roadmap -->

        <div class="card">

            <h2>📚 Personalized Learning Roadmap</h2>

            {
                "".join(
                    f"""
                    <div class="roadmap-item">

                        <h3>{skill.title()}</h3>

                        <ol>
                            {
                                "".join(
                                    f"<li>{step}</li>"
                                    for step in steps
                                )
                            }
                        </ol>

                    </div>
                    """
                    for skill, steps in learning_roadmap.items()
                )
                if learning_roadmap
                else """
                <p>
                    Your current skills cover the selected
                    role requirements. Keep improving your
                    existing skills!
                </p>
                """
            }

        </div>


        <!-- Learning Resources -->

        <div class="card">

            <h2>🌐 Learning Resources</h2>

            {
                "".join(
                    f"""
                    <div class="resource">

                        <h3>{skill.title()}</h3>

                        {
                            "".join(
                                f'''
                                <p>
                                    <b>{resource["title"]}</b>
                                </p>

                                <a href="{resource["url"]}"
                                   target="_blank">
                                    Learn Now
                                </a>

                                <br><br>
                                '''
                                for resource in resources
                            )
                        }

                    </div>
                    """
                    for skill, resources
                    in learning_resources.items()
                )
                if learning_resources
                else """
                <p>
                    No additional resources are required.
                </p>
                """
            }

        </div>


        <!-- Next Step -->

        <div class="card">

            <h2>🎯 Next Step</h2>

            <p>
                Focus on developing the missing skills
                to improve your readiness for the selected
                job role.
            </p>

        </div>


        <a href="/profile" class="btn">
            Analyze Again
        </a>


    </div>

</body>

</html>
"""


if __name__ == "__main__":
    app.run(debug=True)