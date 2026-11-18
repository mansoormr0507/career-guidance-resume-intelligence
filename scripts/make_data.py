"""Create all data files used by the project (run once; the outputs are committed).

Creates:
  data/skills_taxonomy.json            skills grouped by category, with aliases
  data/job_roles.csv                   22 job roles (hand-curated from typical job postings)
  data/sample_resumes/*.pdf            5 FICTIONAL candidate resumes + gold_skills.json (answer key)
  data/sample_job_descriptions/*.txt   3 sample job descriptions
  data/eval/synthetic_resumes.csv      264 synthetic labelled resumes for evaluating the recommender

Everything here is fictional/synthetic: no real person's data is used.
Usage: python scripts/make_data.py
"""
import csv
import json
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
rnd = random.Random(42)

# ----------------------------------------------------------------------------- taxonomy
TAXONOMY = {
    "Programming Languages": {
        "python": ["python3"], "java": [], "javascript": ["js", "ecmascript"], "typescript": ["ts"],
        "c++": ["cpp"], "c#": ["csharp", "c sharp"], "c programming": ["c language"], "r programming": ["r language", "rstudio"],
        "golang": ["go language"], "kotlin": [], "swift": [], "php": [], "ruby": [], "scala": [], "matlab": [],
        "bash": ["shell scripting", "shell script"], "sql": ["structured query language"], "dart": [],
    },
    "Web Development": {
        "html": ["html5"], "css": ["css3"], "react": ["reactjs", "react.js"], "angular": ["angularjs"],
        "vue.js": ["vue", "vuejs"], "node.js": ["nodejs", "node js"], "express.js": ["expressjs"],
        "django": [], "flask": [], "fastapi": [], "spring boot": ["springboot", "spring framework"],
        "rest api": ["rest apis", "restful api", "restful apis"], "graphql": [], "bootstrap": [],
        "tailwind css": ["tailwind"], "jquery": [],
    },
    "Data Science & AI": {
        "machine learning": ["ml"], "deep learning": ["neural networks"], "nlp": ["natural language processing"],
        "computer vision": [], "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"],
        "tensorflow": [], "keras": [], "pytorch": [], "matplotlib": [], "seaborn": [],
        "statistics": ["statistical analysis"], "data analysis": ["data analytics"],
        "data visualization": ["data visualisation"], "feature engineering": [], "opencv": [],
        "hugging face": ["huggingface", "transformers library"], "large language models": ["llm", "llms"],
        "generative ai": ["genai", "gen ai"], "prompt engineering": [], "time series": ["time-series"],
        "a/b testing": ["ab testing"], "mlops": [], "model deployment": ["model serving"], "xgboost": [],
    },
    "Databases & Data Engineering": {
        "mysql": [], "postgresql": ["postgres"], "mongodb": [], "redis": [], "nosql": [],
        "apache spark": ["spark", "pyspark"], "hadoop": [], "apache kafka": ["kafka"],
        "apache airflow": ["airflow"], "etl": ["elt", "etl pipelines"], "data warehousing": ["data warehouse"],
        "power bi": ["powerbi"], "tableau": [], "excel": ["ms excel", "microsoft excel", "advanced excel"],
        "snowflake": [], "big data": [], "data modeling": ["data modelling"], "firebase": [],
    },
    "Cloud & DevOps": {
        "aws": ["amazon web services"], "azure": ["microsoft azure"], "google cloud": ["gcp", "google cloud platform"],
        "docker": [], "kubernetes": ["k8s"], "terraform": [], "jenkins": [], "ci/cd": ["cicd", "continuous integration"],
        "git": [], "github": [], "linux": [], "ansible": [], "prometheus": [], "grafana": [],
        "microservices": ["microservice"], "infrastructure as code": ["iac"], "networking": ["computer networks"],
    },
    "Testing & QA": {
        "selenium": [], "pytest": [], "junit": [], "unit testing": [], "test automation": ["automation testing"],
        "postman": [], "manual testing": [], "api testing": [], "jira": [],
    },
    "Cybersecurity": {
        "cybersecurity": ["cyber security", "information security"], "network security": [],
        "penetration testing": ["pentesting", "pen testing"], "siem": ["splunk"], "ethical hacking": [],
        "wireshark": [], "owasp": [], "vulnerability assessment": [], "incident response": [], "nmap": [], "kali linux": [],
    },
    "Mobile": {"android": ["android studio", "android development"], "ios": [], "flutter": [], "react native": []},
    "Computer Science Fundamentals": {
        "data structures": ["algorithms", "dsa", "data structures and algorithms"],
        "oop": ["object oriented programming", "object-oriented programming"],
        "design patterns": [], "system design": [],
    },
    "Product, Design & Business": {
        "figma": [], "ui/ux design": ["ux design", "ui design", "user experience design"], "wireframing": ["prototyping"],
        "user research": [], "product management": ["product roadmap"], "agile": [], "scrum": [],
        "project management": [], "stakeholder management": [], "requirement gathering": ["requirements gathering"],
        "business analysis": [],
    },
    "Soft Skills": {
        "communication": ["communication skills"], "leadership": [], "teamwork": ["team work", "collaboration"],
        "problem solving": ["problem-solving"], "critical thinking": [], "time management": [],
    },
}
ALL_SKILLS = {s for cat in TAXONOMY.values() for s in cat}

# ----------------------------------------------------------------------------- job roles
CS_FIELDS = "computer science; information technology; engineering; mathematics; statistics; data science"
ROLES = [
    ("Data Scientist", "Analyses large datasets to build predictive models, run experiments and communicate insights to business teams using statistics and machine learning.",
     "python,machine learning,statistics,pandas,numpy,scikit-learn,sql,data visualization", "deep learning,a/b testing,feature engineering,tableau", "Bachelor", "entry-mid", CS_FIELDS),
    ("Machine Learning Engineer", "Builds, trains, deploys and monitors machine learning models in production, working with data pipelines and cloud infrastructure.",
     "python,machine learning,deep learning,pytorch,tensorflow,docker,git,mlops", "aws,kubernetes,model deployment,ci/cd,fastapi", "Bachelor", "mid", CS_FIELDS),
    ("Data Analyst", "Collects, cleans and analyses data, and builds dashboards and reports that support business decisions.",
     "sql,excel,data analysis,data visualization,power bi,tableau,statistics", "python,pandas,communication,etl", "Bachelor", "entry", CS_FIELDS + "; business; commerce; economics"),
    ("Data Engineer", "Designs and maintains data pipelines, warehouses and big data systems that make reliable data available for analytics.",
     "python,sql,etl,apache spark,apache airflow,data warehousing,apache kafka", "aws,snowflake,docker,hadoop,data modeling", "Bachelor", "mid", CS_FIELDS),
    ("Generative AI Engineer", "Develops applications powered by large language models, prompt engineering and AI APIs, and integrates them into products.",
     "python,generative ai,large language models,prompt engineering,nlp,hugging face,fastapi,rest api", "pytorch,docker,machine learning,git", "Bachelor", "entry-mid", CS_FIELDS),
    ("NLP Engineer", "Builds language understanding systems such as text classification, information extraction and chatbots using natural language processing and deep learning.",
     "python,nlp,machine learning,deep learning,hugging face,pytorch,scikit-learn", "tensorflow,large language models,docker", "Bachelor", "mid", CS_FIELDS),
    ("Computer Vision Engineer", "Develops image and video analysis systems such as detection, recognition and tracking using computer vision and deep learning.",
     "python,computer vision,opencv,deep learning,pytorch,tensorflow,numpy,c++", "docker,model deployment,machine learning", "Bachelor", "mid", CS_FIELDS + "; electronics"),
    ("Backend Developer", "Builds server-side applications, APIs and databases that power web and mobile products.",
     "python,rest api,sql,postgresql,git,docker,microservices", "django,flask,fastapi,redis,aws,ci/cd", "Bachelor", "entry-mid", CS_FIELDS),
    ("Java Developer", "Develops enterprise applications and services in Java using Spring Boot, relational databases and REST APIs.",
     "java,spring boot,rest api,sql,mysql,git,microservices,oop", "docker,kubernetes,junit,aws", "Bachelor", "entry-mid", CS_FIELDS),
    ("Frontend Developer", "Creates responsive, accessible user interfaces for web applications using modern JavaScript frameworks.",
     "javascript,html,css,react,git,rest api,typescript", "tailwind css,bootstrap,figma,vue.js,angular", "Bachelor", "entry", CS_FIELDS),
    ("Full Stack Developer", "Builds complete web applications covering the user interface, server logic, APIs and databases.",
     "javascript,react,node.js,express.js,mongodb,html,css,rest api,git", "typescript,docker,aws,sql", "Bachelor", "entry-mid", CS_FIELDS),
    ("DevOps Engineer", "Automates build, test and deployment pipelines and manages containerised infrastructure for reliable software delivery.",
     "linux,docker,kubernetes,ci/cd,jenkins,git,terraform,aws", "ansible,prometheus,grafana,bash,python,infrastructure as code", "Bachelor", "mid", CS_FIELDS),
    ("Cloud Engineer", "Designs, deploys and secures applications on public cloud platforms using infrastructure as code.",
     "aws,azure,google cloud,linux,terraform,networking,docker,infrastructure as code", "kubernetes,python,bash,prometheus", "Bachelor", "mid", CS_FIELDS),
    ("Cybersecurity Analyst", "Monitors systems for threats, assesses vulnerabilities and responds to security incidents to protect organisational data.",
     "cybersecurity,network security,siem,vulnerability assessment,incident response,linux,owasp,networking", "penetration testing,wireshark,ethical hacking,python", "Bachelor", "entry-mid", CS_FIELDS),
    ("QA Test Automation Engineer", "Designs and automates tests to ensure software quality across web and API layers.",
     "test automation,selenium,pytest,api testing,postman,manual testing,jira,git", "python,java,ci/cd,jenkins,agile", "Bachelor", "entry-mid", CS_FIELDS),
    ("Mobile App Developer", "Builds native and cross-platform mobile applications for Android and iOS.",
     "android,kotlin,java,firebase,rest api,git", "flutter,dart,react native,ios,swift", "Bachelor", "entry-mid", CS_FIELDS),
    ("Business Intelligence Analyst", "Turns business data into dashboards and models using BI tools, data warehouses and ETL processes.",
     "power bi,tableau,sql,data warehousing,etl,excel,data visualization,data modeling", "python,communication,stakeholder management", "Bachelor", "entry-mid", CS_FIELDS + "; business; commerce"),
    ("Business Analyst", "Bridges business and technology teams by gathering requirements, analysing processes and documenting solutions.",
     "requirement gathering,stakeholder management,business analysis,sql,excel,jira,agile,communication", "power bi,tableau,project management", "Bachelor", "entry-mid", "business; commerce; economics; " + CS_FIELDS),
    ("Technical Product Manager", "Owns the product roadmap, prioritises features with engineering and design, and measures product success with data.",
     "product management,agile,scrum,stakeholder management,communication,data analysis,requirement gathering,leadership", "sql,jira,figma,a/b testing", "Bachelor", "mid", "business; engineering; " + CS_FIELDS),
    ("UI/UX Designer", "Researches user needs and designs intuitive interfaces, wireframes and prototypes.",
     "ui/ux design,figma,wireframing,user research,communication", "html,css,problem solving,teamwork", "Bachelor", "entry", "design; arts; " + CS_FIELDS),
    ("Software Engineer", "Designs, writes, tests and maintains software using strong computer science fundamentals and version control.",
     "data structures,oop,git,sql,problem solving,design patterns,system design", "java,python,c++,docker,agile", "Bachelor", "entry-mid", CS_FIELDS),
    ("Database Administrator", "Installs, tunes, secures and backs up relational and NoSQL databases to keep data available and fast.",
     "sql,mysql,postgresql,data modeling,linux,nosql,mongodb", "aws,bash,python,etl", "Bachelor", "mid", CS_FIELDS),
]
for r in ROLES:
    for s in (r[2] + "," + r[3]).split(","):
        assert s in ALL_SKILLS, f"unknown skill in role {r[0]}: {s}"

# ----------------------------------------------------------------------------- sample JDs
JDS = {
    "data_scientist_jd.txt": """Data Scientist - Northwind Analytics (fictional company)

We are looking for a Data Scientist to join our analytics team. You will build predictive models, design experiments and present insights to business stakeholders.

Requirements:
- Strong programming skills in Python and SQL
- Hands-on experience with pandas, numpy and scikit-learn
- Solid understanding of statistics and machine learning
- Experience with data visualization tools such as Tableau or Power BI
- Excellent communication skills

Nice to have:
- Knowledge of deep learning (PyTorch or TensorFlow)
- Experience with A/B testing and feature engineering
- Familiarity with AWS
""",
    "full_stack_developer_jd.txt": """Full Stack Developer - Brightwave Technologies (fictional company)

You will build and maintain web applications end to end for our customers.

Requirements:
- Proficiency in JavaScript, TypeScript, HTML and CSS
- Experience with React and Node.js
- Experience designing REST API services
- Working knowledge of MongoDB or PostgreSQL
- Version control with Git, experience with Docker is a plus
- Good teamwork and problem solving
""",
    "devops_engineer_jd.txt": """DevOps Engineer - Cloudmesh Systems (fictional company)

Join our platform team to automate delivery and run reliable infrastructure.

Requirements:
- Strong Linux administration and Bash scripting
- Hands-on experience with Docker and Kubernetes
- CI/CD pipelines with Jenkins or GitHub
- Infrastructure as code using Terraform
- Experience with AWS or Azure
- Monitoring using Prometheus and Grafana
- Python for automation
""",
}

# ----------------------------------------------------------------------------- sample resumes
CANDIDATES = [
    dict(file="aarav_mehta", name="Aarav Mehta", degree="B.Tech in Computer Science and Engineering", school="Greenfield Institute of Technology", years="2021 - 2025", cgpa="8.4 CGPA",
         skills=["python", "pandas", "numpy", "scikit-learn", "sql", "matplotlib", "statistics", "tableau", "machine learning", "data visualization", "git", "excel"],
         exp=[("Data Science Intern", "Northwind Analytics", "Jun 2024 - Aug 2024",
               ["Built a churn prediction model using Python, pandas and scikit-learn.", "Created Tableau dashboards that tracked weekly customer metrics.", "Wrote SQL queries to extract and clean data for analysis."])],
         projects=[("Student Performance Predictor", ["Trained Logistic Regression and Random Forest models with scikit-learn on a student dataset.", "Explored data with pandas and matplotlib and evaluated models using cross-validation."]),
                   ("Sales Dashboard", ["Designed an interactive Tableau dashboard from sales data cleaned with SQL and Excel."])],
         certs=["Google Data Analytics Professional Certificate (Coursera)", "Machine Learning Specialization (Coursera)"],
         also=["data analysis"],  # "Google Data Analytics" certificate
         interests=["Data science", "Predictive modelling", "Open-source contributions"], summary="Final-year computer science student interested in data science and analytics."),
    dict(file="priya_nair", name="Priya Nair", degree="B.E. in Information Technology", school="Lakeview College of Engineering", years="2019 - 2023", cgpa="8.1 CGPA",
         skills=["javascript", "typescript", "react", "node.js", "express.js", "mongodb", "html", "css", "rest api", "git", "docker", "tailwind css", "teamwork"],
         exp=[("Software Developer", "Brightwave Technologies", "Jul 2023 - Present",
               ["Developed responsive React interfaces with TypeScript and Tailwind CSS.", "Built REST API services with Node.js, Express.js and MongoDB.", "Containerised services with Docker and collaborated with a team of 6 developers."])],
         projects=[("Expense Tracker Web App", ["Full stack app using React, Node.js, Express.js and MongoDB with JWT authentication."]),
                   ("Portfolio Website", ["Designed with HTML, CSS and JavaScript and deployed using GitHub."])],
         certs=["Meta Front-End Developer Certificate (Coursera)"],
         also=["github", "ui/ux design"],  # "deployed using GitHub"; interest "UI design"
         interests=["Web development", "UI design", "Developer tooling"], summary="Full stack developer with 3 years of experience building web applications."),
    dict(file="rohan_iyer", name="Rohan Iyer", degree="M.Tech in Computer Science", school="Riverstone University", years="2016 - 2018", cgpa="8.9 CGPA",
         skills=["linux", "docker", "kubernetes", "jenkins", "terraform", "aws", "git", "bash", "python", "ci/cd", "prometheus", "grafana", "ansible", "infrastructure as code"],
         exp=[("DevOps Engineer", "Cloudmesh Systems", "Aug 2020 - Present", ["Automated CI/CD pipelines with Jenkins and Git, reducing release time.", "Managed Kubernetes clusters and Docker images on AWS.", "Provisioned infrastructure as code with Terraform and Ansible.", "Set up monitoring dashboards using Prometheus and Grafana."]),
              ("Systems Engineer", "Datacore Solutions", "Jul 2018 - Jul 2020", ["Administered Linux servers and wrote Bash and Python automation scripts."])],
         projects=[("Self-service Deployment Platform", ["Built an internal deployment tool using Python, Docker and Jenkins."])],
         certs=["AWS Certified Solutions Architect - Associate", "Certified Kubernetes Administrator (CKA)"],
         interests=["Site reliability", "Cloud infrastructure", "Automation"], summary="DevOps engineer with 8 years of experience automating delivery and operating cloud infrastructure."),
    dict(file="sneha_kulkarni", name="Sneha Kulkarni", degree="B.Sc in Information Technology", school="Hillcrest College", years="2020 - 2023", cgpa="7.9 CGPA",
         skills=["cybersecurity", "network security", "wireshark", "nmap", "kali linux", "owasp", "siem", "linux", "python", "networking", "vulnerability assessment", "incident response"],
         exp=[("Security Analyst Trainee", "Datacore Solutions", "Jan 2024 - Present", ["Monitored alerts in a SIEM and followed incident response playbooks.", "Ran vulnerability assessment scans and documented findings against the OWASP Top 10.", "Analysed traffic with Wireshark and scanned hosts using Nmap."])],
         projects=[("Home Security Lab", ["Built a lab with Kali Linux and Linux servers to practise network security and ethical hacking."])],
         certs=["Certified Ethical Hacker (EC-Council)", "CompTIA Security+"],
         also=["ethical hacking"],  # "network security and ethical hacking" in a project bullet
         interests=["Cybersecurity", "Capture the flag competitions", "Threat hunting"], summary="Security analyst trainee passionate about defending networks."),
    dict(file="kabir_singh", name="Kabir Singh", degree="B.Tech in Artificial Intelligence and Data Science", school="Northgate Institute of Technology", years="2021 - 2025", cgpa="8.7 CGPA",
         skills=["python", "large language models", "prompt engineering", "nlp", "hugging face", "fastapi", "pytorch", "git", "docker", "generative ai", "rest api", "machine learning"],
         exp=[("AI Engineering Intern", "Brightwave Technologies", "May 2024 - Jul 2024", ["Built a document question-answering prototype using large language models and prompt engineering.", "Exposed the model through a FastAPI REST API and packaged it with Docker."])],
         projects=[("Resume Chatbot", ["Created a terminal chatbot that answers career questions using a generative AI API and NLP preprocessing."]),
                   ("Sentiment Classifier", ["Fine-tuned a Hugging Face transformer with PyTorch on a public review dataset."])],
         certs=["Deep Learning Specialization (Coursera)", "Hugging Face NLP Course (Certificate of Completion)"],
         also=["deep learning"],  # "Deep Learning Specialization" certificate
         interests=["Generative AI", "Natural language processing", "Open-source AI"], summary="AI engineering graduate focused on building applications with language models."),
]


def write_resume_pdf(c, path):
    ss = getSampleStyleSheet()
    name = ParagraphStyle("n", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1e3a8a"), spaceBefore=10, spaceAfter=3)
    b = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=13)
    bl = ParagraphStyle("bl", parent=b, leftIndent=12)
    first = c["name"].split()[0].lower()
    st = [Paragraph(c["name"], name),
          Paragraph(f"{first}.{c['name'].split()[1].lower()}@example.com | +91 90000 0{rnd.randint(1000, 9999)} | linkedin.com/in/{first}-{c['name'].split()[1].lower()} | github.com/{first}{c['name'].split()[1].lower()}", b),
          Paragraph("SUMMARY", h), Paragraph(c["summary"], b), Paragraph("EDUCATION", h),
          Paragraph(f"<b>{c['degree']}</b>", b), Paragraph(f"{c['school']} | {c['years']} | {c['cgpa']}", b), Paragraph("SKILLS", h),
          Paragraph("Technical Skills: " + ", ".join(s.title() if len(s) > 3 else s.upper() for s in c["skills"]), b), Paragraph("EXPERIENCE", h)]
    for title, org, dates, bullets in c["exp"]:
        st += [Paragraph(f"<b>{title}</b>, {org} | {dates}", b)] + [Paragraph("- " + x, bl) for x in bullets] + [Spacer(1, 3)]
    st.append(Paragraph("PROJECTS", h))
    for title, bullets in c["projects"]:
        st += [Paragraph(f"<b>{title}</b>", b)] + [Paragraph("- " + x, bl) for x in bullets] + [Spacer(1, 3)]
    st += [Paragraph("CERTIFICATIONS", h)] + [Paragraph("- " + x, bl) for x in c["certs"]]
    st += [Paragraph("INTERESTS", h), Paragraph(", ".join(c["interests"]), b)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=c["name"] + " - Resume").build(st)


def skill_mentions(text_skills):
    """Canonical skills actually written in the resume text (used as the answer key)."""
    return sorted(set(text_skills))


# ----------------------------------------------------------------------------- synthetic evaluation set
def alias_variant(skill):
    for cat in TAXONOMY.values():
        if skill in cat and cat[skill] and rnd.random() < 0.35:
            return rnd.choice(cat[skill])
    return skill


def synth_resume(role):
    title, _, req, pref, *_ = role
    req, pref = req.split(","), pref.split(",")
    own = set(req + pref)
    chosen = rnd.sample(req, max(3, int(len(req) * rnd.uniform(0.55, 0.9)))) + rnd.sample(pref, int(len(pref) * rnd.uniform(0, 0.6)))
    # distractors: skills from a *neighbouring* role (makes the task realistically hard) and random skills
    other = rnd.choice([r for r in ROLES if r[0] != title])
    neighbours = [s for s in (other[2] + "," + other[3]).split(",") if s not in own]
    chosen += rnd.sample(neighbours, min(len(neighbours), rnd.randint(1, 3)))
    chosen += rnd.sample(sorted(ALL_SKILLS - own), 2) + ["communication", "teamwork"][: rnd.randint(0, 2)]
    chosen = list(dict.fromkeys(chosen))
    rnd.shuffle(chosen)
    shown = [alias_variant(s) for s in chosen]
    k = len(shown) // 2
    lines = ["SUMMARY", f"Motivated graduate with {rnd.choice(['no', 'six months of', '1 year of', '2 years of', '3 years of'])} experience.",
             "EDUCATION", rnd.choice(["B.Tech in Computer Science", "B.E. in Information Technology", "B.Sc in Computer Science", "M.Tech in Computer Science"]),
             "SKILLS", "Technical Skills: " + ", ".join(shown[:k]),
             "PROJECTS", "Project work: " + "Built an application using " + ", ".join(shown[k:k + 3]) + ".",
             "Implemented features with " + ", ".join(shown[k + 3:]) + "." if shown[k + 3:] else "Documented the work."]
    return "\n".join(lines)


def main():
    (DATA / "sample_resumes").mkdir(parents=True, exist_ok=True)
    (DATA / "eval").mkdir(parents=True, exist_ok=True)
    (DATA / "skills_taxonomy.json").write_text(json.dumps(TAXONOMY, indent=1))
    with open(DATA / "job_roles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "description", "required_skills", "preferred_skills", "min_education", "experience_level", "relevant_fields"])
        for r in ROLES:
            w.writerow([r[0], r[1], r[2].replace(",", ";"), r[3].replace(",", ";"), r[4], r[5], r[6]])
    for fn, text in JDS.items():
        (DATA / "sample_job_descriptions" / fn).write_text(text)
    gold = {}
    for c in CANDIDATES:
        write_resume_pdf(c, DATA / "sample_resumes" / f"{c['file']}.pdf")
        gold[c["file"] + ".pdf"] = skill_mentions(c["skills"] + c.get("also", []))
    (DATA / "sample_resumes" / "gold_skills.json").write_text(json.dumps(gold, indent=1))
    rows = [(synth_resume(role), role[0]) for role in ROLES for _ in range(12)]
    with open(DATA / "eval" / "synthetic_resumes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["resume_text", "true_role"]); w.writerows(rows)
    print(f"taxonomy: {len(ALL_SKILLS)} skills | roles: {len(ROLES)} | resumes: {len(CANDIDATES)} PDFs | eval rows: {len(rows)}")


if __name__ == "__main__":
    main()
