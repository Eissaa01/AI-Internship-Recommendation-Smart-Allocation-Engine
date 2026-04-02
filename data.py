"""
data.py — Synthetic dataset generation for internship matching
Generates 200 labeled (user, internship) pairs for ML training
"""

import random
import json
import sqlite3
import os
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)


EDUCATIONS = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]

SKILL_POOL = {
    "tech":    ["Python", "Machine Learning", "Data Analysis", "Web Dev", "SQL",
                "Cloud", "Cybersecurity", "IoT", "Android Dev", "UI/UX"],
    "finance": ["Accounting", "Financial Modeling", "Excel", "Taxation",
                "Risk Analysis", "Audit", "Equity Research", "Tally"],
    "health":  ["First Aid", "Patient Care", "Medical Research", "Pharmacology",
                "Lab Skills", "Public Health", "Nutrition"],
    "media":   ["Content Writing", "Video Editing", "Graphic Design",
                "Social Media", "Photography", "SEO", "Copywriting"],
    "govt":    ["Policy Analysis", "Report Writing", "Documentation",
                "GIS Mapping", "Survey Methods", "Community Outreach"],
    "legal":   ["Legal Research", "Contract Drafting", "Case Analysis",
                "Compliance", "IP Law", "Labour Law"],
}
ALL_SKILLS = [s for group in SKILL_POOL.values() for s in group]

SECTORS = list(SKILL_POOL.keys())

LOCATIONS = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
             "Hyderabad", "Pune", "Jaipur", "Lucknow", "Bhopal",
             "Remote"]

DISTRICT_TYPES = ["Urban", "Semi-Urban", "Rural"]

CATEGORIES = ["General", "OBC", "SC", "ST", "EWS"]


INTERNSHIP_CATALOGUE = []
for i in range(50):
    sector = random.choice(SECTORS)
    skills = random.sample(SKILL_POOL[sector], k=random.randint(2, 4))
    INTERNSHIP_CATALOGUE.append({
        "internship_id": f"INT-{i+1:03d}",
        "title": f"{sector.capitalize()} Intern #{i+1}",
        "sector": sector,
        "location": random.choice(LOCATIONS),
        "required_skills": skills,
        "stipend": random.randint(5000, 20000),
        "duration_weeks": random.choice([4, 6, 8, 12]),
    })


def skill_overlap(user_skills, req_skills):
    """Fraction of required skills the user possesses."""
    if not req_skills:
        return 0.0
    return len(set(user_skills) & set(req_skills)) / len(req_skills)

def sector_match(user_sector, intern_sector):
    return 1 if user_sector == intern_sector else 0

def location_score(user_loc, intern_loc):
    if user_loc == intern_loc:
        return 1.0
    if intern_loc == "Remote":
        return 0.9
    return 0.3

def social_weight(category, district_type):
    """Higher weight for underrepresented groups."""
    cat_w  = {"SC": 1.0, "ST": 1.0, "OBC": 0.8, "EWS": 0.7, "General": 0.5}
    dist_w = {"Rural": 1.0, "Semi-Urban": 0.7, "Urban": 0.4}
    return round((cat_w.get(category, 0.5) + dist_w.get(district_type, 0.5)) / 2, 3)

def education_level(edu):
    mapping = {"High School": 1, "Diploma": 2, "Bachelor's": 3, "Master's": 4, "PhD": 5}
    return mapping.get(edu, 3)

def make_features(user, internship):
    return {
        "skill_overlap":    skill_overlap(user["skills"], internship["required_skills"]),
        "sector_match":     sector_match(user["sector_interest"], internship["sector"]),
        "location_score":   location_score(user["location"], internship["location"]),
        "social_weight":    social_weight(user["category"], user["district_type"]),
        "prev_internship":  user["prev_internship"],
        "edu_level":        education_level(user["education"]),
    }


def generate_label(features):
    """
    Deterministic (yet noisy) rule used only for TRAINING data generation.
    The ML model learns to approximate this signal.
    """
    score = (
        0.40 * features["skill_overlap"] +
        0.25 * features["sector_match"] +
        0.15 * features["location_score"] +
        0.10 * features["social_weight"] +
        0.05 * features["prev_internship"] +
        0.05 * (features["edu_level"] / 5)
    )
    
    score += np.random.normal(0, 0.08)
    return int(score >= 0.45)


def build_dataset(n_users=200):
    rows = []
    for _ in range(n_users):
        sector = random.choice(SECTORS)
        user = {
            "education":       random.choice(EDUCATIONS),
            "skills":          random.sample(SKILL_POOL[sector], k=random.randint(2, 5)) +
                               random.sample(ALL_SKILLS, k=random.randint(0, 3)),
            "sector_interest": sector,
            "location":        random.choice(LOCATIONS),
            "category":        random.choice(CATEGORIES),
            "district_type":   random.choice(DISTRICT_TYPES),
            "prev_internship": random.randint(0, 1),
        }
        
        sample_interns = random.sample(INTERNSHIP_CATALOGUE, k=min(8, len(INTERNSHIP_CATALOGUE)))
        for intern in sample_interns:
            feats = make_features(user, intern)
            label = generate_label(feats)
            rows.append({
                
                "education":       user["education"],
                "skills":          json.dumps(user["skills"]),
                "sector_interest": user["sector_interest"],
                "location":        user["location"],
                "category":        user["category"],
                "district_type":   user["district_type"],
                "prev_internship": user["prev_internship"],
                
                "internship_id":       intern["internship_id"],
                "required_skills":     json.dumps(intern["required_skills"]),
                "internship_sector":   intern["sector"],
                "internship_location": intern["location"],
                
                **feats,
                
                "label": label,
            })
    return pd.DataFrame(rows)



def init_db(db_path="internships.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS internships (
            internship_id    TEXT PRIMARY KEY,
            title            TEXT,
            sector           TEXT,
            location         TEXT,
            required_skills  TEXT,
            stipend          INTEGER,
            duration_weeks   INTEGER,
            capacity         INTEGER DEFAULT 5
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            internship_id TEXT,
            user_name     TEXT,
            score         REAL,
            allocated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    for intern in INTERNSHIP_CATALOGUE:
        c.execute("""
            INSERT OR IGNORE INTO internships
              (internship_id, title, sector, location, required_skills, stipend, duration_weeks, capacity)
            VALUES (?,?,?,?,?,?,?,5)
        """, (
            intern["internship_id"], intern["title"], intern["sector"],
            intern["location"], json.dumps(intern["required_skills"]),
            intern["stipend"], intern["duration_weeks"]
        ))
    conn.commit()
    conn.close()
    return db_path


def get_internships(db_path="internships.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM internships WHERE capacity > 0", conn)
    conn.close()
    df["required_skills"] = df["required_skills"].apply(json.loads)
    return df.to_dict(orient="records")


def reduce_capacity(internship_id, user_name, score, db_path="internships.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE internships SET capacity = capacity - 1 WHERE internship_id = ? AND capacity > 0",
              (internship_id,))
    c.execute("INSERT INTO allocations (internship_id, user_name, score) VALUES (?,?,?)",
              (internship_id, user_name, score))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("📦 Generating dataset …")
    df = build_dataset(200)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/training_data.csv", index=False)
    print(f"   ✓ {len(df)} rows  |  {df['label'].mean():.1%} positive matches")

    print("🗄️  Initialising SQLite database …")
    init_db()
    print("   ✓ internships.db ready")
