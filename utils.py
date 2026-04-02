"""
utils.py — Feature engineering for the internship matching model
All feature transformations live here so they're shared by
training (model.py) and prediction (main.py).
"""

import json
from typing import Dict, Any, List

# ── Individual feature functions ─────────────────────────────────────────────

def skill_overlap(user_skills: List[str], req_skills: List[str]) -> float:
    """Fraction of required skills the user possesses (0.0 – 1.0)."""
    if not req_skills:
        return 0.0
    return len(set(user_skills) & set(req_skills)) / len(req_skills)


def sector_match(user_sector: str, intern_sector: str) -> int:
    """Binary: 1 if sectors align, else 0."""
    return int(user_sector.lower() == intern_sector.lower())


def location_score(user_loc: str, intern_loc: str) -> float:
    """
    1.0  → same city
    0.9  → remote internship (flexible)
    0.3  → different city
    """
    if user_loc.lower() == intern_loc.lower():
        return 1.0
    if intern_loc.lower() == "remote":
        return 0.9
    return 0.3


def social_weight(category: str, district_type: str) -> float:
    """
    Composite equity score.
    Higher value → greater social support priority.
    """
    cat_map  = {"SC": 1.0, "ST": 1.0, "OBC": 0.8, "EWS": 0.7, "General": 0.5}
    dist_map = {"Rural": 1.0, "Semi-Urban": 0.7, "Urban": 0.4}
    c = cat_map.get(category, 0.5)
    d = dist_map.get(district_type, 0.5)
    return round((c + d) / 2, 4)


def education_level(education: str) -> int:
    """Ordinal encoding of education stage."""
    mapping = {
        "high school": 1,
        "diploma":     2,
        "bachelor's":  3,
        "master's":    4,
        "phd":         5,
    }
    return mapping.get(education.lower(), 3)


# ── Main feature builder ─────────────────────────────────────────────────────

FEATURE_NAMES = [
    "skill_overlap",
    "sector_match",
    "location_score",
    "social_weight",
    "prev_internship",
    "edu_level",
]


def build_feature_vector(user: Dict[str, Any], internship: Dict[str, Any]) -> Dict[str, float]:
    """
    Given a user profile dict and an internship dict, return
    a flat feature dict whose keys match FEATURE_NAMES.
    """
    u_skills = user.get("skills", [])
    if isinstance(u_skills, str):
        u_skills = json.loads(u_skills)

    r_skills = internship.get("required_skills", [])
    if isinstance(r_skills, str):
        r_skills = json.loads(r_skills)

    return {
        "skill_overlap":  skill_overlap(u_skills, r_skills),
        "sector_match":   sector_match(
                              user.get("sector_interest", ""),
                              internship.get("sector", internship.get("internship_sector", ""))
                          ),
        "location_score": location_score(
                              user.get("location", ""),
                              internship.get("location", internship.get("internship_location", ""))
                          ),
        "social_weight":  social_weight(
                              user.get("category", "General"),
                              user.get("district_type", "Urban")
                          ),
        "prev_internship": int(user.get("prev_internship", 0)),
        "edu_level":       education_level(user.get("education", "Bachelor's")),
    }


def feature_vector_list(user: Dict[str, Any], internship: Dict[str, Any]) -> List[float]:
    """Return features as an ordered list (for sklearn prediction)."""
    fv = build_feature_vector(user, internship)
    return [fv[k] for k in FEATURE_NAMES]


# ── Explainability helper ─────────────────────────────────────────────────────

FEATURE_LABELS = {
    "skill_overlap":   "Skill Match",
    "sector_match":    "Sector Alignment",
    "location_score":  "Location Proximity",
    "social_weight":   "Social Priority Score",
    "prev_internship": "Prior Experience",
    "edu_level":       "Education Level",
}


def explain_prediction(feature_vector: Dict[str, float],
                        model_coefs: Dict[str, float]) -> List[Dict]:
    """
    Returns top contributing factors for the prediction.
    Each item: {label, value, contribution, direction}
    """
    explanations = []
    for feat, val in feature_vector.items():
        coef = model_coefs.get(feat, 0)
        contribution = abs(val * coef)
        explanations.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        round(val, 3),
            "coef":         round(coef, 3),
            "contribution": round(contribution, 4),
            "direction":    "positive" if coef >= 0 else "negative",
        })
    explanations.sort(key=lambda x: x["contribution"], reverse=True)
    return explanations[:4]  # top 4 reasons
