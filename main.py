"""
main.py — FastAPI application for the Internship Recommendation Engine
Run: uvicorn main:app --reload --port 8000
"""

import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from data import get_internships, reduce_capacity, init_db
from model import recommend, load_model


import os
if not os.path.exists("internships.db"):
    init_db()


try: # model cache
    load_model()
    print("✅ ML model loaded into memory")
except FileNotFoundError:
    print("⚠️  Model not found. Run: python model.py  — then restart the server.")

app = FastAPI(
    title="AI Internship Recommendation Engine",
    description="ML-powered internship matching using Logistic Regression",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class UserProfile(BaseModel):
    name:            str                    = Field(default="Anonymous")
    education:       str                    = Field(default="Bachelor's",
                                               example="Bachelor's")
    skills:          List[str]              = Field(default=[],
                                               example=["Python", "Data Analysis"])
    sector_interest: str                    = Field(default="tech",
                                               example="tech")
    location:        str                    = Field(default="Delhi",
                                               example="Delhi")
    category:        str                    = Field(default="General",
                                               example="General")
    district_type:   str                    = Field(default="Urban",
                                               example="Urban")
    prev_internship: int                    = Field(default=0, ge=0, le=1)

class FeatureBreakdown(BaseModel):
    feature:      str
    label:        str
    value:        float
    contribution: float
    direction:    str

class RecommendationItem(BaseModel):
    internship_id:   str
    title:           str
    sector:          str
    location:        str
    required_skills: List[str]
    stipend:         int
    duration_weeks:  int
    capacity:        int
    score:           float
    score_pct:       float
    explanation:     List[FeatureBreakdown]

class RecommendationResponse(BaseModel):
    user_name:       str
    total_available: int
    recommendations: List[RecommendationItem]
    model_info:      dict

class AllocateRequest(BaseModel):
    internship_id: str
    user_name:     str
    score:         float




@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend."""
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}


@app.get("/internships")
async def list_internships(sector: Optional[str] = None, location: Optional[str] = None):
    """List all available internships (with optional filters)."""
    interns = get_internships()
    if sector:
        interns = [i for i in interns if i["sector"].lower() == sector.lower()]
    if location:
        interns = [i for i in interns
                   if i["location"].lower() == location.lower()
                   or i["location"].lower() == "remote"]
    return {"count": len(interns), "internships": interns}


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    profile: UserProfile,
    top_n: int = Query(default=5, ge=1, le=10, description="Number of recommendations"),
):
    """
    ML-powered internship recommendation.
    
    Accepts a user profile and returns top N internships ranked by
    predicted match probability, with feature-level explanations.
    """
    try:
        _, meta = load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Run: python model.py  and restart the server.",
        )

    internships = get_internships()
    if not internships:
        raise HTTPException(status_code=404, detail="No available internships in database.")

    user_dict = profile.dict()

    recs = recommend(user_dict, internships, top_n=top_n)

    return RecommendationResponse(
        user_name=profile.name,
        total_available=len(internships),
        recommendations=[RecommendationItem(**r) for r in recs],
        model_info={
            "type":     meta.get("model_type"),
            "accuracy": meta.get("accuracy"),
            "roc_auc":  meta.get("roc_auc"),
        },
    )


@app.post("/allocate")
async def allocate_internship(req: AllocateRequest):
    """
    Mark a user as allocated to an internship and reduce available capacity.
    """
    try:
        reduce_capacity(req.internship_id, req.user_name, req.score)
        return {"success": True, "message": f"Allocated {req.user_name} → {req.internship_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Return model metadata and feature importance."""
    try:
        _, meta = load_model()
        return meta
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")


@app.get("/sectors")
async def list_sectors():
    return {"sectors": ["tech", "finance", "health", "media", "govt", "legal"]}


@app.get("/locations")
async def list_locations():
    return {"locations": [
        "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Jaipur", "Lucknow", "Bhopal", "Remote"
    ]}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
