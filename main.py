from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uuid

# ----------------------------
# Load trained model & scaler
# ----------------------------
rf_model = joblib.load('stale_issue_model.pkl')
scaler = joblib.load('scaler.pkl')

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(
    title="Stale Issue Predictor API",
    description="Predicts whether a GitHub issue is stale and generates a nudge message",
    version="1.0"
)

# ----------------------------
# Request body model
# ----------------------------
class IssueFeatures(BaseModel):
    total_PRs: float
    merged_PRs: float
    merge_rate: float
    days_since_assigned: float
    issue_age_days: float
    num_comments: float
    assignee_past_commits: float
    assignee_current_issues: float
    issue_complexity: float
    repo_commit_freq: float
    days_since_last_comment: float
    is_bug: float
    assignee_response_rate: float

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/predict/")
def predict_issue(issue: IssueFeatures):
    # Convert input to numpy array
    x = np.array([[
        issue.total_PRs,
        issue.merged_PRs,
        issue.merge_rate,
        issue.days_since_assigned,
        issue.issue_age_days,
        issue.num_comments,
        issue.assignee_past_commits,
        issue.assignee_current_issues,
        issue.issue_complexity,
        issue.repo_commit_freq,
        issue.days_since_last_comment,
        issue.is_bug,
        issue.assignee_response_rate
    ]])

    # Scale features
    x_scaled = scaler.transform(x)

    # Predict
    pred = rf_model.predict(x_scaled)[0]
    prob = rf_model.predict_proba(x_scaled)[0, 1]

    # Generate nudge message
    if pred == 1:
        issue_id = str(uuid.uuid4())[:8]
        nudge_message = (
            f"Hi, it looks like issue #{issue_id} has been inactive for a while "
            f"(staleness probability: {prob:.2f}). Are you still working on this?"
        )
    else:
        nudge_message = f"Issue is predicted as active (staleness probability: {prob:.2f})."

    # Return response
    return {
        "stale_prediction": int(pred),
        "stale_probability": float(prob),
        "nudge_message": nudge_message
    }
