from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict

from .my_environment import HospitalEnvironment
from .models import HospitalAction

# One instance, lives for the entire process lifetime
_env = HospitalEnvironment()
app = FastAPI()


class ActionRequest(BaseModel):
    action_type: str
    patient_id: int
    priority: Optional[int] = None
    resources: Optional[Dict[str, int]] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = _env.reset()
    return {"observation": {
        "current_time": obs.current_time,
        "patients": obs.patients,
        "available_resources": obs.available_resources,
        "message": obs.message,
    }}


@app.post("/step")
def step(req: ActionRequest):
    action = HospitalAction(
        actions=[{
        "action_type": req.action_type,
        "patient_id": req.patient_id,
        "priority": req.priority,
        "resources": req.resources,
        }]
    )
    reward_before = _env.state.total_reward  # snapshot before
    obs = _env.step(action)
    step_reward = obs.total_reward - reward_before
    return {
        "observation": {
            "current_time": obs.current_time,
            "patients": obs.patients,
            "available_resources": obs.available_resources,
            "message": obs.message,
        },
        "step_reward": step_reward,
        "total_reward": obs.total_reward,
        "done": _env._check_done(),
        
    }


@app.get("/state")
def state():
    s = _env.state
    return {
        "episode_id": s.episode_id,
        "current_time": s.current_time,
        "step_count": s.step_count,
        "patients": s.patients,
        "waiting_queue": s.waiting_queue,
        "treated_patients": s.treated_patients,
        "available_resources": s.available_resources,
    }

