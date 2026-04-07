from dataclasses import dataclass, field
from typing import Optional, Dict, List
from openenv.core.env_server import Action, Observation, State



class HospitalAction(Action):
    action_type: str  # "assign_priority" or "treat"
    patient_id: int
    priority: Optional[int] = None
    resources: Optional[Dict[str, int]] = None



class HospitalObservation(Observation):
    current_time: int
    patients: List[dict]
    available_resources: Dict[str, int]
    message: str



class HospitalState(State):
    episode_id: str = ""
    current_time: int = 0
    step_count: int = 0
    patients: List[dict] = field(default_factory=list)
    waiting_queue: List[int] = field(default_factory=list)
    treated_patients: List[int] = field(default_factory=list)
    available_resources: Dict[str, int] = field(default_factory=dict)
