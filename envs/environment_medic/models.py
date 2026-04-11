from dataclasses import dataclass, field
from typing import Optional, Dict, List
from openenv.core.env_server import Action, Observation, State


class HospitalAction(Action):
    # list of dicts: [{action_type, patient_id, priority?}, ...]
    actions: List[dict] = field(default_factory=list)


class HospitalObservation(Observation):
    current_time: int
    patients: List[dict]
    available_resources: Dict[str, int]
    message: str
    total_reward: float = 0.0


class HospitalState(State):
    episode_id: str = ""
    current_time: int = 0
    step_count: int = 0
    patients: List[dict] = field(default_factory=list)
    waiting_queue: List[int] = field(default_factory=list)
    treated_patients: List[int] = field(default_factory=list)
    available_resources: Dict[str, int] = field(default_factory=dict)
    active_treatments: List[dict] = field(default_factory=list)  # {patient_id, beds, treatment_end_step}
    total_reward: float = 0.0
    next_patient_id: int = 0