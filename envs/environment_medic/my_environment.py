import uuid
import random
from typing import Tuple, Dict, Any


from openenv.core.env_server import Environment
from .models import HospitalAction, HospitalObservation, HospitalState


disease_db = {
    "flu":     {"fatality_rate": 0.01, "resource_need": {"doctor": 1}},
    "covid":   {"fatality_rate": 0.10, "resource_need": {"doctor": 1, "bed": 1}},
    "cardiac": {"fatality_rate": 0.50, "resource_need": {"doctor": 2}},
    "stroke":  {"fatality_rate": 0.40, "resource_need": {"doctor": 2, "bed": 1}},
    "pneumonia": {"fatality_rate": 0.15, "resource_need": {"doctor": 1, "bed": 1}},
}

MAX_STEPS = 50
INITIAL_RESOURCES = {"doctor": 3, "bed": 2}


def generate_patient(patient_id: int) -> dict:
    disease = random.choice(list(disease_db.keys()))
    severity = round(random.uniform(0.2, 1.0), 2)
    return {
        "id": patient_id,
        "disease": disease,
        "severity": severity,
        "waiting_time": 0,
        "treated": False,
        "alive": True,
        "priority": None,
    }


class HospitalEnvironment(Environment):
    _global_state = None
    def __init__(self):
        super().__init__()
        if HospitalEnvironment._global_state is not None:
            self._state = HospitalEnvironment._global_state
        else:
            self._state = HospitalState(
                episode_id=str(uuid.uuid4()),
                current_time=0,
                step_count=0,
                patients=[],
                waiting_queue=[],
                treated_patients=[],
                available_resources={"doctor": 3, "bed": 2},
            )
        HospitalEnvironment._global_state = self._state
        

    def reset(self) -> HospitalObservation:
        print("RESET STATE ID:", id(self._state))
        num_patients = random.randint(5, 10)
        patients = [generate_patient(i) for i in range(num_patients)]
        self._state = HospitalState(          # ← this creates a NEW object
            episode_id=str(uuid.uuid4()),
            current_time=0,
            step_count=0,
            patients=patients,
            waiting_queue=[p["id"] for p in patients],
            treated_patients=[],
            available_resources=dict(INITIAL_RESOURCES),
        )
        HospitalEnvironment._global_state = self._state   
        # Remove all print statements
        return self._build_observation("Environment reset. New episode started.")

    def step(self, action: HospitalAction) -> HospitalObservation:
        state = self._state
        reward = 0.0
        info: Dict[str, Any] = {}

        if not state.patients:
            obs = self._build_observation("No active episode. Call /reset first.")
            return obs, reward, self._check_done(), {"error": "No active episode."}

        patient = next((p for p in state.patients if p["id"] == action.patient_id), None)
        if patient is None:
            msg = f"Invalid action: patient_id {action.patient_id} does not exist."
            return self._tick_and_build(msg)

        if patient["treated"]:
            msg = f"Patient {action.patient_id} is already treated."
            return self._tick_and_build(msg)

        if action.action_type == "assign_priority":
            if action.priority is None:
                msg = "assign_priority requires a priority value."
                return self._tick_and_build(msg)
            patient["priority"] = action.priority
            msg = f"Assigned priority {action.priority} to patient {action.patient_id}."

        elif action.action_type == "treat":
            needed = dict(disease_db.get(patient["disease"], {}).get("resource_need", {}))
            if action.resources:
                needed = action.resources

            for resource, amount in needed.items():
                if state.available_resources.get(resource, 0) < amount:
                    msg = (f"Insufficient {resource}: need {amount}, "
                        f"have {state.available_resources.get(resource, 0)}.")
                    return self._tick_and_build(msg)

            for resource, amount in needed.items():
                state.available_resources[resource] -= amount

            patient["treated"] = True
            state.waiting_queue = [i for i in state.waiting_queue if i != action.patient_id]
            state.treated_patients.append(action.patient_id)
            msg = f"Patient {action.patient_id} ({patient['disease']}) treated successfully."

        else:
            msg = f"Unknown action_type '{action.action_type}'."
            return self._tick_and_build(msg)

        return self._tick_and_build(msg)

    def _tick_and_build(self, message: str) -> HospitalObservation:
        state = self._state
        # Increment waiting time for all untreated patients
        for p in state.patients:
            if not p["treated"]:
                p["waiting_time"] += 1
        state.current_time += 1
        state.step_count += 1
        return self._build_observation(message)

    def _build_observation(self, message: str) -> HospitalObservation:
        state = self._state
        visible_patients = [
            {
                "id": p["id"],
                "disease": p["disease"],
                "severity": p["severity"],
                "waiting_time": p["waiting_time"],
                "treated": p["treated"],
                "priority": p["priority"],
            }
            for p in state.patients
        ]
        return HospitalObservation(
            current_time=state.current_time,
            patients=visible_patients,
            available_resources=dict(state.available_resources),
            message=message,
        )

    def _check_done(self) -> bool:
        state = self._state
        all_treated = all(p["treated"] for p in state.patients)
        max_reached = state.step_count >= MAX_STEPS
        return all_treated or max_reached

    @property
    def state(self) -> HospitalState:
        return self._state
