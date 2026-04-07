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
            print("STEP STATE ID AFTER:", id(self._state))
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

    def reset(self) -> HospitalObservation:
        print("RESET STATE ID:", id(self._state))
        num_patients = random.randint(5, 10)
        patients = [generate_patient(i) for i in range(num_patients)]
        self._state = HospitalState(
            episode_id=str(uuid.uuid4()),
            current_time=0,
            step_count=0,
            patients=patients,
            waiting_queue=[p["id"] for p in patients],
            treated_patients=[],
            available_resources=dict(INITIAL_RESOURCES),
        )
        return self._build_observation("Environment reset. New episode started.")

    def step(self, action: HospitalAction) -> Tuple[HospitalObservation, float, bool, Dict[str, Any]]:
        print("DEBUG ACTION:", action)
        print("CURRENT STATE:", self._state)
        print("GLOBAL STATE:", HospitalEnvironment._global_state)
        print("STEP STATE ID BEFORE:", id(self._state))

        state = self._state
        if not hasattr(self, "_state") or not self._state.patients:
            print("STATE EMPTY → RESETTING")
            return self.reset()
        reward = 0.0
        info: Dict[str, Any] = {}

        # --- Validate patient exists ---
        patient = next((p for p in state.patients if p["id"] == action.patient_id), None)
        if patient is None:
            msg = f"Invalid action: patient_id {action.patient_id} does not exist."
            obs = self._tick_and_build(msg)
            done = self._check_done()
            return obs

        if patient["treated"]:
            msg = f"Invalid action: patient {action.patient_id} is already treated."
            obs = self._tick_and_build(msg)
            done = self._check_done()
            return obs

        # --- Handle action types ---
        if action.action_type == "assign_priority":
            if action.priority is None:
                msg = f"assign_priority requires a priority value."
                obs = self._tick_and_build(msg)
                done = self._check_done()
                return obs
            patient["priority"] = action.priority
            msg = f"Assigned priority {action.priority} to patient {action.patient_id}."

        elif action.action_type == "treat":
            disease_info = disease_db.get(patient["disease"], {})
            needed = dict(disease_info.get("resource_need", {}))

            # Override with explicitly requested resources if provided
            if action.resources:
                needed = action.resources

            # Check resource availability
            for resource, amount in needed.items():
                available = state.available_resources.get(resource, 0)
                if available < amount:
                    msg = (
                        f"Insufficient resources to treat patient {action.patient_id}: "
                        f"need {amount} {resource}, have {available}."
                    )
                    obs = self._tick_and_build(msg)
                    done = self._check_done()
                    return obs

            # Allocate resources
            for resource, amount in needed.items():
                state.available_resources[resource] -= amount

            # Mark patient treated
            patient["treated"] = True
            if action.patient_id in state.waiting_queue:
                state.waiting_queue.remove(action.patient_id)
            state.treated_patients.append(action.patient_id)
            msg = f"Patient {action.patient_id} ({patient['disease']}) treated successfully."

        else:
            msg = f"Unknown action_type '{action.action_type}'."
            obs = self._tick_and_build(msg)
            done = self._check_done()
            return obs

        obs = self._tick_and_build(msg)
        done = self._check_done()
        return obs

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
