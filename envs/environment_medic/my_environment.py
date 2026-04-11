import uuid
import random
import os
import math
import pandas as pd
from typing import Dict, Any

from openenv.core.env_server import Environment
from .models import HospitalAction, HospitalObservation, HospitalState


# ---------------------------------------------------------------------------
# Load disease data from CSV
# ---------------------------------------------------------------------------
_CSV_PATH = os.path.join(os.path.dirname(__file__), "disease_data.csv")

def _load_groups():
    df = pd.read_csv(_CSV_PATH)
    # Replace inf days_to_die with 999 (patient won't die within episode)
    df['days_to_die'] = df['days_to_die'].replace([float('inf')], 999.0)
    df['days_to_die'] = df['days_to_die'].fillna(999.0)
    # Scale beds: round to int, min 1, max 3
    df['beds_needed'] = df['beds_per_patient'].apply(lambda x: max(1, min(3, round(x))))
    low = df[df['fatality_rate'] < 0.1].to_dict('records')
    high = df[df['fatality_rate'] > 0.3].to_dict('records')
    return low, high

LOW_GROUP, HIGH_GROUP = _load_groups()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 150
INITIAL_BEDS = 20
INITIAL_PATIENT_COUNT = 8   # always even for 50/50
ARRIVAL_INTERVAL = 3
ARRIVALS_PER_INTERVAL = 3   # always odd → round down to 2 low + 1 high or 1+2


# ---------------------------------------------------------------------------
# Patient generation
# ---------------------------------------------------------------------------
def _make_patient(patient_id: int, row: dict, arrival_step: int) -> dict:
    return {
        "id": patient_id,
        "disease": row["disease"],
        "fatality_rate": round(row["fatality_rate"], 4),
        "avg_los": int(row["avg_los"]),
        "beds_needed": int(row["beds_needed"]),
        "days_to_die": int(row["days_to_die"]),
        "waiting_time": 0,
        "treated": False,
        "admitted": False,
        "alive": True,
        "priority": None,
        "arrival_step": arrival_step,
        "death_step": arrival_step + int(row["days_to_die"]),
        "treatment_end_step": None,
    }


def _generate_batch(start_id: int, count: int, arrival_step: int) -> list:
    """Generate count patients with ~50/50 low/high fatality split."""
    low_count = count // 2
    high_count = count - low_count
    patients = []
    for i in range(low_count):
        patients.append(_make_patient(start_id + i, random.choice(LOW_GROUP), arrival_step))
    for i in range(high_count):
        patients.append(_make_patient(start_id + low_count + i, random.choice(HIGH_GROUP), arrival_step))
    random.shuffle(patients)
    return patients


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class HospitalEnvironment(Environment):
    _global_state = None

    def __init__(self):
        super().__init__()
        if HospitalEnvironment._global_state is not None:
            self._state = HospitalEnvironment._global_state
        else:
            self._state = HospitalState(
                episode_id="",
                current_time=0,
                step_count=0,
                patients=[],
                waiting_queue=[],
                treated_patients=[],
                available_resources={"beds": INITIAL_BEDS},
                active_treatments=[],
                total_reward=0.0,
                next_patient_id=0,
            )

    # ------------------------------------------------------------------
    def reset(self) -> HospitalObservation:
        patients = _generate_batch(0, INITIAL_PATIENT_COUNT, 0)
        self._state = HospitalState(
            episode_id=str(uuid.uuid4()),
            current_time=0,
            step_count=0,
            patients=patients,
            waiting_queue=[p["id"] for p in patients],
            treated_patients=[],
            available_resources={"beds": INITIAL_BEDS},
            active_treatments=[],
            total_reward=0.0,
            next_patient_id=INITIAL_PATIENT_COUNT,
        )
        HospitalEnvironment._global_state = self._state
        return self._build_observation("Environment reset. New episode started.")

    # ------------------------------------------------------------------
    def step(self, action: HospitalAction) -> HospitalObservation:
        state = self._state

        if not state.patients:
            return self._build_observation("No active episode. Call /reset first.")

        messages = []
        step_reward = 0.0

        for act in action.actions:
            act_type = act.get("action_type")
            patient_id = act.get("patient_id")
            priority = act.get("priority")

            # --- validate patient ---
            patient = next((p for p in state.patients if p["id"] == patient_id), None)
            if patient is None:
                messages.append(f"Patient {patient_id} not found.")
                continue
            if not patient["alive"]:
                messages.append(f"Patient {patient_id} is dead.")
                continue
            if patient["treated"] or patient["admitted"]:
                messages.append(f"Patient {patient_id} already admitted/treated.")
                continue

            # --- handle action ---
            if act_type == "assign_priority":
                if priority is None:
                    messages.append("assign_priority requires priority value.")
                    continue
                patient["priority"] = priority
                messages.append(f"Priority {priority} → patient {patient_id}.")

            elif act_type == "treat":
                beds_needed = patient["beds_needed"]
                available = state.available_resources.get("beds", 0)
                if available < beds_needed:
                    messages.append(
                        f"Not enough beds for patient {patient_id} "
                        f"(need {beds_needed}, have {available})."
                    )
                    continue

                # allocate beds
                state.available_resources["beds"] -= beds_needed
                patient["admitted"] = True
                treatment_end = state.current_time + patient["avg_los"]
                patient["treatment_end_step"] = treatment_end
                # reset death clock — admitted patients are being managed
                patient["death_step"] = treatment_end + patient["avg_los"]

                state.active_treatments.append({
                    "patient_id": patient_id,
                    "beds": beds_needed,
                    "treatment_end_step": treatment_end,
                })
                if patient_id in state.waiting_queue:
                    state.waiting_queue.remove(patient_id)

                # reward: higher for high-fatality + treated early
                urgency = max(0.1, 1.0 - patient["waiting_time"] / max(patient["days_to_die"], 1))
                if patient["fatality_rate"] > 0.3:
                    r = round(20 * patient["fatality_rate"] * urgency, 2)
                else:
                    r = 5.0
                step_reward += r
                messages.append(
                    f"Patient {patient_id} ({patient['disease']}) admitted. +{r} reward."
                )

            else:
                messages.append(f"Unknown action_type '{act_type}'.")

        state.total_reward += step_reward
        msg = " | ".join(messages) if messages else "No valid actions taken."
        return self._tick_and_build(msg)

    # ------------------------------------------------------------------
    def _tick_and_build(self, message: str) -> HospitalObservation:
        state = self._state

        # 1. Free beds from completed treatments
        still_active = []
        for t in state.active_treatments:
            if state.current_time >= t["treatment_end_step"]:
                state.available_resources["beds"] += t["beds"]
                p = next((x for x in state.patients if x["id"] == t["patient_id"]), None)
                if p:
                    p["treated"] = True
                    if t["patient_id"] not in state.treated_patients:
                        state.treated_patients.append(t["patient_id"])
            else:
                still_active.append(t)
        state.active_treatments = still_active

        # 2. Tick waiting time + death check for unadmitted patients
        death_reward = 0.0
        for p in state.patients:
            if p["alive"] and not p["admitted"] and not p["treated"]:
                p["waiting_time"] += 1
                if state.current_time >= p["death_step"]:
                    p["alive"] = False
                    if p["id"] in state.waiting_queue:
                        state.waiting_queue.remove(p["id"])
                    death_reward -= 20
        state.total_reward += death_reward

        # 3. New arrivals every ARRIVAL_INTERVAL steps
        state.current_time += 1
        state.step_count += 1
        if state.current_time % ARRIVAL_INTERVAL == 0:
            new_patients = _generate_batch(
                state.next_patient_id, ARRIVALS_PER_INTERVAL, state.current_time
            )
            state.patients.extend(new_patients)
            state.waiting_queue.extend([p["id"] for p in new_patients])
            state.next_patient_id += ARRIVALS_PER_INTERVAL

        return self._build_observation(message)

    # ------------------------------------------------------------------
    def _build_observation(self, message: str) -> HospitalObservation:
        state = self._state
        visible = [
            {
                "id": p["id"],
                "disease": p["disease"],
                "fatality_rate": p["fatality_rate"],
                "beds_needed": p["beds_needed"],
                "avg_los": p["avg_los"],
                "days_to_die": p["days_to_die"],
                "waiting_time": p["waiting_time"],
                "treated": p["treated"],
                "admitted": p["admitted"],
                "alive": p["alive"],
                "priority": p["priority"],
                "treatment_end_step": p.get("treatment_end_step"),
            }
            for p in state.patients
        ]
        return HospitalObservation(
            current_time=state.current_time,
            patients=visible,
            available_resources=dict(state.available_resources),
            message=message,
            total_reward=state.total_reward,
        )

    # ------------------------------------------------------------------
    def _check_done(self) -> bool:
        state = self._state
        all_done = all(p["treated"] or not p["alive"] for p in state.patients)
        return all_done or state.step_count >= MAX_STEPS

    @property
    def state(self) -> HospitalState:
        return self._state