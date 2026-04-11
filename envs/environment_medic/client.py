from openenv.core import EnvClient
from .models import HospitalAction, HospitalObservation, HospitalState


class HospitalEnv(EnvClient[HospitalAction, HospitalObservation, HospitalState]):

    def _step_payload(self, action: HospitalAction) -> dict:
        return {"actions": action.actions}

    def _parse_result(self, payload: dict):
        obs_data = payload["observation"]
        obs = HospitalObservation(
            current_time=obs_data["current_time"],
            patients=obs_data["patients"],
            available_resources=obs_data["available_resources"],
            message=obs_data["message"],
            total_reward=obs_data.get("total_reward", 0.0),
        )
        return {
            "observation": obs,
            "reward": payload.get("reward", 0.0),
            "done": payload.get("done", False),
        }

    def _parse_state(self, payload: dict) -> HospitalState:
        return HospitalState(
            episode_id=payload.get("episode_id", ""),
            current_time=payload.get("current_time", 0),
            step_count=payload.get("step_count", 0),
            patients=payload.get("patients", []),
            waiting_queue=payload.get("waiting_queue", []),
            treated_patients=payload.get("treated_patients", []),
            available_resources=payload.get("available_resources", {}),
            active_treatments=payload.get("active_treatments", []),
            total_reward=payload.get("total_reward", 0.0),
            next_patient_id=payload.get("next_patient_id", 0),
        )