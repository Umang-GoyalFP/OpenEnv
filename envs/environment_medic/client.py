from openenv.core import EnvClient
from .models import HospitalAction, HospitalObservation, HospitalState


class HospitalEnv(EnvClient[HospitalAction, HospitalObservation, HospitalState]):

    def _step_payload(self, action: HospitalAction) -> dict:
        payload = {
            "action_type": action.action_type,
            "patient_id": action.patient_id,
        }
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.resources is not None:
            payload["resources"] = action.resources
        return payload

    def _parse_result(self, payload: dict):
        obs_data = payload["observation"]
        obs = HospitalObservation(
            current_time=obs_data["current_time"],
            patients=obs_data["patients"],
            available_resources=obs_data["available_resources"],
            message=obs_data["message"],
        )
        return {
            "observation":obs,
            "reward":payload.get("reward", 0.0),
            "done":payload.get("done", False),
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
        )
