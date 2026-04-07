from openenv.core.env_server import create_fastapi_app
from .models import HospitalAction, HospitalObservation
from .my_environment import HospitalEnvironment

env = HospitalEnvironment
app = create_fastapi_app(env, HospitalAction, HospitalObservation)
