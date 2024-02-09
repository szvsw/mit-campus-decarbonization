from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

FrontendEnvs = Literal["dev", "prod"]


class FrontendSettings(BaseSettings):

    env: str = Field("dev", env="ENV")
    password: str = Field(..., env="PASSWORD")

    class Config:
        env_prefix = "FRONTEND_"
        env_file = ".env"


frontend_settings = FrontendSettings()
