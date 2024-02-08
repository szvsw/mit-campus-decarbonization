from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class SupaSettings(BaseSettings):
    url: str = Field(..., env="URL")
    service_role_key: str = Field(..., env="SERVICE_ROLE_KEY")
    anon_key: Optional[str] = Field(..., env="ANON_KEY")
    host: Optional[str] = Field(..., env="HOST")
    port: Optional[int] = Field(..., env="PORT")
    user: Optional[str] = Field(..., env="USER")
    database: Optional[str] = Field(..., env="DATABASE")
    password: Optional[str] = Field(..., env="PASSWORD")
    echo: Optional[bool] = Field(False, env="ECHO")

    @computed_field
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "SUPABASE_"
        env_file = ".env"


supa_settings = SupaSettings()
