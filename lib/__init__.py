from functools import cached_property
from typing import Optional

import boto3
from pydantic import DirectoryPath, Field, SecretStr, computed_field
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


class AwsSettings(BaseSettings):
    access_key_id: SecretStr = Field(..., env="ACCESS_KEY_ID")
    secret_access_key: SecretStr = Field(..., env="SECRET_ACCESS_KEY")
    default_region: str = Field(..., env="DEFAULT_REGION")
    region: str = Field(..., env="REGION")
    account_id: SecretStr = Field(..., env="ACCOUNT_ID")
    bucket: SecretStr = Field(..., env="BUCKET")

    class Config:
        env_prefix = "AWS_"
        env_file = ".env"

    @cached_property
    def s3_client(self):
        return boto3.client(
            "s3",
            aws_access_key_id=self.access_key_id.get_secret_value(),
            aws_secret_access_key=self.secret_access_key.get_secret_value(),
            region_name=self.region,
        )


class ConfigSettings(BaseSettings):
    log_level: str = Field(..., env="LOG_LEVEL")
    data_dir: DirectoryPath = Field("data", env="DATA_DIR")

    class Config:
        env_prefix = "CONFIG_"
        env_file = ".env"


supa_settings = SupaSettings()
aws_settings = AwsSettings()
config_settings = ConfigSettings()
