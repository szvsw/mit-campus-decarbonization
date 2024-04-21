from functools import cached_property
from typing import Optional

import boto3
from pydantic import DirectoryPath, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings


class SupaSettings(BaseSettings):
    url: str = Field("https://127.0.0.1:54321", env="URL")
    service_role_key: str = Field("123456", env="SERVICE_ROLE_KEY")
    anon_key: Optional[str] = Field("123456", env="ANON_KEY")
    host: Optional[str] = Field("127.0.0.1", env="HOST")
    port: Optional[int] = Field(54322, env="PORT")
    user: Optional[str] = Field("postgres", env="USER")
    database: Optional[str] = Field("postgres", env="DATABASE")
    password: Optional[str] = Field("postgres", env="PASSWORD")
    echo: Optional[bool] = Field(False, env="ECHO")

    @computed_field
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "SUPABASE_"
        env_file = ".env"


class AwsSettings(BaseSettings):
    access_key_id: SecretStr = Field("foobar", env="ACCESS_KEY_ID")
    secret_access_key: SecretStr = Field("foobar", env="SECRET_ACCESS_KEY")
    default_region: str = Field("us-east-1", env="DEFAULT_REGION")
    region: str = Field("us-east-1", env="REGION")
    account_id: SecretStr = Field("foobar", env="ACCOUNT_ID")
    bucket: SecretStr = Field(..., env="BUCKET")
    env: str = Field("dev", env="ENV")

    class Config:
        env_prefix = "AWS_"
        env_file = ".env"

    @cached_property
    def s3_client(self):
        return (
            boto3.client(
                "s3",
                aws_access_key_id=self.access_key_id.get_secret_value(),
                aws_secret_access_key=self.secret_access_key.get_secret_value(),
                region_name=self.region,
            )
            if self.env != "prod"
            else boto3.client("s3")
        )


class ConfigSettings(BaseSettings):
    log_level: str = Field("DEBUG", env="LOG_LEVEL")
    data_dir: DirectoryPath = Field("data", env="DATA_DIR")

    class Config:
        env_prefix = "CONFIG_"
        env_file = ".env"


supa_settings = SupaSettings()
aws_settings = AwsSettings()
config_settings = ConfigSettings()
