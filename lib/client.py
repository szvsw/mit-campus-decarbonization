import os

from supabase.client import ClientOptions

from lib import supa_settings
from supabase import create_client

client_options = ClientOptions(postgrest_client_timeout=60)
client = create_client(
    supabase_url=supa_settings.url,
    supabase_key=supa_settings.service_role_key,
    options=client_options,
)
