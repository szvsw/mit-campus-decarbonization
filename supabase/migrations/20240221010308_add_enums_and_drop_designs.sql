create type "public"."building_retrofit_level_type" as enum ('BASELINE', 'SHALLOW', 'DEEP');

create type "public"."building_schedules_type" as enum ('STANDARD', 'SETBACKS', 'ADVANCED');

create type "public"."climate_scenario_type" as enum ('BUSINESS_AS_USUAL', 'STABILIZED', 'RUNAWAY', 'IMPROVING');

revoke delete on table "public"."DesignVector" from "anon";

revoke insert on table "public"."DesignVector" from "anon";

revoke references on table "public"."DesignVector" from "anon";

revoke select on table "public"."DesignVector" from "anon";

revoke trigger on table "public"."DesignVector" from "anon";

revoke truncate on table "public"."DesignVector" from "anon";

revoke update on table "public"."DesignVector" from "anon";

revoke delete on table "public"."DesignVector" from "authenticated";

revoke insert on table "public"."DesignVector" from "authenticated";

revoke references on table "public"."DesignVector" from "authenticated";

revoke select on table "public"."DesignVector" from "authenticated";

revoke trigger on table "public"."DesignVector" from "authenticated";

revoke truncate on table "public"."DesignVector" from "authenticated";

revoke update on table "public"."DesignVector" from "authenticated";

revoke delete on table "public"."DesignVector" from "service_role";

revoke insert on table "public"."DesignVector" from "service_role";

revoke references on table "public"."DesignVector" from "service_role";

revoke select on table "public"."DesignVector" from "service_role";

revoke trigger on table "public"."DesignVector" from "service_role";

revoke truncate on table "public"."DesignVector" from "service_role";

revoke update on table "public"."DesignVector" from "service_role";

alter table "public"."DemandScenarioBuilding" drop constraint "DemandScenarioBuilding_design_vector_id_fkey";

alter table "public"."DesignVector" drop constraint "DesignVector_pkey";

drop index if exists "public"."DesignVector_pkey";

drop table "public"."DesignVector";

alter table "public"."DemandScenario" add column "building_retrofit_level" building_retrofit_level_type;

alter table "public"."DemandScenario" add column "building_schedules" building_schedules_type;

alter table "public"."DemandScenario" add column "climate_scenario" climate_scenario_type;

alter table "public"."DemandScenarioBuilding" drop column "design_vector_id";


