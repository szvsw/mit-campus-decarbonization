create type "public"."building_lab_strategy_type" as enum ('BASELINE', 'SHALLOW', 'DEEP');

alter table "public"."DemandScenario" add column "lab_strategy" building_lab_strategy_type;


