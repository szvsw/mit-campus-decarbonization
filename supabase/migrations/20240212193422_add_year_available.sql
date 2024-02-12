alter table "public"."DemandScenario" add column "year_available" smallint;

alter table "public"."PowerPlantScenario" add column "year_available" smallint;

alter table "public"."DemandScenario" add constraint "DemandScenario_year_available_check" CHECK ((year_available > 2023)) not valid;

alter table "public"."DemandScenario" validate constraint "DemandScenario_year_available_check";

alter table "public"."PowerPlantScenario" add constraint "PowerPlantScenario_year_available_check" CHECK ((year_available > 2023)) not valid;

alter table "public"."PowerPlantScenario" validate constraint "PowerPlantScenario_year_available_check";


