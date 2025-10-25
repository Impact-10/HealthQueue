-- Add body metrics to patient_profiles
alter table public.patient_profiles
  add column if not exists height_cm numeric check (height_cm is null or (height_cm > 30 and height_cm < 300)),
  add column if not exists weight_kg numeric check (weight_kg is null or (weight_kg > 2 and weight_kg < 500));

-- Add BMI as a stored generated column if not present (Postgres 12+ supports generated columns)
do $$
begin
  if not exists (
    select 1 from information_schema.columns 
    where table_schema='public' and table_name='patient_profiles' and column_name='bmi'
  ) then
    execute 'alter table public.patient_profiles add column bmi numeric generated always as (case when height_cm is not null and weight_kg is not null and height_cm > 0 then round(weight_kg / ((height_cm/100)^2)::numeric, 1) end) stored';
  end if;
end;$$;

comment on column public.patient_profiles.height_cm is 'Height in centimeters';
comment on column public.patient_profiles.weight_kg is 'Weight in kilograms';
comment on column public.patient_profiles.bmi is 'Body Mass Index (auto-computed)';