-- Patient health profile table
-- Includes basic demographic & clinical context for simulated doctor consultations
create table if not exists public.patient_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  age int check (age >= 0 and age < 130),
  gender text check (gender in ('male','female','non-binary','other','prefer-not-to-say')),
  medications text, -- comma or newline separated list
  conditions text,
  allergies text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create or replace function public.update_patient_profiles_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger patient_profiles_updated_at
before update on public.patient_profiles
for each row execute procedure public.update_patient_profiles_updated_at();

-- RLS (optional - enable and restrict to owner)
alter table public.patient_profiles enable row level security;

create policy "patient_profiles_select_own" on public.patient_profiles
for select using (auth.uid() = user_id);

create policy "patient_profiles_upsert_own" on public.patient_profiles
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
