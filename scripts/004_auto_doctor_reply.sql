-- Auto doctor reply simulation trigger
-- This trigger watches for new user messages in doctor_messages and inserts
-- a heuristic doctor reply. Logic mirrors the API heuristic but moves into SQL.

create or replace function public.auto_doctor_reply()
returns trigger as $$
declare
  v_content text := lower(new.content);
  v_focus text[] := array[]::text[];
  v_redflags text[] := array[]::text[];
  v_questions text[] := array[]::text[];
  v_reply text;
begin
  -- Only act on user role
  if new.role <> 'user' then
    return new;
  end if;

  -- Focus tokens
  if v_content like '%pain%' then v_focus := array_append(v_focus,'pain'); end if;
  if v_content like '%cough%' then v_focus := array_append(v_focus,'cough'); end if;
  if v_content like '%fever%' then v_focus := array_append(v_focus,'fever'); end if;
  if v_content like '%headache%' then v_focus := array_append(v_focus,'headache'); end if;
  if v_content like '%fatigue%' then v_focus := array_append(v_focus,'fatigue'); end if;
  if v_content like '%chest%' then v_focus := array_append(v_focus,'chest'); end if;
  if v_content like '%sleep%' then v_focus := array_append(v_focus,'sleep'); end if;
  if v_content like '%rash%' then v_focus := array_append(v_focus,'rash'); end if;
  if v_content like '%dizz%' then v_focus := array_append(v_focus,'dizziness'); end if;

  -- Red flags
  if v_content like '%chest%' and (v_content like '%pain%' or v_content like '%tight%' or v_content like '%pressure%') then
    v_redflags := array_append(v_redflags,'possible chest pain');
  end if;
  if v_content like '%shortness of breath%' or v_content like '%breathless%' then
    v_redflags := array_append(v_redflags,'breathing difficulty');
  end if;

  -- Questions
  if 'pain' = any(v_focus) then v_questions := array_append(v_questions,'When did the pain start and what helps?'); end if;
  if 'fever' = any(v_focus) then v_questions := array_append(v_questions,'How high has your temperature been and for how long?'); end if;
  if 'cough' = any(v_focus) then v_questions := array_append(v_questions,'Is the cough dry or producing mucus?'); end if;
  if 'sleep' = any(v_focus) then v_questions := array_append(v_questions,'How many hours of quality sleep are you getting?'); end if;
  if array_length(v_questions,1) < 2 then v_questions := array_append(v_questions,'Have you noticed appetite or energy changes?'); end if;
  if array_length(v_questions,1) < 3 then v_questions := array_append(v_questions,'Are you taking any remedies for this already?'); end if;

  v_reply := case
    when array_length(v_focus,1) > 0 then 'Key points I noted: ' || array_to_string( (select array_agg(distinct f) from unnest(v_focus) f), ', ') || '.'
    else 'I am noting your recent symptoms.'
  end || E'\n\n' || array_to_string(v_questions, E'\n• ');

  if array_length(v_questions,1) > 0 then
    v_reply := regexp_replace(v_reply, E'\\n• ', E'\n• ', 'g'); -- ensure bullet formatting
    v_reply := v_reply || E'\n\nLet me know the answers to these and anything else you feel matters.';
  end if;

  if array_length(v_redflags,1) > 0 then
    v_reply := v_reply || E'\n\nIMPORTANT: Potential concern (' || array_to_string(v_redflags, '; ') || '). If severe or worsening, seek in-person urgent evaluation.';
  end if;

  insert into public.doctor_messages(doctor_thread_id, role, content)
  values (new.doctor_thread_id, 'doctor', v_reply);

  return new;
end;
$$ language plpgsql security definer;

create or replace trigger trg_auto_doctor_reply
after insert on public.doctor_messages
for each row execute function public.auto_doctor_reply();
