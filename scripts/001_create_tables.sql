-- Create profiles table for user management
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  date_of_birth DATE,
  gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
  height_cm INTEGER,
  weight_kg DECIMAL(5,2),
  activity_level TEXT CHECK (activity_level IN ('sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extremely_active')),
  medical_conditions TEXT[],
  medications TEXT[],
  allergies TEXT[],
  dietary_restrictions TEXT[],
  health_goals TEXT[],
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create threads table for conversation management
CREATE TABLE IF NOT EXISTS public.threads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create messages table for chat history
CREATE TABLE IF NOT EXISTS public.messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id UUID NOT NULL REFERENCES public.threads(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  file_urls TEXT[],
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create usage_tracking table for rate limiting
CREATE TABLE IF NOT EXISTS public.usage_tracking (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  date DATE NOT NULL DEFAULT CURRENT_DATE,
  message_count INTEGER DEFAULT 0,
  file_upload_count INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, date)
);

-- Create health_records table for storing health data
CREATE TABLE IF NOT EXISTS public.health_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  record_type TEXT NOT NULL CHECK (record_type IN ('vital_signs', 'lab_results', 'medication', 'symptom', 'appointment', 'other')),
  title TEXT NOT NULL,
  description TEXT,
  data JSONB,
  file_urls TEXT[],
  recorded_date DATE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Doctors catalog (basic). In real-world, would be managed separately/admin only
CREATE TABLE IF NOT EXISTS public.doctors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  full_name TEXT NOT NULL,
  specialty TEXT,
  bio TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Doctor threads: a consultation initiated by a user, optionally assigned to a doctor
CREATE TABLE IF NOT EXISTS public.doctor_threads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  doctor_id UUID REFERENCES public.doctors(id) ON DELETE SET NULL,
  source_thread_id UUID REFERENCES public.threads(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages between user and doctors (role can be 'user' or 'doctor')
CREATE TABLE IF NOT EXISTS public.doctor_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_thread_id UUID NOT NULL REFERENCES public.doctor_threads(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  doctor_id UUID REFERENCES public.doctors(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','doctor','system')),
  content TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.health_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doctors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doctor_threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doctor_messages ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for profiles table
CREATE POLICY "profiles_select_own" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "profiles_insert_own" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "profiles_update_own" ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "profiles_delete_own" ON public.profiles FOR DELETE USING (auth.uid() = id);

-- Create RLS policies for threads table
CREATE POLICY "threads_select_own" ON public.threads FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "threads_insert_own" ON public.threads FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "threads_update_own" ON public.threads FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "threads_delete_own" ON public.threads FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for messages table
CREATE POLICY "messages_select_own" ON public.messages FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "messages_insert_own" ON public.messages FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "messages_update_own" ON public.messages FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "messages_delete_own" ON public.messages FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for usage_tracking table
CREATE POLICY "usage_tracking_select_own" ON public.usage_tracking FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "usage_tracking_insert_own" ON public.usage_tracking FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "usage_tracking_update_own" ON public.usage_tracking FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "usage_tracking_delete_own" ON public.usage_tracking FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for health_records table
CREATE POLICY "health_records_select_own" ON public.health_records FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "health_records_insert_own" ON public.health_records FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "health_records_update_own" ON public.health_records FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "health_records_delete_own" ON public.health_records FOR DELETE USING (auth.uid() = user_id);

-- Doctors: read-only public listing (simplified) - allow all authenticated users to select
CREATE POLICY "doctors_select_all" ON public.doctors FOR SELECT USING (auth.role() = 'authenticated');

-- Doctor threads policies (user sees own, doctor sees assigned). For simplicity, allow only user ownership now.
CREATE POLICY "doctor_threads_select_own" ON public.doctor_threads FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "doctor_threads_insert_own" ON public.doctor_threads FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "doctor_threads_update_own" ON public.doctor_threads FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "doctor_threads_delete_own" ON public.doctor_threads FOR DELETE USING (auth.uid() = user_id);

-- Doctor messages: user can CRUD own; (future) doctor policies to be added when doctor auth implemented
CREATE POLICY "doctor_messages_select_own" ON public.doctor_messages FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "doctor_messages_insert_user" ON public.doctor_messages FOR INSERT WITH CHECK (auth.uid() = user_id);
