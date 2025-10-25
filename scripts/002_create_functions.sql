-- Function to handle new user profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, full_name)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data ->> 'full_name', NULL)
  )
  ON CONFLICT (id) DO NOTHING;
  
  RETURN NEW;
END;
$$;

-- Function to update usage tracking
CREATE OR REPLACE FUNCTION public.update_usage_tracking(
  p_user_id UUID,
  p_message_count INTEGER DEFAULT 0,
  p_file_upload_count INTEGER DEFAULT 0
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  INSERT INTO public.usage_tracking (user_id, date, message_count, file_upload_count)
  VALUES (p_user_id, CURRENT_DATE, p_message_count, p_file_upload_count)
  ON CONFLICT (user_id, date)
  DO UPDATE SET
    message_count = usage_tracking.message_count + p_message_count,
    file_upload_count = usage_tracking.file_upload_count + p_file_upload_count,
    updated_at = NOW();
END;
$$;

-- Function to check daily usage limits
CREATE OR REPLACE FUNCTION public.check_usage_limit(
  p_user_id UUID,
  p_daily_message_limit INTEGER DEFAULT 50,
  p_daily_file_limit INTEGER DEFAULT 10
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  current_usage RECORD;
BEGIN
  SELECT 
    COALESCE(message_count, 0) as message_count,
    COALESCE(file_upload_count, 0) as file_upload_count
  INTO current_usage
  FROM public.usage_tracking
  WHERE user_id = p_user_id AND date = CURRENT_DATE;
  
  IF current_usage IS NULL THEN
    current_usage.message_count := 0;
    current_usage.file_upload_count := 0;
  END IF;
  
  RETURN jsonb_build_object(
    'message_count', current_usage.message_count,
    'file_upload_count', current_usage.file_upload_count,
    'can_send_message', current_usage.message_count < p_daily_message_limit,
    'can_upload_file', current_usage.file_upload_count < p_daily_file_limit,
    'message_limit', p_daily_message_limit,
    'file_limit', p_daily_file_limit
  );
END;
$$;

-- Create trigger for new user profile creation
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-- Function to update thread updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_thread_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE public.threads 
  SET updated_at = NOW() 
  WHERE id = NEW.thread_id;
  RETURN NEW;
END;
$$;

-- Create trigger to update thread timestamp when new message is added
CREATE TRIGGER update_thread_on_message
  AFTER INSERT ON public.messages
  FOR EACH ROW
  EXECUTE FUNCTION public.update_thread_timestamp();
