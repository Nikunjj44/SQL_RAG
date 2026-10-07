-- Create the role only if it doesn't exist yet
-- this role is created for our copilot or chatbot
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'copilot_ro') THEN
        CREATE ROLE copilot_ro LOGIN PASSWORD 'readonly';
    END IF;
END
$$;


-- GRanting only READ-ONLY role to bot
-- Such that it can only make SELECT commands, and no table is altered
GRANT CONNECT ON DATABASE olist TO copilot_ro;
GRANT USAGE ON SCHEMA public TO copilot_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO copilot_ro;

-- Any single query from this user is killed after 15 seconds
ALTER ROLE copilot_ro SET statement_timeout = '15s';