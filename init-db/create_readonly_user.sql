-- Revoke all privileges from the public role on the database
REVOKE ALL ON DATABASE strader_db FROM PUBLIC;

-- Create a new role for the trainer
CREATE ROLE trainer_user WITH LOGIN PASSWORD 'trainer_password';

-- Grant connect access to the database
GRANT CONNECT ON DATABASE strader_db TO trainer_user;

-- Revoke all privileges from the public role on the public schema
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- Grant usage on the public schema
GRANT USAGE ON SCHEMA public TO trainer_user;

-- Grant select on all existing tables in the public schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trainer_user;

-- Grant select on future tables in the public schema
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO trainer_user;
