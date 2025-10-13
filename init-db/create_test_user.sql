-- Create a new database for testing
CREATE DATABASE strader_test_db;

-- Create a new role for testing
CREATE ROLE test_user WITH LOGIN PASSWORD 'test_pw';

-- Grant connect access to the test database
GRANT CONNECT ON DATABASE strader_test_db TO test_user;

-- Grant all privileges on the test database to the test user
GRANT ALL PRIVILEGES ON DATABASE strader_test_db TO test_user;
