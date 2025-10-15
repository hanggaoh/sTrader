-- Create a new role for testing
CREATE ROLE test_user WITH LOGIN PASSWORD 'test_pw';

-- Create a new database for testing and set the owner to test_user.
-- As the owner, test_user will have all necessary permissions within this database.
CREATE DATABASE strader_test_db WITH OWNER = test_user;
