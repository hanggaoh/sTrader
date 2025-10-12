# sTrader

## Data source

## Model Selection

## Training, Validation, and BackTesting

## Evaluation Metrics


# Project management

1. Lock down data foundation:
    - [x] sentiment fetcher
    - [x] unit tests for news correctly insert into database
    - [x] news processing into sentiment per stock

2. Minimal data pipeline
    - [x] finbert pipeline -> process sentiment -> write score back to database
    - [] Fetch data from DB → preprocess → run model → store predictions back into DB.

3. Backend portfolio tracking Glue
    portfolio table
    manager that provides BUY and sell signal

4. Testing framework
    data ingestion
    model loop
    portfolio tracking
    CICD script

5. Improve model performance