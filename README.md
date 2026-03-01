# Airline Passenger Satisfaction Predictor 

This project builds and deploys a machine learning model that predicts airline passenger satisfaction based on travel experience features.

## Problem Statement
Airlines must understand customer satisfaction drivers to improve service quality and reduce churn. This project predicts whether a passenger is satisfied using service ratings and travel details.

## Dataset
Airline Passenger Satisfaction Dataset  
Source: Kaggle

## Machine Learning Approach
The machine learning solution is a **supervised classification model**:

- Logistic Regression (baseline)
- Random Forest (primary model)
- Gradient Boosting (performance improvement)

## Features Used
- Flight distance & delays
- WiFi, boarding, comfort & service ratings
- Customer type & travel class
- Online booking & gate convenience
- Cleanliness & inflight service

## Model Performance
- Validation Accuracy: **~96%**
- Test Accuracy: **~96%**

## Local Deployment
Run the Flask app:

```bash
python app/app.py

### Heroku Container Deployment

Login:
heroku container:login

Build:
docker build -f docker/Dockerfile -t registry.heroku.com/airline-passenger-satisfaction/web .

Push:
docker push registry.heroku.com/airline-passenger-satisfaction/web

Release:
heroku container:release web -a airline-passenger-satisfaction

Open:
heroku open -a airline-passenger-satisfaction
