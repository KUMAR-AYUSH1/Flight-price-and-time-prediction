import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
st.sidebar.write(f"Model Evaluation and Error analysis of \n model1 XGBRegressor_MultiOutputRegressor_model \n model2 XGBRegressor_RegressorChain_model")
# Load model
model = joblib.load("XGBRegressor_MultiOutputRegressor_model.joblib")
model2 = joblib.load("XGBRegressor_RegressorChain_model.joblib")
ct = joblib.load("ct.joblib")

# Load test data
test = pd.read_csv("test.csv")
test.drop("date", axis=1, inplace=True)

x = test.drop(["total_duration_mins", "price"], axis=1)
y = test[["total_duration_mins", "price"]]

x_transformed = ct.transform(x)

predictions = model.predict(x_transformed)
predictions2 = model2.predict(x_transformed)

metric1 = {
    "R2 Score": r2_score(y, predictions),
    "Mean Absolute Error": mean_absolute_error(y, predictions),
    "Mean Squared Error": mean_squared_error(y, predictions),
    "Root Mean Squared Error": root_mean_squared_error(y, predictions),
}
metric2 = {
    "R2 Score": r2_score(y, predictions2),
    "Mean Absolute Error": mean_absolute_error(y, predictions2),
    "Mean Squared Error": mean_squared_error(y, predictions2),
    "Root Mean Squared Error": root_mean_squared_error(y, predictions2),
}
metric_df = pd.DataFrame([metric1])
metric_df2 = pd.DataFrame([metric2])
st.subheader("Model1 Performance")
st.dataframe(metric_df, use_container_width=True)

st.subheader("Model2 Performance")
st.dataframe(metric_df2, use_container_width=True)

pred_df = pd.DataFrame(
    predictions,
    columns=["predicted_total_duration_mins", "predicted_price"],
    index=test.index,
)

result_df = pd.concat([y, pred_df], axis=1)

result_df["duration_error"] = (
    result_df["predicted_total_duration_mins"]
    - result_df["total_duration_mins"]
).abs()

result_df["price_error"] = (
    result_df["predicted_price"]
    - result_df["price"]
).abs()

st.subheader("Model1 Error Analysis")
result_df.drop(["total_duration_mins", "price"], axis=1, inplace=True)
result_df= pd.concat([x, result_df], axis=1)

time_error_mean = result_df["duration_error"].mean()
price_error_mean = result_df["price_error"].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Time Duration Error in Mins",
        f"{result_df['duration_error'].mean():.2f}"
    )

with col2:
    st.metric(
        "Average Price Error",
        f"₹{result_df['price_error'].mean():.2f}"
    )


columns = ['airline', 'from', 'stop', 'to', 'class', 'time_of_day', 'is_weekday']
for i in columns:
    st.subheader(i)
    error=result_df.groupby(i)[['duration_error', 'price_error']].mean()
    st.bar_chart(error, use_container_width=True,x_label=i,y_label="Average Error")

st.divider()

st.subheader("Model2 error Analysis")
pred_df = pd.DataFrame(
    predictions2,
    columns=["predicted_total_duration_mins", "predicted_price"],
    index=test.index,
)

result_df = pd.concat([y, pred_df], axis=1)

result_df["duration_error"] = (
    result_df["predicted_total_duration_mins"]
    - result_df["total_duration_mins"]
).abs()

result_df["price_error"] = (
    result_df["predicted_price"]
    - result_df["price"]
).abs()

st.subheader("Actual vs Predicted")
result_df.drop(["total_duration_mins", "price"], axis=1, inplace=True)
result_df= pd.concat([x, result_df], axis=1)

time_error_mean = result_df["duration_error"].mean()
price_error_mean = result_df["price_error"].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Time Duration Error in Mins",
        f"{result_df['duration_error'].mean():.2f}"
    )

with col2:
    st.metric(
        "Average Price Error",
        f"₹{result_df['price_error'].mean():.2f}"
    )


columns = ['airline', 'from', 'stop', 'to', 'class', 'time_of_day', 'is_weekday']
for i in columns:
    st.subheader(i)
    error=result_df.groupby(i)[['duration_error', 'price_error']].mean()
    st.bar_chart(error, use_container_width=True,x_label=i,y_label="Average Error")

st.divider()

st.write("With this we can say error is higher in business class and lower in economy class in both models")

st.divider()

st.subheader("A/B testing of models using prediction_reviews.csv")

df = pd.read_csv("prediction_reviews.csv")
df = df[df['Rating']!=0]

df_model1 = df[df['Model']=='MultiOutputRegressor (Model 1)']
df_model2 = df[df['Model']=='RegressorChain (Model 2)']

avg_rating_model1 = df_model1['Rating'].mean()
avg_rating_model2 = df_model2['Rating'].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Rating of Model 1",
        f"{avg_rating_model1:.2f}"
    )

with col2:
    st.metric(
        "Average Rating of Model 2",
        f"{avg_rating_model2:.2f}"
    )