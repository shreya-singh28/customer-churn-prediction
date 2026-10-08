# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
import os
from dotenv import load_dotenv
load_dotenv()
# ----------------- CONFIG -----------------
DATA_PATH = r"Telcom-Customer-Churn.csv"
MODEL_PATH = "model.pkl"
ENC_PATH = "encoders.pkl"
CHURN_THRESHOLD = 0.40

# 🔐 ADMIN PASSWORD
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# ------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ----------------- UTILITIES -----------------

@st.cache_data
def load_dataset(path):

    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    return df


@st.cache_data
def load_model_encoders(mpath, encpath):

    model = joblib.load(mpath)
    encoders = joblib.load(encpath)

    return model, encoders


def safe_transform_single(encoders, col, value):

    le = encoders.get(col)

    if le is None:
        return value

    value = str(value)

    classes = list(le.classes_)

    if value in classes:

        return int(
            le.transform([value])[0]
        )

    else:

        return 0


# ----------------- FEATURE ENGINEERING -----------------

def feature_engineering(df):

    df["AvgMonthlySpend"] = (
        df["TotalCharges"] /
        (df["tenure"] + 1)
    )

    df["HighCharges"] = (
        df["MonthlyCharges"] > 80
    ).astype(int)

    df["LongTermCustomer"] = (
        df["tenure"] > 24
    ).astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1,12,24,48,72],
        labels=[0,1,2,3]
    )

    df["TenureGroup"] = (
        df["TenureGroup"]
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    df["ChargeRatio"] = (
        df["MonthlyCharges"] /
        (df["TotalCharges"] + 1)
    )

    return df


# ----------------- PREPARE INPUT -----------------

def prepare_input_df(encoders, data_dict):

    df = pd.DataFrame([data_dict]).copy()

    for col in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen"
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # FEATURE ENGINEERING
    df = feature_engineering(df)

    # ENCODING
    for col in df.columns:

        if col in encoders:

            df[col] = df[col].apply(
                lambda x: safe_transform_single(
                    encoders,
                    col,
                    x
                )
            )

    # FIX FEATURE ORDER
    df = df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    return df


# ----------------- PDF REPORT -----------------

def make_pdf_bytes(customer_info, prediction_text, prob):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=14)

    pdf.cell(
        200,
        10,
        txt="Customer Churn Prediction Report",
        ln=True,
        align='C'
    )

    pdf.ln(6)

    pdf.set_font("Arial", size=12)

    pdf.cell(
        200,
        8,
        txt=f"Prediction: {prediction_text}",
        ln=True
    )

    pdf.cell(
        200,
        8,
        txt=f"Probability (churn): {prob:.2f}",
        ln=True
    )

    pdf.ln(6)

    pdf.set_font("Arial", size=11)

    pdf.cell(
        200,
        6,
        txt="Customer details:",
        ln=True
    )

    pdf.ln(2)

    for k, v in customer_info.items():

        pdf.cell(
            200,
            6,
            txt=f"{k}: {v}",
            ln=True
        )

    return pdf.output(dest='S').encode('latin-1')


# ----------------- LOAD DATA & MODEL -----------------

df = load_dataset(DATA_PATH)

model, encoders = load_model_encoders(
    MODEL_PATH,
    ENC_PATH
)

accuracy = joblib.load("accuracy.pkl")

# ----------------- HEADER & ROLE -----------------

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    "A complete interactive dashboard with prediction, analytics, PDF report & bulk prediction."
)

role = st.selectbox(
    "Choose role",
    ["User", "Admin (full access)"]
)

# ----------------- LAYOUT -----------------

tabs = st.tabs([
    "Home",
    "Predict",
    "Insights",
    "Bulk Predict"
])

# ----------------- HOME -----------------

with tabs[0]:

    st.header("Welcome")

    st.markdown("""
    *How to use*:
    - Go to *Predict* to run a single-customer prediction and download a PDF report.
    - Go to *Bulk Predict* to upload a CSV and download predictions for many customers.
    - Admin role provides dataset download and deeper EDA in Insights.
    """)

# ----------------- PREDICT -----------------

with tabs[1]:

    st.header("🔮 Predict Single Customer")

    with st.form(
        "predict_form",
        clear_on_submit=False
    ):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "SeniorCitizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)",
                0,
                72,
                12
            )

        with col2:

            phone = st.selectbox(
                "PhoneService",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "MultipleLines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "InternetService",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "OnlineSecurity",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "OnlineBackup",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            device_protection = st.selectbox(
                "DeviceProtection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "TechSupport",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "StreamingTV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "StreamingMovies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                df["Contract"].unique()
            )

        col4, col5 = st.columns(2)

        with col4:

            paperless = st.selectbox(
                "PaperlessBilling",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "PaymentMethod",
                df["PaymentMethod"].unique()
            )

        with col5:

            monthly = st.number_input(
                "MonthlyCharges",
                min_value=0.0,
                value=70.0
            )

            total = st.number_input(
                "TotalCharges",
                min_value=0.0,
                value=1000.0
            )

        submitted = st.form_submit_button("Predict")

    if submitted:

        raw_input = {

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        input_df = prepare_input_df(
            encoders,
            raw_input
        )

        proba = model.predict_proba(
            input_df
        )[0][1]

        if proba < 0.3:

            risk = "Low Risk"

        elif proba < 0.6:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        st.subheader(f"Risk Level: {risk}")

        is_churn = proba > CHURN_THRESHOLD

        fig = go.Figure(go.Indicator(

            mode="gauge+number+delta",

            value=float(proba * 100),

            number={'suffix': "%"},

            domain={'x': [0, 1], 'y': [0, 1]},

            title={'text': "Churn Probability"},

            gauge={

                'axis': {'range': [0, 100]},

                'bar': {
                    'color': "crimson"
                    if is_churn
                    else "green"
                },

                'steps': [

                    {
                        'range': [
                            0,
                            CHURN_THRESHOLD * 100
                        ],
                        'color': "lightgreen"
                    },

                    {
                        'range': [
                            CHURN_THRESHOLD * 100,
                            100
                        ],
                        'color': "lightpink"
                    }
                ]
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if is_churn:

            st.error(
                f"⚠ Customer likely to churn "
                f"(probability = {proba:.2f})"
            )

        else:

            st.success(
                f"✅ Customer NOT likely to churn "
                f"(probability = {proba:.2f})"
            )

        st.subheader("💡 Suggested actions")

        suggestions = []

        if raw_input["tenure"] < 12:

            suggestions.append(
                "Offer a small discount on longer contracts."
            )

        if raw_input["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75):

            suggestions.append(
                "Offer a loyalty discount or bundle."
            )

        if raw_input["InternetService"] == "Fiber optic":

            suggestions.append(
                "Bundle premium support services."
            )

        if raw_input["TechSupport"] == "No":

            suggestions.append(
                "Offer free tech-support trial."
            )

        if not suggestions:

            suggestions = [
                "No immediate action suggested."
            ]

        for s in suggestions:

            st.write("- " + s)

        pdf_bytes = make_pdf_bytes(
            raw_input,
            "Churn" if is_churn else "No Churn",
            proba
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="churn_report.pdf",
            mime="application/pdf"
        )

# ----------------- INSIGHTS -----------------

with tabs[2]:

    st.header("📈 Insights & EDA")

    st.markdown("Key Performance Indicator")

    col1, col2, col3, col4 = st.columns(4)

    churn_rate = df["Churn"].map({
        "Yes":1,
        "No":0
    }).mean()

    col1.metric(
        "Total Customers",
        df.shape[0]
    )

    col2.metric(
        "Churn Customers",
        df[df["Churn"]=="Yes"].shape[0]
    )

    col3.metric(
        "Retention Rate",
        f"{100 - churn_rate*100:.2f}%"
    )

    col4.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    if role == "Admin (full access)":

        entered_pass = st.text_input(
            "Enter Admin Password",
            type="password"
        )

        if entered_pass != ADMIN_PASSWORD:

            st.error("Incorrect password! Access denied.")

            st.stop()

        st.subheader("Dataset quick stats")

        st.write(
            "Rows:",
            df.shape[0],
            "| Columns:",
            df.shape[1]
        )

        st.dataframe(df.head())

        st.metric(
            "Overall Churn Rate",
            f"{churn_rate*100:.2f}%"
        )

        st.subheader("Model Confusion Matrix")

        st.image("confusion_matrix.png")

        st.subheader("ROC Curve")

        st.image("roc_curve.png")

        #st.subheader("📈 Model Comparison")

        #st.image("model_comparison.png")

        st.subheader("Churn by Contract")

        fig1 = px.histogram(
            df,
            x="Contract",
            color="Churn",
            barmode="group",
            height=400
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        st.subheader(
            "MonthlyCharges distribution by Churn"
        )

        fig2 = px.violin(
            df,
            y="MonthlyCharges",
            x="Churn",
            box=True,
            points="all",
            height=400
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        #st.subheader("Churn by Internet Service")

        # fig = px.histogram(
          #  df,
           # x="InternetService",
          #  color="Churn"
        #)
       # st.plotly_chart(fig)

       # st.subheader("Churn by Payment Method")

      #  fig = px.histogram(
        #    df,
          #  x="PaymentMethod",
           # color="Churn"
        #)

        #st.plotly_chart(fig)

        st.subheader("Tenure vs Churn")

        fig = px.box(
            df,
            x="Churn",
            y="tenure"
        )

        st.plotly_chart(fig)

        st.subheader("📊 Feature Importance")

        try:

            fi = model.feature_importances_

            feat_names = model.feature_names_in_

            fi_df = pd.DataFrame({

                "feature": feat_names,
                "importance": fi

            }).sort_values(
                "importance",
                ascending=False
            ).head(20)

            fig3 = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Top Features"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        except Exception as e:

            st.write(
                "Could not render feature importance:",
                e
            )

        csv = df.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            "Download original dataset (CSV)",
            data=csv,
            file_name="telco_dataset.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Insights are available in Admin view."
        )

# ----------------- BULK PREDICT -----------------

with tabs[3]:

    st.header(
        "📥 Bulk CSV Upload + Batch Prediction"
    )

    st.markdown(
        "Upload a CSV with same columns as dataset."
    )

    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded is not None:

        batch_df = pd.read_csv(uploaded)

        st.write("Preview of uploaded data:")

        st.dataframe(batch_df)

        required_cols = [

            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]

        missing = [

            c for c in required_cols
            if c not in batch_df.columns
        ]

        if missing:

            st.warning(
                f"Missing columns: {missing}"
            )

        for c in required_cols:

            if c not in batch_df.columns:

                if c in [
                    "SeniorCitizen",
                    "tenure"
                ]:

                    batch_df[c] = 0

                elif c in [
                    "MonthlyCharges",
                    "TotalCharges"
                ]:

                    batch_df[c] = 0.0

                else:

                    batch_df[c] = "No"

        trans_rows = []

        for _, r in batch_df.iterrows():

            d = {

                c: r.get(c, None)
                for c in required_cols

            }

            transformed = prepare_input_df(
                encoders,
                d
            )

            trans_rows.append(
                transformed.iloc[0]
            )

        trans_df = pd.DataFrame(trans_rows)

        probs = model.predict_proba(
            trans_df
        )[:, 1]

        preds = (
            probs > CHURN_THRESHOLD
        ).astype(int)

        batch_df["ChurnProbability"] = probs

        batch_df["PredictedChurn"] = np.where(
            preds == 1,
            "Yes",
            "No"
        )

        st.subheader("Predictions Preview")

        st.dataframe(batch_df)

        csv_out = batch_df.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            "Download predictions CSV",
            data=csv_out,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

# ----------------- SIDEBAR -----------------

st.sidebar.markdown("---")

st.sidebar.write(
    "Model & encoders loaded. "
    "Threshold = " + str(CHURN_THRESHOLD)
)