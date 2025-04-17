import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Dickinsonstile.com Flooring Data 2025", layout="wide")

# Title and description
st.title("Dickinsonstile.com Flooring Data 2025")
st.markdown("""
This application provides insights into flooring market data across various cities in Florida. 
Explore the dataset, visualize trends, and predict the number of current contractors using a machine learning model.
Data is sourced from a GitHub-hosted CSV file.
""")

# Load data from GitHub
@st.cache_data
def load_data():
    github_raw_url = "https://raw.githubusercontent.com/BurstSoftware/dickinsonstile-machine-learning-app-v1/main/data/flooring_data_2025.csv"
    try:
        data = pd.read_csv(github_raw_url)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "ML Prediction"])

    # Data Overview Page
    if page == "Data Overview":
        st.header("Data Overview")
        st.write("Explore the Dickinsonstile.com Flooring Data 2025 dataset with filtering options.")

        # Raw Data with Filtering
        st.subheader("Raw Data")
        st.write("Filter rows and columns to display specific data.")

        # Column selection
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns)
        
        # Row filtering
        st.write("Filter rows by specific criteria:")
        col1, col2 = st.columns(2)
        with col1:
            filter_city = st.multiselect("Filter by City", options=df["City"].unique(), default=[])
        with col2:
            filter_market_tier = st.multiselect("Filter by Market Tier", options=df["Market Tier"].unique(), default=[])
        
        # Numeric filters
        st.write("Filter by numeric ranges:")
        col3, col4 = st.columns(2)
        with col3:
            pop_min, pop_max = st.slider("Population range", 
                                         min_value=int(df["Population"].min()), 
                                         max_value=int(df["Population"].max()), 
                                         value=(int(df["Population"].min()), int(df["Population"].max())))
        with col4:
            home_sales_min, home_sales_max = st.slider("Annual Home Sales range", 
                                                       min_value=int(df["Annual Home Sales"].min()), 
                                                       max_value=int(df["Annual Home Sales"].max()), 
                                                       value=(int(df["Annual Home Sales"].min()), int(df["Annual Home Sales"].max())))
        
        # Apply filters
        filtered_df = df.copy()
        if filter_city:
            filtered_df = filtered_df[filtered_df["City"].isin(filter_city)]
        if filter_market_tier:
            filtered_df = filtered_df[filtered_df["Market Tier"].isin(filter_market_tier)]
        filtered_df = filtered_df[(filtered_df["Population"].between(pop_min, pop_max)) & 
                                 (filtered_df["Annual Home Sales"].between(home_sales_min, home_sales_max))]
        
        # Display filtered data
        if selected_columns:
            st.dataframe(filtered_df[selected_columns])
        else:
            st.warning("Please select at least one column to display.")

        # Basic Statistics with Filtering
        st.subheader("Basic Statistics")
        st.write("View statistics for selected columns and filtered rows.")

        if selected_columns:
            numeric_columns = filtered_df[selected_columns].select_dtypes(include=[np.number]).columns
            if numeric_columns.empty:
                st.warning("No numeric columns selected for statistics.")
            else:
                st.write(filtered_df[numeric_columns].describe())
        else:
            st.warning("Please select columns to view statistics.")

        # Missing Values with City Selection and Filtering
        st.subheader("Missing Values")
        st.write("Select a city to fill missing values and filter the missing values table.")

        # City selection for filling missing values
        selected_city = st.selectbox("Select a city to fill missing values", options=["None"] + df["City"].tolist())
        
        # Create a copy for missing value operations
        df_missing = df.copy()
        
        if selected_city != "None":
            # Get the selected city's row
            city_data = df[df["City"] == selected_city].iloc[0]
            # Fill missing values with city's values
            for col in df_missing.select_dtypes(include=[np.number]).columns:
                df_missing[col].fillna(city_data[col], inplace=True)
            for col in df_missing.select_dtypes(include=[object]).columns:
                df_missing[col].fillna(city_data[col], inplace=True)
        
        # Calculate missing values
        missing_values = df_missing.isnull().sum().reset_index()
        missing_values.columns = ["Column", "Missing Count"]
        
        # Filter missing values table
        st.write("Filter missing values table:")
        missing_columns = st.multiselect("Select columns for missing values", all_columns, default=all_columns)
        
        # Check for missing values to configure slider
        max_missing = int(missing_values["Missing Count"].max())
        filtered_missing = missing_values[missing_values["Column"].isin(missing_columns)]
        
        if max_missing > 0:
            missing_threshold = st.slider("Filter rows by minimum missing count", 0, max_missing, 0)
            filtered_missing = filtered_missing[filtered_missing["Missing Count"] >= missing_threshold]
            st.dataframe(filtered_missing)
        else:
            st.info("No missing values in the dataset.")
            st.dataframe(filtered_missing)

    # Visualizations Page
    elif page == "Visualizations":
        st.header("Visualizations")
        st.write("Visualize key metrics from the dataset.")
        
        st.subheader("Average Home Value by Market Tier")
        fig1 = px.bar(df, x="Market Tier", y="Avg. Home Value", color="Market Tier", 
                      title="Average Home Value by Market Tier")
        st.plotly_chart(fig1)
        
        st.subheader("Population vs. Annual Home Sales")
        fig2 = px.scatter(df, x="Population", y="Annual Home Sales", color="Market Tier", 
                          size="Avg. Home Value", hover_data=["City"], 
                          title="Population vs. Annual Home Sales")
        st.plotly_chart(fig2)
        
        st.subheader("Number of Current Contractors by City")
        fig3, ax = plt.subplots()
        sns.barplot(x="Number Of Current Contractor", y="City", hue="Market Tier", data=df, ax=ax)
        plt.title("Number of Current Contractors by City")
        st.pyplot(fig3)

    # ML Prediction Page
    elif page == "ML Prediction":
        st.header("Predict Number of Current Contractors")
        st.write("Use a Random Forest model to predict the number of current contractors based on input features.")
        
        df_ml = df.copy()
        le = LabelEncoder()
        df_ml["Market Tier"] = le.fit_transform(df_ml["Market Tier"])
        features = ["Population", "Number of Homes", "Market Tier", "Avg. Home Value", 
                    "Avg. Home Price", "Annual Home Sales"]
        X = df_ml[features]
        y = df_ml["Number Of Current Contractor"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        
        st.subheader("Make a Prediction")
        population = st.number_input("Population", min_value=0, value=50000)
        num_homes = st.number_input("Number of Homes", min_value=0, value=20000)
        market_tier = st.selectbox("Market Tier", df["Market Tier"].unique())
        avg_home_value = st.number_input("Average Home Value", min_value=0, value=300000)
        avg_home_price = st.number_input("Average Home Price", min_value=0, value=310000)
        annual_home_sales = st.number_input("Annual Home Sales", min_value=0, value=1000)
        
        market_tier_encoded = le.transform([market_tier])[0]
        
        input_data = np.array([[population, num_homes, market_tier_encoded, avg_home_value, 
                                avg_home_price, annual_home_sales]])
        prediction = model.predict(input_data)
        
        if st.button("Predict"):
            st.success(f"Predicted Number of Current Contractors: {int(prediction[0])}")
        
        st.subheader("Feature Importance")
        importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        fig4 = px.bar(importance, x="Feature", y="Importance", title="Feature Importance in Prediction")
        st.plotly_chart(fig4)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ by Streamlit | Data Source: [Dickinsonstile.com Flooring Data 2025 on GitHub](https://github.com/BurstSoftware/dickinsonstile-machine-learning-app-v1)")

else:
    st.error("Failed to load data. Please check the GitHub URL or try again later.")
