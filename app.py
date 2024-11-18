import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Option", ["Home", "Sales Data", "Predict Sales by Product Line", "Visualize Product Trends"])

# Home Page
if menu == "Home":
    st.title("Supermarket Sales Prediction System")
    st.write("""
    Welcome to the Supermarket Sales Prediction System! 
    Use this app to analyze historical sales, predict future sales by product line, and visualize trends.
    """)

# Sales Data Page
elif menu == "Sales Data":
    st.title("Sales Data Overview")
    # Load dataset
    sales_data = pd.read_csv("superSales 2.csv")
    st.write("### Dataset Sample")
    st.write(sales_data.head())

    # Display summary
    st.write("### Dataset Summary")
    st.write(sales_data.describe())

# Predict Sales by Product Line Page
elif menu == "Predict Sales by Product Line":
    st.title("Predict Sales by Product Line")
    st.write("Select a product line to predict sales for the next 3 months.")

    # Load dataset
    sales_data = pd.read_csv("superSales 2.csv")
    
    # Ensure Order_date is datetime and YearMonth is added
    sales_data['Order_date'] = pd.to_datetime(sales_data['Order_date'], errors='coerce')
    sales_data['YearMonth'] = sales_data['Order_date'].dt.to_period('M')
    
    # Select a product line
    product_line = st.selectbox("Choose Product Line", sales_data['Product_line'].unique())

    # Aggregate data for selected product line
    product_data = sales_data[sales_data['Product_line'] == product_line]
    monthly_sales = product_data.groupby('YearMonth').agg({'Total_price': 'sum'}).reset_index()
    monthly_sales.rename(columns={'YearMonth': 'ds', 'Total_price': 'y'}, inplace=True)
    monthly_sales['ds'] = monthly_sales['ds'].dt.to_timestamp()

    # Train Prophet model
    model = Prophet()
    model.fit(monthly_sales)

    # Predict sales for the next 3 months
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)

    # Display forecast
    st.write(f"### Sales Prediction for '{product_line}'")
    st.write(forecast[['ds', 'yhat']].tail(3))

    # Plot forecast
    fig = model.plot(forecast)
    st.pyplot(fig)

# Visualize Product Trends Page
elif menu == "Visualize Product Trends":
    st.title("Sales Trends by Product Line")
    st.write("### Historical Sales Data by Product Line")

    # Load dataset
    sales_data = pd.read_csv("superSales 2.csv")
    
    # Convert Order_date to datetime
    sales_data['Order_date'] = pd.to_datetime(sales_data['Order_date'], errors='coerce')
    sales_data['YearMonth'] = sales_data['Order_date'].dt.to_period('M')
    
    # Aggregate monthly sales by product line
    monthly_sales = sales_data.groupby(['YearMonth', 'Product_line']).agg({'Total_price': 'sum'}).reset_index()
    monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

    # Select product lines for visualization
    selected_lines = st.multiselect("Select Product Lines", sales_data['Product_line'].unique(), default=sales_data['Product_line'].unique()[:2])

    # Filter data for selected product lines
    filtered_sales = monthly_sales[monthly_sales['Product_line'].isin(selected_lines)]

    # Plot trends for selected product lines
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=filtered_sales, x='YearMonth', y='Total_price', hue='Product_line', marker='o', ax=ax)
    ax.set_title("Monthly Sales Trend by Product Line", fontsize=16)
    ax.set_xlabel("Year-Month", fontsize=14)
    ax.set_ylabel("Total Sales", fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)
