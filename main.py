import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Set up the app title and description
st.title("Car Price Prediction App")
st.write("This app predicts the price of cars based on various features.")

# Add a sidebar with car brand selection

st.sidebar.header("Select Car Brand")
car_brands = ["Audi", "BMW", "Mercedes-Benz C-Class", "Ford Focus", "Ford Motor Company", "Hyundai Motor Company", "Mercedes-Benz", "Skoda Auto", "Toyota", "Vauxhall", "Volkswagen"]
selected_brand = st.sidebar.selectbox("Choose a Car Brand", car_brands)

if selected_brand == "BMW":
    st.title("BMW Car Overview")
    st.write("""
    - Founded : Bayerische Motoren Werke AG (BMW) was founded in 1916 in Munich, Germany.

    - Type of Company : Premium automobile manufacturer specializing in luxury vehicles, motorcycles, and bicycles.

    - Brand Philosophy : The brand emphasizes performance, engineering excellence, and driving pleasure, encapsulated in their slogan, "The Ultimate Driving Machine."

    - Product Range : Offers a wide range of vehicles including sedans (3 Series, 5 Series, 7 Series), SUVs (X1, X3, X5, X7), and sports cars (Z4, M Series).

    - Electric Vehicles : BMW is investing heavily in electric mobility with its BMW i series, including models like the i3 and iX.

    - Innovative Technology : Known for incorporating advanced technology in vehicles, including the latest infotainment systems (iDrive), driver-assistance features, and performance enhancements.

    - Motorsport Heritage : Has a rich motorsport history, particularly in touring car racing, Formula E, and the prestigious 24 Hours of Le Mans.

    - Global Presence : BMW operates production facilities worldwide, including in Germany, the U.S., and China, catering to a global market.

    - Sustainability Efforts : Focuses on sustainability with initiatives for reduced emissions, increased use of recycled materials, and promoting electric mobility.
    """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/bmw.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("BMW Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Audi":
    st.title("AUDI Car Overview")
    st.markdown("""
    - **Founded**: Audi AG was founded in 1909 in Zwickau, Germany.
                
    - **Type of Company**: Premium automobile manufacturer known for luxury vehicles, part of the Volkswagen Group.
                
    - **Brand Philosophy**: Focuses on technology, performance, and sophisticated design, encapsulated in their slogan, "Vorsprung durch Technik" (Advancement through Technology).
                
    - **Product Range**: Offers a diverse lineup including sedans (A3, A4, A6, A8), SUVs (Q2, Q3, Q5, Q7, Q8), and performance models (S and RS series).
                
    - **Electric Vehicles**: Committed to electric mobility with its e-tron series, including fully electric and hybrid models.
                
    - **Innovative Technology**: Renowned for incorporating cutting-edge technology such as Quattro all-wheel drive and advanced driver-assistance systems.
                
    - **Motorsport Heritage**: Has a strong motorsport background with notable successes in rallying and endurance racing, particularly the 24 Hours of Le Mans.
                
    - **Global Presence**: Operates production facilities in multiple countries, including Germany, Hungary, and Mexico, serving a global market.
                
    - **Sustainability Efforts**: Focuses on reducing environmental impact through sustainable production practices and developing electric vehicles.
    """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/audi.csv")
        return df

    df = load_data()    
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)

    # Prediction
    st.title("AUDI Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Mercedes-Benz C-Class":
    st.title("Mercedes-Benz C-Class Car Overview")
    st.markdown("""
    - **Founded**: The Mercedes-Benz C-Class was first introduced in 1993 by Mercedes-Benz, part of the Daimler AG group.

    - **Type of Car**: A compact executive car offering luxury features, known for its balanced combination of comfort, technology, and performance.

    - **Brand Philosophy**: Represents Mercedes-Benz's commitment to luxury, safety, and advanced engineering with a focus on providing a premium driving experience.

    - **Product Range**: The C-Class lineup includes sedan, coupe, and cabriolet models, with various trims ranging from the standard versions to high-performance AMG variants.

    - **Electric Vehicles**: The C-Class does not have a fully electric model yet, but there are hybrid variants (C-Class Plug-in Hybrid) that focus on reducing emissions.

    - **Innovative Technology**: Equipped with cutting-edge technology such as the MBUX infotainment system, autonomous driving features, and a wide array of safety systems like Active Brake Assist and Attention Assist.

    - **Motorsport Heritage**: While not as race-oriented as the AMG GT series, high-performance AMG models of the C-Class have been successful in motorsport competitions, particularly in touring car championships.

    - **Global Presence**: The C-Class is manufactured in multiple locations worldwide, including Germany, South Africa, and the United States, making it one of Mercedes-Benz’s most popular and widely sold models globally.

    - **Sustainability Efforts**: Mercedes-Benz has a strong focus on reducing carbon emissions and aims to offer a fully electric model in most of its vehicle segments, including the C-Class in the future.
    """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/cclass.csv")
        return df

    df = load_data()    
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)

    # Prediction
    st.title("Mercedes-Benz C-Class Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Ford Focus":
    st.title("Ford Focus Car Overview")
    st.markdown("""
    - **Founded**: The Ford Focus was first introduced in 1998 by Ford Motor Company as a replacement for the Ford Escort.

    - **Type of Car**: A compact car available in various body styles including sedan, hatchback, and wagon, offering a balance between affordability, performance, and technology.

    - **Brand Philosophy**: The Focus emphasizes affordability, fuel efficiency, and advanced technology, making it a popular choice among compact car buyers worldwide.

    - **Product Range**: The Focus comes in multiple variants, including the standard models, fuel-efficient EcoBoost versions, and high-performance models like the Focus ST and Focus RS.

    - **Electric Vehicles**: The Ford Focus Electric, introduced in 2011, was one of Ford's early electric vehicle offerings, although it has since been discontinued to focus on newer EV models.

    - **Innovative Technology**: Known for its SYNC infotainment system, driver-assist features like adaptive cruise control, lane-keeping assist, and enhanced fuel efficiency technologies such as EcoBoost engines.

    - **Motorsport Heritage**: The Focus has been a strong contender in rally racing, especially with its high-performance Focus RS model, which has seen success in World Rally Championship (WRC) events.

    - **Global Presence**: The Focus is sold globally and manufactured in several countries, including the United States, Germany, and China, making it one of Ford’s most widely recognized models.

    - **Sustainability Efforts**: Ford has focused on improving fuel efficiency and reducing emissions through its EcoBoost engine technology and plans to shift towards electric vehicles across its lineup in the near future.
    """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/focus.csv")
        return df

    df = load_data()    
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)

    # Prediction
    st.title("Ford Focus Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Ford Motor Company":
    st.title("Ford Motor Company Car Overview")
    st.markdown("""
    - **Founded**: Ford Motor Company was founded by Henry Ford on June 16, 1903, in Detroit, Michigan, USA.

    - **Type of Company**: A multinational automobile manufacturer producing a wide range of vehicles, including sedans, trucks, SUVs, and electric vehicles. Ford also owns the luxury brand Lincoln.

    - **Brand Philosophy**: Ford emphasizes innovation, affordability, and reliability, with a focus on making vehicles that are accessible to a broad range of consumers. Their slogan, "Built Ford Tough," reflects their emphasis on durability and performance.

    - **Product Range**: Ford’s lineup includes popular models like the Mustang, F-150, Explorer, Escape, and new electric vehicles like the Mustang Mach-E and F-150 Lightning.

    - **Electric Vehicles**: Ford is investing heavily in electric mobility with its new lineup of electric vehicles under the "Ford EV" initiative, including the Mustang Mach-E and the F-150 Lightning, as part of their broader shift toward sustainability.

    - **Innovative Technology**: Known for innovations such as the SYNC infotainment system, Ford Co-Pilot360 driver-assist technologies, and advancements in hybrid and electric vehicle technology.

    - **Motorsport Heritage**: Ford has a strong motorsport presence, especially in rally racing, NASCAR, and endurance racing. Iconic models like the Ford GT have a storied history in events like the 24 Hours of Le Mans.

    - **Global Presence**: Ford operates manufacturing plants and sells vehicles globally, with a significant presence in markets across North America, Europe, Asia, and South America.

    - **Sustainability Efforts**: Ford is committed to reducing its environmental impact, with ambitious plans to electrify its vehicle lineup, achieve carbon neutrality by 2050, and adopt sustainable manufacturing practices.
    """)

    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/ford.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Ford Motor Company Car Predictions")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Hyundai Motor Company":
    st.title("Hyundai Motor Company Car Overview")
    st.markdown("""
    - **Founded**: Hyundai Motor Company was founded on December 29, 1967, in Seoul, South Korea, by Chung Ju-Yung.

    - **Type of Company**: A multinational automotive manufacturer producing a wide range of vehicles, including sedans, SUVs, trucks, and electric vehicles. Hyundai also owns the luxury brand Genesis.

    - **Brand Philosophy**: Hyundai emphasizes value for money, reliability, and innovation, with a strong focus on delivering cutting-edge technology at competitive prices. Their slogan, "New Thinking. New Possibilities," reflects their drive for innovation.

    - **Product Range**: Hyundai offers a variety of vehicles, including popular models such as the Elantra, Sonata, Tucson, Santa Fe, and the fully electric Ioniq series.

    - **Electric Vehicles**: Hyundai is expanding its electric vehicle lineup under the Ioniq brand, with models like the Ioniq 5 and Ioniq 6, positioning itself as a leader in the EV market with a focus on sustainability.

    - **Innovative Technology**: Hyundai is known for incorporating advanced technology in its vehicles, such as Hyundai SmartSense (a suite of driver assistance features), infotainment systems, and hydrogen fuel cell technology in the Hyundai Nexo.

    - **Motorsport Heritage**: Hyundai has been active in motorsports, particularly in the World Rally Championship (WRC), with competitive performance from models like the Hyundai i20 WRC.

    - **Global Presence**: Hyundai operates production plants and sells vehicles globally, with a strong presence in markets like South Korea, the United States, Europe, and India. It is one of the largest car manufacturers in the world.

    - **Sustainability Efforts**: Hyundai is committed to sustainability, with a focus on expanding its electric vehicle offerings, reducing emissions, and investing in hydrogen fuel cell technology to promote a cleaner future.
    """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/hyundi.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Hyundai Motor Company Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Mercedes-Benz":
    st.title("Mercedes-Benz Car Overview")

    st.markdown("""
        - **Founded**: Mercedes-Benz was founded in 1926 through the merger of Karl Benz's and Gottlieb Daimler's companies, with origins dating back to the invention of the first automobile in 1886.

        - **Type of Company**: A German luxury automobile manufacturer producing a wide range of vehicles, including luxury sedans, SUVs, trucks, and buses. Mercedes-Benz is part of Daimler AG.

        - **Brand Philosophy**: Mercedes-Benz focuses on luxury, innovation, and performance, with the slogan "The Best or Nothing," reflecting its commitment to producing high-quality, premium vehicles.

        - **Product Range**: Mercedes-Benz offers a diverse range of vehicles, including popular models like the C-Class, E-Class, S-Class, GLE, G-Class, and the high-performance AMG line. They also produce electric vehicles under the EQ brand.

        - **Electric Vehicles**: Mercedes-Benz is advancing in the electric vehicle market with its EQ series, including the EQS, EQC, and EQA, as part of its strategy to transition to electric mobility and reduce emissions.

        - **Innovative Technology**: Known for cutting-edge technology, Mercedes-Benz integrates advanced features such as MBUX (Mercedes-Benz User Experience) infotainment, autonomous driving capabilities, and numerous driver-assistance systems.

        - **Motorsport Heritage**: Mercedes-Benz has a rich motorsport history, particularly in Formula 1, where it has dominated in recent years. The high-performance AMG models also have a strong racing pedigree.

        - **Global Presence**: Mercedes-Benz has a global footprint with manufacturing plants and sales in countries worldwide, making it one of the most recognized and sought-after luxury car brands globally.

        - **Sustainability Efforts**: Mercedes-Benz is committed to sustainability through its "Ambition 2039" initiative, aiming for carbon neutrality in its vehicles and production processes by 2039, with a focus on electric vehicles and renewable energy.
        """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/merc.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Mercedes-Benz Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Skoda Auto":
    st.title("Škoda Auto Overview")

    st.markdown("""
        - **Founded**: Škoda Auto was founded in 1895 in Mladá Boleslav, Czech Republic, initially as a bicycle manufacturing company by Václav Laurin and Václav Klement.

        - **Type of Company**: Škoda is a Czech automobile manufacturer that produces a range of affordable, practical, and reliable cars. It is a subsidiary of the Volkswagen Group.

        - **Brand Philosophy**: Škoda is known for offering value-for-money vehicles that combine reliability, practicality, and advanced technology with the slogan "Simply Clever," emphasizing smart design and innovation.

        - **Product Range**: Škoda's lineup includes popular models like the Octavia, Superb, Kodiaq, Karoq, and Fabia, offering everything from compact cars to family SUVs.

        - **Electric Vehicles**: Škoda is moving into electric mobility with its all-electric model, the Škoda Enyaq iV, as part of the Volkswagen Group’s broader push toward electrification.

        - **Innovative Technology**: Known for practical and user-friendly technology, Škoda integrates features like advanced infotainment systems, smart connectivity options, and driver-assistance systems such as adaptive cruise control and lane assist.

        - **Motorsport Heritage**: Škoda has a strong presence in motorsports, particularly in rallying, where models like the Škoda Fabia R5 have seen significant success in international rally competitions.

        - **Global Presence**: Škoda sells vehicles in over 100 countries and has a strong presence in Europe, Asia, and emerging markets. It operates several production plants worldwide.

        - **Sustainability Efforts**: Škoda is committed to reducing its environmental impact by developing electric vehicles, improving fuel efficiency, and focusing on sustainable manufacturing practices under the Volkswagen Group's sustainability initiatives.
        """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/skoda.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Skoda Auto Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Toyota":
    st.title("Toyota Auto Car Overview")

    st.markdown("""
        - **Founded**: Toyota Motor Corporation was founded on August 28, 1937, by Kiichiro Toyoda in Japan.

        - **Type of Company**: A multinational automobile manufacturer that produces a wide range of vehicles, including sedans, SUVs, trucks, hybrids, and electric vehicles. Toyota is one of the largest automakers in the world.

        - **Brand Philosophy**: Toyota emphasizes reliability, innovation, and environmental sustainability, with a focus on producing high-quality, durable vehicles. Their slogan, "Let's Go Places," reflects their commitment to mobility and adventure.

        - **Product Range**: Toyota's lineup includes popular models like the Corolla, Camry, RAV4, Land Cruiser, and the hybrid/electric Prius series. They also offer a luxury line under the Lexus brand.

        - **Electric Vehicles**: Toyota is a pioneer in hybrid technology with the Prius and has expanded into electric vehicles with models like the bZ4X under its bZ (Beyond Zero) brand. They continue to focus on hydrogen fuel cell technology with the Toyota Mirai.

        - **Innovative Technology**: Toyota is known for its advancements in hybrid technology, fuel efficiency, and safety features such as Toyota Safety Sense (TSS), a suite of driver-assistance technologies, and the development of autonomous driving.

        - **Motorsport Heritage**: Toyota has a strong presence in motorsports, particularly in the World Rally Championship (WRC) and endurance racing, including the 24 Hours of Le Mans. Toyota's Gazoo Racing division also develops high-performance models.

        - **Global Presence**: Toyota operates manufacturing plants and sells vehicles globally, with a significant presence in markets like North America, Asia, Europe, and the Middle East, making it a truly global brand.

        - **Sustainability Efforts**: Toyota is committed to sustainability with a focus on developing hybrid, electric, and hydrogen vehicles. They aim to achieve carbon neutrality by 2050 through sustainable manufacturing practices and eco-friendly vehicles.
        """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/vauxhall.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Toyota Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Vauxhall":
    st.title("Vauxhall Car Overview")

    st.markdown("""
        - **Founded**: Vauxhall Motors was founded in 1857 by Alexander Wilson in Vauxhall, London, originally as a pump and marine engine manufacturer. It transitioned to automotive manufacturing in 1903.

        - **Type of Company**: Vauxhall is a British automotive manufacturer producing a wide range of vehicles, including compact cars, SUVs, and commercial vehicles. It is a subsidiary of Stellantis, formerly part of General Motors.

        - **Brand Philosophy**: Vauxhall focuses on providing affordable, practical, and reliable vehicles with a blend of modern technology and design. The slogan "Isn't Life Brilliant" reflects their approach to everyday practicality.

        - **Product Range**: Vauxhall offers popular models such as the Corsa, Astra, Crossland, Mokka, and Vivaro, catering to both private and commercial buyers with compact cars, SUVs, and vans.

        - **Electric Vehicles**: Vauxhall is moving into electric mobility with models like the all-electric Corsa-e and Mokka-e, aligning with Stellantis' broader strategy toward electrification.

        - **Innovative Technology**: Vauxhall vehicles are equipped with features like IntelliLink infotainment systems, advanced driver assistance systems, and a focus on fuel-efficient engines and electric powertrains.

        - **Motorsport Heritage**: Vauxhall has a history in British motorsports, particularly in the British Touring Car Championship (BTCC), where models like the Vauxhall Astra have competed successfully.

        - **Global Presence**: While primarily focused on the UK and European markets, Vauxhall's models are sometimes rebadged and sold under other brands like Opel in other regions.

        - **Sustainability Efforts**: Vauxhall is committed to sustainability with the introduction of electric vehicles, improved fuel efficiency in its combustion engines, and a broader goal to reduce its environmental impact in line with Stellantis' sustainability targets.
        """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/skoda.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Vauxhall Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

if selected_brand == "Volkswagen":
    st.title("Volkswagen Car Overview")

    st.markdown("""
        - **Founded**: Volkswagen was founded on May 28, 1937, in Wolfsburg, Germany, under the direction of the German Labour Front to produce affordable cars for the masses.

        - **Type of Company**: A multinational automobile manufacturer producing a broad range of vehicles, including sedans, SUVs, electric vehicles, and commercial vehicles. Volkswagen is the flagship brand of the Volkswagen Group.

        - **Brand Philosophy**: Volkswagen emphasizes quality, innovation, and sustainability, with a commitment to producing cars that combine modern design, cutting-edge technology, and reliable performance. The slogan "Das Auto" ("The Car") reflects this focus.

        - **Product Range**: Volkswagen offers popular models such as the Golf, Passat, Tiguan, Jetta, and the electric ID series. The brand caters to a wide audience with compact cars, family sedans, SUVs, and electric vehicles.

        - **Electric Vehicles**: Volkswagen is at the forefront of electric mobility with its ID. series, including models like the ID.3, ID.4, and ID.Buzz. They aim to lead the transition to electric vehicles as part of their "Way to Zero" strategy.

        - **Innovative Technology**: Volkswagen is known for its advancements in infotainment systems, driver-assistance technologies, and electric powertrains. The brand integrates features like the Digital Cockpit, Car-Net connectivity, and a focus on fuel efficiency.

        - **Motorsport Heritage**: Volkswagen has a strong motorsport background, particularly in rally racing, where models like the Volkswagen Polo R WRC have achieved significant success. The company has also competed in touring car and endurance racing.

        - **Global Presence**: Volkswagen operates in over 150 countries with manufacturing plants worldwide, including major markets in Europe, China, and the Americas. It is one of the largest automakers globally.

        - **Sustainability Efforts**: Volkswagen is heavily investing in sustainability through its "Way to Zero" initiative, aiming to become carbon neutral by 2050. The company is focusing on electric vehicles, renewable energy, and sustainable production practices.
        """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Datasets/vw.csv")
        return df

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    x = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # Features
    y = df.iloc[:, 2].values  # Target: Price

    # Apply Label Encoding
    le1 = LabelEncoder()
    x[:, 0] = le1.fit_transform(x[:, 0])  # Model column
    le2 = LabelEncoder()
    x[:, 4] = le2.fit_transform(x[:, 4])  # Fuel Type column

    # Apply One-Hot Encoding to the transmission column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
    x = ct.fit_transform(x)

    # Scale the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Model performance
    y_pred = model.predict(x_test)
    
    # Prediction
    st.title("Volkswagon Car Price Prediction")
    st.write("### Predictions")

    # Get user inputs for the prediction

    model_choice = st.selectbox("Car Model", df['model'].unique())
    year_choice = st.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].mean()))
    transmission_choice = st.selectbox("Transmission", df['transmission'].unique())
    mileage_choice = st.number_input("Mileage", min_value=0, value=int(df['mileage'].mean()))
    fuel_type_choice = st.selectbox("Fuel Type", df['fuelType'].unique())
    tax_choice = st.number_input("Tax", min_value=0, value=int(df['tax'].mean()))
    mpg_choice = st.number_input("Miles per Gallon (MPG)", min_value=0.0, value=float(df['mpg'].mean()))
    engine_size_choice = st.number_input("Engine Size (L)", min_value=0.0, value=float(df['engineSize'].mean()))

    # Apply the same preprocessing to the user inputs
    input_data = np.array([[model_choice, year_choice, transmission_choice, mileage_choice, fuel_type_choice, tax_choice, mpg_choice, engine_size_choice]])

    # Apply Label Encoding to model and fuel type
    input_data[:, 0] = le1.transform(input_data[:, 0])  # Apply label encoding on the model
    input_data[:, 4] = le2.transform(input_data[:, 4])  # Apply label encoding on the fuel type

    # Apply One-Hot Encoding to the transmission column
    input_data = ct.transform(input_data)

    # Apply scaling
    input_data = sc.transform(input_data)

    # Make prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")

