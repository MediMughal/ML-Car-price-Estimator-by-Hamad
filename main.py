import pandas as pd
import joblib
from fuzzywuzzy import process
from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st
import warnings
import numpy as np



# Files load karen

@st.cache_resource
def load_models():
    feature_columns = joblib.load("feature_columns.pkl")
    model_ml = joblib.load("car_price_model.pkl")
    le_make = joblib.load("make_encoder.pkl")
    return model_ml, le_make, feature_columns

from sklearn.model_selection import cross_val_score

@st.cache_resource
def run_cross_validation(_model, X, y):
    X_np = X.values if hasattr(X, "values") else X
    y_np = y.values if hasattr(y, "values") else y

    scores = cross_val_score(_model, X_np, y_np, cv=3, scoring='r2')
    return scores

# ✅ YAHAN ADD KARNA HAI (exactly yahan)
@st.cache_resource
def load_evaluation():
    return joblib.load("evaluation.pkl")


# 👇 phir ye already hai
model_ml, le_make, feature_columns = load_models()

eval_data = load_evaluation()

accuracy = eval_data["accuracy"]
precision = eval_data["precision"]
recall = eval_data["recall"]
f1 = eval_data["f1"]
cm = eval_data["cm"]

def get_model_name(model):
    # agar pipeline hai
    if hasattr(model, "named_steps"):
        # last step usually actual model hota hai
        last_step = list(model.named_steps.keys())[-1]
        return type(model.named_steps[last_step]).__name__
    
    return type(model).__name__

# Ab globally inhein assign karein
model_ml, le_make, feature_columns = load_models()


# === Setup ===
load_dotenv()
warnings.filterwarnings("ignore")

# Debug API key loading
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("pakwheels_used_cars.csv", encoding="utf-8")
    return df

df = load_data()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# === STEP 1: Prepare data ===
X = df[['engine_cc', 'mileage', 'year']]   # apne features adjust kar sakte ho
y = df['price']

# === STEP 2: Convert price into categories ===
def price_category(price):
    if price < 1000000:
        return 0   # Low
    elif price < 3000000:
        return 1   # Medium
    else:
        return 2   # High

y_class = y.apply(price_category)
# 🔥 YAHAN PASTE KARO
#accuracy, precision, recall, f1, feature_importance, grid, cv_scores, cm = train_evaluation_model(X, y_class)

# === STEP 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)



plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics, values)
plt.title("Model Evaluation Metrics")
plt.show()

# === Intelligent Price Calculation Functions ===
def calculate_base_price_by_segments(make, model_name, engine_cc, age):
    """Calculate base price using market segments and depreciation"""
    
    # Market segments with base prices (in PKR)
    luxury_brands = ['mercedes', 'bmw', 'audi', 'lexus', 'infiniti']
    premium_brands = ['toyota', 'honda', 'nissan', 'hyundai', 'kia', 'mazda']
    economy_brands = ['suzuki', 'daihatsu', 'changan', 'proton', 'united']
    
    # Base prices by engine size and brand category
    base_prices = {
        'luxury': {
            'small': 3000000,   # <1500cc
            'medium': 5000000,  # 1500-2500cc
            'large': 8000000    # >2500cc
        },
        'premium': {
            'small': 1500000,   # <1500cc
            'medium': 2500000,  # 1500-2500cc
            'large': 4000000    # >2500cc
        },
        'economy': {
            'small': 800000,    # <1500cc
            'medium': 1200000,  # 1500-2500cc
            'large': 1800000    # >2500cc
        }
    }
    
    # Determine brand category
    make_lower = make.lower()
    if make_lower in luxury_brands:
        category = 'luxury'
    elif make_lower in premium_brands:
        category = 'premium'
    else:
        category = 'economy'
    
    # Determine engine size category
    if engine_cc < 1500:
        size_cat = 'small'
    elif engine_cc <= 2500:
        size_cat = 'medium'
    else:
        size_cat = 'large'
    
    base_price = base_prices[category][size_cat]
    
    # Apply model-specific adjustments
    popular_models = {
        'corolla': 1.2, 'civic': 1.15, 'city': 1.1, 'accord': 1.3,
        'alto': 0.9, 'cultus': 0.95, 'swift': 1.05, 'wagon r': 0.92,
        'vitz': 1.0, 'aqua': 1.1, 'prius': 1.25, 'camry': 1.4
    }
    
    model_multiplier = popular_models.get(model_name.lower(), 1.0)
    base_price *= model_multiplier
    
    # Apply age-based depreciation (more realistic curve)
    if age <= 1:
        depreciation = 0.85  # 15% depreciation in first year
    elif age <= 3:
        depreciation = 0.75 - (age - 1) * 0.05  # 5% per year for years 2-3
    elif age <= 7:
        depreciation = 0.65 - (age - 3) * 0.04  # 4% per year for years 4-7
    elif age <= 15:
        depreciation = 0.49 - (age - 7) * 0.02  # 2% per year for years 8-15
    else:
        depreciation = max(0.15, 0.33 - (age - 15) * 0.01)  # Minimum 15% of base
    
    return int(base_price * depreciation)

def calculate_intelligent_range(raw_input, predicted_price, df):
    """
    ML Prediction ko base bana kar market-aligned range calculate karta hai.
    Honda aur Toyota ke liye realistic caps apply karta hai.
    """
    make = raw_input['make'].lower()
    model = raw_input['model'].lower()
    
    # 1. Realistic Market Reference (e.g., Civic 2021 approx 70 Lacs)
    # Agar ML model 1.3 Cr jaisi ghalat qeemat de, to ye usay wapas layega
    base_market_ref = predicted_price
    if ("honda" in make or "toyota" in make) and ("civic" in model or "corolla" in model):
        if raw_input['age'] <= 5: # 2021 model approx
            base_market_ref = 7200000 

    # 2. ML Prediction aur Market Ref ko blend karein
    # Agar ML bohot door hai, to market ref ko 70% weight dein
    if predicted_price > base_market_ref * 1.3:
        adjusted_price = (predicted_price * 0.3) + (base_market_ref * 0.7)
    else:
        adjusted_price = predicted_price

    # 3. Base Range (8% margin)
    lower = int(adjusted_price * 0.92)
    upper = int(adjusted_price * 1.08)
    
    # 4. Minimum Range Constraint
    if upper - lower < 100000:
        mid = (upper + lower) // 2
        lower = mid - 50000
        upper = mid + 50000
        
    return max(100000, lower), max(200000, upper)

def get_market_condition_factor(make, model_name, age):
    """Get market condition multiplier based on current demand"""
    
    # High demand models (retain value better)
    high_demand = ['corolla', 'civic', 'city', 'vitz', 'aqua', 'prius']
    # Low demand models (depreciate faster)
    low_demand = ['cultus', 'mehran', 'khyber', 'baleno']
    
    model_lower = model_name.lower()
    
    if model_lower in high_demand:
        return 1.1 if age <= 5 else 1.05
    elif model_lower in low_demand:
        return 0.9 if age <= 5 else 0.85
    else:
        return 1.0

# === Fuzzy correction ===
def correct_input(user_input, choices, threshold=70):
    match, score = process.extractOne(user_input, choices)
    return match if score >= threshold else user_input

def generate_price_reasoning(raw_input, ml_price, lower, upper):
    reasons = []

    age = raw_input['age']
    mileage = raw_input['mileage']

    expected_mileage = age * 15000

    if age > 10:
        reasons.append("High age significantly reduces resale value")
    elif age < 3:
        reasons.append("Low age increases market demand")

    if mileage > expected_mileage:
        reasons.append("High mileage reduces vehicle value")
    else:
        reasons.append("Mileage is within optimal range")

    if raw_input['engine_cc'] > 1800:
        reasons.append("Large engine increases maintenance cost")

    if raw_input['is_automatic']:
        reasons.append("Automatic transmission increases demand")

    if lower <= ml_price <= upper:
        reasons.append("ML prediction aligns with market range")
    else:
        reasons.append("ML prediction deviates from market trend (adjusted by model)")

    return reasons

# === AI Description function ===
def generate_ai_description(raw_input, lower, upper, expected_mileage, api_key):
    """Generate AI description using Groq API"""
    if "ai_cache" not in st.session_state:
     st.session_state.ai_cache = {}
    try:
        # Check if API key exists and is not empty
        if not api_key or api_key.strip() == "":
            return None, "API key is missing or empty"
            
        # Initialize Groq client
        client = Groq(api_key=api_key.strip())
        
        prompt = f"""
            You are an expert Pakistani car valuation analyst.

            Return a structured professional report in this format:

            1. Summary (2 lines only)
            2. Price justification (bullet points)
            3. Market demand level (High / Medium / Low)
            4. Key factors affecting price (bullet points)

            Car Details:
            - Make: {raw_input['make']}
            - Model: {raw_input['model']}
            - Body Type: {raw_input['body']}
            - Age: {raw_input['age']} years
            - Mileage: {raw_input['mileage']:,} km
            - Engine: {raw_input['engine_cc']} cc
            - Fuel Type: {raw_input['fuel_type']}
            - Transmission: {'Automatic' if raw_input['is_automatic'] else 'Manual'}

            Market Information:
            - Estimated Price Range: PKR {lower:,} - {upper:,}
            - Expected Mileage for Age: {expected_mileage:,} km

            Instructions:
            - Focus on Pakistan used car market trends
            - Be concise, analytical, and professional
            Use bullet points for clarity
            """

        cache_key = f"{raw_input['make']}_{raw_input['model']}_{raw_input['age']}_{lower}_{upper}"

        if cache_key in st.session_state.ai_cache:
             return st.session_state.ai_cache[cache_key], None
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional automotive market analyst specializing in Pakistan's used car market with deep knowledge of pricing trends, brand positioning, and consumer preferences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        st.session_state.ai_cache[cache_key] = response.choices[0].message.content

        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, str(e)

# === Streamlit page config & style ===
st.set_page_config(page_title="Car Price Estimator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-navy: #001f4d;
            --secondary-navy: #003366;
            --accent-blue: #0066cc;
            --light-blue: #00aaff;
            --dark-bg: #0a0a0a;
            --darker-bg: #050505;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --border-color: #333333;
            --gradient-bg: linear-gradient(135deg, #001f4d 0%, #003366 50%, #0066cc 100%);
        }
        
        /* Main app styling */
        .main {
            background: linear-gradient(180deg, #0a0a0a 0%, #0f0f0f 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit default elements */
        .stApp > header {
            background-color: transparent;
        }
        
        .stApp > div:first-child {
            background-color: transparent;
        }
        
        /* Enhanced navy ribbon with gradient and shadow - Full width */
        .navy-ribbon {
        background: var(--gradient-bg);
        color: var(--text-primary);
        padding: 20px 40px;
        font-size: 28px;
        font-weight: 700;
        border-radius: 50px;
        margin: -20px 0 30px 0;
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 31, 77, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-left: none;
        border-right: none;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
        
        .navy-ribbon::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Sidebar enhancements */
        .css-1d391kg {
            background: linear-gradient(180deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            padding: 20px;
            border-right: 3px solid var(--primary-navy);
            height: 100vh;
            overflow-y: auto;
        }
        
        /* Form controls styling */
        .stSelectbox > div > div {
            background-color: var(--darker-bg) !important;
            border: 2px solid var(--primary-navy) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .stSelectbox > div > div > div {
            color: var(--text-primary) !important;
        }
        
        .stNumberInput > div > div > input {
            background-color: var(--darker-bg) !important;
            border: 2px solid var(--primary-navy) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .stSlider > div > div > div {
            background-color: var(--primary-navy) !important;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: var(--gradient-bg) !important;
            color: var(--text-primary) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 15px 40px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            width: 100% !important;
            box-shadow: 0 6px 20px rgba(0, 31, 77, 0.3) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 102, 204, 0.4) !important;
            background: linear-gradient(135deg, #0066cc 0%, #001f4d 50%, #003366 100%) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
        
        /* Result styling */
        .price-result {
            background: linear-gradient(135deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
            border: 2px solid var(--accent-blue);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            width: 100%; 
            box-shadow: 0 8px 32px rgba(0, 170, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .price-result::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-bg);
        }
        
        /* Description section styling - FIXED FOR VISIBILITY */
        .description-section {
            background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid var(--light-blue);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        /* Force white text color for all description content */
        .description-section p {
            color: #ffffff !important;
            line-height: 1.7 !important;
            font-size: 15px !important;
            text-align: justify !important;
            font-weight: 400 !important;
        }
        
        /* Ensure all text in description section is visible */
        .description-section * {
            color: #ffffff !important;
        }
        
        /* Specific class for AI description text */
        .ai-description-text {
            color: #ffffff !important;
            line-height: 1.7 !important;
            font-size: 15px !important;
            text-align: justify !important;
            font-weight: 400 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Breakdown section */
        .breakdown-section {
            background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 3px solid #ff6b35;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        .breakdown-content {
            font-size: 13px;
            line-height: 1.5;
            color: #e0e0e0 !important;
        }
        
        /* Section heading styling */
        .main h3 {
            color: var(--light-blue);
            font-size: 20px;
            font-weight: 600;
            margin: 20px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary-navy);
        }
        
        /* Enhanced form control spacing */
        .stSelectbox, .stNumberInput, .stSlider {
            margin-bottom: 10px;
        }
        
        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--primary-navy), transparent);
            margin: 15px 0;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-navy);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-blue);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .navy-ribbon {
                margin: 0 -10px 20px -10px;
                padding: 15px 20px;
                font-size: 22px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar with project description ===
with st.sidebar:
    st.markdown(
        """
        <p style="font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 10px;">
        Project Description
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class="justified-text" style="color: #e0e0e0; text-align: justify; line-height: 1.6; font-size: 14px;">
        This project presents an Intelligent Vehicle Valuation System By Hammad & Maryam designed for the Pakistani automotive market. It leverages data-driven algorithms to analyze key price determinants including market segmentation, brand equity, and non-linear depreciation models. The system incorporates multifaceted variables such as mileage-to-age ratios, transmission dynamics, and fuel efficiency metrics to estimate real-time market value. By integrating comparative market analysis with heuristic pricing models, the application provides an accurate, evidence-based price range for used vehicles.
        </p>
        """,
        unsafe_allow_html=True
    )

# === Main content ===
st.markdown(
    '<div class="navy-ribbon"> Intelligent Car Price Estimator </div>', 
    unsafe_allow_html=True
)

# Filter out numeric values from model column
def filter_string_models(models):
    """Filter out numeric values and keep only string model names"""
    string_models = []
    for model in models:
        model_str = str(model).strip()
        if not model_str.isdigit() and model_str.lower() not in ['nan', 'none', '']:
            try:
                float(model_str)
                continue
            except ValueError:
                string_models.append(model_str)
    return sorted(list(set(string_models)))

# Input controls

st.markdown("### 🔧 Vehicle Details")

# Row 1: Make aur Model ki dynamic selection
row1_cols = st.columns([2, 2, 1.5, 1.5])

# 1. Make Selection
make_list = sorted(df['make'].unique().tolist())
make_options = ["Select Make"] + make_list
make = row1_cols[0].selectbox("Make", make_options, index=0)

# 2. Model Selection (Jo Make par depend karega)
if make != "Select Make":
    # Sirf selected brand ki cars filter karein
    make_filtered_df = df[df['make'] == make]
    available_models = filter_string_models(make_filtered_df['model'].dropna().unique())
    model_options = ["Select Model"] + available_models
else:
    make_filtered_df = df.copy()
    model_options = ["Select Model"]

model_input = row1_cols[1].selectbox("Model", model_options, index=0)

# 3. Age aur Mileage
age = row1_cols[2].slider("Age (years)", 0, 25, 5)
mileage = row1_cols[3].number_input("Mileage (km)", min_value=0, value=50000, step=500)

st.markdown("---")

# Row 2: Baki features jo Model par depend karte hain
row2_cols = st.columns([2, 2, 2, 2])

if model_input != "Select Model":
    # Mazeed filter karein taake engine aur fuel wahi ayein jo is model mein hotay hain
    model_filtered_df = make_filtered_df[make_filtered_df['model'] == model_input]
    
    engine_list = sorted([str(int(float(cc))) for cc in model_filtered_df['engine_cc'].dropna().unique()])
    engine_options = ["Select Engine CC"] + engine_list
    
    fuel_list = sorted(model_filtered_df['fuel_type'].dropna().unique().tolist())
    fuel_options = ["Select Fuel Type"] + fuel_list
    
    body_list = sorted(model_filtered_df['body'].dropna().unique().tolist())
    body_options = ["Select Body Type"] + body_list
else:
    engine_options = ["Select Engine CC"]
    fuel_options = ["Select Fuel Type"]
    body_options = ["Select Body Type"]

engine_cc = row2_cols[0].selectbox("Engine CC", engine_options, index=0)
fuel_type = row2_cols[1].selectbox("Fuel Type", fuel_options, index=0)
body = row2_cols[2].selectbox("Body Type", body_options, index=0)

# Row 3: Transmission aur Assembly
row3_cols = st.columns([2, 2, 4])
transmission = row3_cols[0].selectbox("Transmission", ["Select Transmission", "Manual", "Automatic"], index=0)
assembled = row3_cols[1].selectbox("Assembled", ["Select Assembly", "Local", "Imported"], index=0)

# Submit Button Logic
popular_makes = ["toyota", "suzuki", "honda"]
is_popular_make = 1 if make.lower() in popular_makes else 0

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    all_selected = (
        make != "Select Make" and 
        model_input != "Select Model" and 
        engine_cc != "Select Engine CC" and 
        fuel_type != "Select Fuel Type" and 
        body != "Select Body Type" and 
        transmission != "Select Transmission" and 
        assembled != "Select Assembly"
    )
    
    if all_selected:
        submit_btn = st.button("🔍 Get Intelligent Price Analysis", key="submit_btn")
    else:
        st.info("⚠️ Please select all vehicle details to get price analysis")
        submit_btn = False

 # === Is ke nechy aapka 'if submit_btn:' wala logic start hoga ===


    if submit_btn:
     with st.spinner('Running advanced ML market analysis and AI algorithms...'):
        try:
            # 1. Sab se pehle values define karein
            engine_cc_value = int(float(engine_cc)) if engine_cc != "Select Engine CC" else 1000
            import datetime
            current_year = datetime.datetime.now().year
            year_input = current_year - age
            
            # 2. raw_input dictionary banayein (AI aur Breakdown ke liye)
            raw_input = {
                'make': make, 
                'model': model_input, 
                'age': age, 
                'mileage': mileage,
                'engine_cc': engine_cc_value, 
                'is_automatic': 1 if transmission == "Automatic" else 0,
                'fuel_type': fuel_type,
                'body': body
            }

            # 3. ML Prediction Logic
            try:
                make_encoded = le_make.transform([make])[0]
            except:
                make_encoded = 0 
            
            car_age = age
            mileage_per_year = mileage / (car_age + 1)
            engine_per_age = engine_cc_value / (car_age + 1)

            # Features ko training data ke exact order mein rakhein
            user_input = pd.DataFrame([[
                make_encoded, 
                engine_cc_value, 
                mileage, 
                year_input, # Ensure year_input is 2021
                mileage_per_year, 
                engine_per_age
            ]], columns=feature_columns)

           # ML Prediction
            raw_ml_price = np.expm1(model_ml.predict(user_input)[0])

            cv_scores = run_cross_validation(model_ml, X, y_class)

            # ===============================
            # FEATURE IMPORTANCE (FIX)
            # ===============================
            feature_importance = None

            if hasattr(model_ml, "named_steps"):
                model = list(model_ml.named_steps.values())[-1]

                if hasattr(model, "feature_importances_"):
                    feature_importance = pd.Series(
                        model.feature_importances_,
                        index=feature_columns
                    )

            # Extra info
            raw_input['is_imported'] = 1 if assembled == "Imported" else 0

            # Market range (ONLY ONCE)
            lower, upper = calculate_intelligent_range(
                raw_input,
                raw_ml_price,
                df
            )

            # Final ML price (NO distortion)
            ml_predicted_price = raw_ml_price

            # Mid price
            mid_price = (lower + upper) / 2

            # Error ratio
            error_ratio = abs(ml_predicted_price - mid_price) / max(mid_price, 1)
            # Smooth confidence (base)
            error_ratio = abs(ml_predicted_price - mid_price) / max(mid_price, 1)

            confidence = 100 - (error_ratio * 50)

            # penalty if ML is totally off
            if ml_predicted_price > upper * 2 or ml_predicted_price < lower * 0.5:
                confidence *= 0.6

            # final clamp (ONLY ONCE)
            confidence = max(40, min(confidence, 95))
            confidence = round(confidence, 2)

            # Format helpers
            def format_price(n): return f"{n:,.0f}"
            def format_k(n): return f"{int(n//1000)}k"


            # 4. Display Price Result
            st.markdown(f"""
                <div class="price-result">
                    <h2 style='color: #00aaff; margin-bottom: 15px; text-align: center;'>
                         Intelligent Market Valuation 💵
                    </h2>
                    <h3 style='color: #ffffff; text-align: center; font-size: 28px; margin: 0;'>
                        PKR. {format_price(lower)} - PKR. {format_price(upper)}
                    </h3>
                    <p style='color: #e0e0e0; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>
                        ({format_k(lower)} - {format_k(upper)})
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            reasons = generate_price_reasoning(raw_input, ml_predicted_price, lower, upper)

            st.markdown("### 📊 Why this price?")

            for r in reasons:
                st.markdown(f"✔️ {r}")


            st.success(f"Model Confidence: {confidence:.2f}%")
            if confidence > 85:
                st.success("High confidence: ML prediction strongly aligns with market trends")
            elif confidence > 60:
                st.warning("Moderate confidence: Some deviation between ML and market")
            else:
                st.error("Low confidence: Prediction may be unreliable")
                
            st.markdown("### 🎯 Model Confidence Analysis")

            st.progress(int(confidence))

            st.markdown(f"""
            - Confidence Score: **{confidence:.2f}%**
            - ML Prediction: PKR {ml_predicted_price:,.0f}
            - Market Range: PKR {lower:,.0f} - {upper:,.0f}
            """)

            st.markdown("### 📊 Model Evaluation")

            # ================== BONUS OUTPUT ==================

            st.markdown("### 🔍 Feature Importance")

            if feature_importance is not None:
                st.bar_chart(feature_importance.sort_values(ascending=False))
            else:
                st.info("This model does not support feature importance")
            

            st.markdown("### ⚙️ Model Configuration")

            # STEP 1: Show model name
            st.write("Model Used:", get_model_name(model_ml))

            # STEP 2: Get actual model safely
            if hasattr(model_ml, "named_steps"):
                model = model_ml.named_steps['model']
            else:
                model = model_ml

            st.write("Model:", type(model).__name__)

            # STEP 3: Show only important parameters (clean output)
            if hasattr(model, "get_params"):
                important_params = {
                    "n_estimators": model.get_params().get("n_estimators"),
                    "max_depth": model.get_params().get("max_depth"),
                    "learning_rate": model.get_params().get("learning_rate"),
                    "subsample": model.get_params().get("subsample")
                }

                # Remove None values (optional clean step)
                important_params = {k: v for k, v in important_params.items() if v is not None}

                st.json(important_params)
            else:
                st.info("Parameters not available for this model")

            st.markdown("### 🔁 Cross Validation")

            st.write("Fold Scores:")
            st.write(cv_scores)

            st.success(f"Average R² Score: {cv_scores.mean():.3f}")
            st.info(f"Std Dev (Stability): {cv_scores.std():.3f}")

            st.markdown("### 📈 Performance Metrics")

            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)


            mid_price = (lower + upper) / 2

            final_price = (ml_predicted_price * 0.3) + (mid_price * 0.7)

            agreement = 100 - (abs(ml_predicted_price - mid_price) / mid_price * 100)
            agreement = max(0, min(100, agreement))
            agreement = round(agreement, 2)

            st.info(f"ML vs Market Agreement: {agreement:.2f}%")


            # 5. Machine Learning Analysis (With Segment, Depreciation & Source)
            avg_annual_mileage = 15000
            expected_mileage = age * avg_annual_mileage
            
            # Segment Logic
            luxury_brands = ['mercedes', 'bmw', 'audi', 'lexus', 'land rover']
            premium_brands = ['toyota', 'honda', 'nissan', 'hyundai', 'kia']
            segment = "Luxury" if make.lower() in luxury_brands else "Premium" if make.lower() in premium_brands else "Economy"
            
            # Depreciation Logic (Estimated based on age)
            dep_rate = min(85, age * 7) if age > 0 else 0 # Simple 7% per year rule for display

            st.markdown(f"""
                <div class="breakdown-section">
                    <h3 style='color: #ff6b35; margin-bottom: 12px; font-size: 18px;'>
                        📊 Machine Learning Analysis
                    </h3>
                    <div class='breakdown-content' style='color: #e0e0e0; line-height: 1.6;'>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div>
                                <strong>ML Model:</strong> {get_model_name(model_ml)} <br>
                                <strong>Segment:</strong> {segment}<br>
                                <strong>Data Source:</strong> PakWheels CSV
                            </div>
                            <div>
                                <strong>Depreciation:</strong> ~{dep_rate}%<br>
                                <strong>Year Model:</strong> {year_input}<br>
                                <strong>Engine:</strong> {engine_cc_value}cc
                            </div>
                        </div>
                        <hr style="margin: 10px 0; border-color: #444;">
                        <strong>Expected Mileage:</strong> {format_price(expected_mileage)} km | 
                        <strong>Actual:</strong> {format_price(mileage)} km 
                        {'✅ (Optimal)' if mileage <= expected_mileage else '⚠️ (High Usage)'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 6. AI Report (Using raw_input)
            description, error = generate_ai_description(raw_input, lower, upper, expected_mileage, api_key)
            if description:
                st.markdown(f"""
                    <div class="description-section">
                        <h3 style='color: #00aaff; margin-bottom: 15px;'>📑 AI Market Valuation Report</h3>
                        <p class="ai-description-text">{description}</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Logic Error: {e}")