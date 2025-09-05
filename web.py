import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix NumPy bool deprecation issue
if not hasattr(np, 'bool'):
    np.bool = bool

# Global matplotlib font settings to ensure proper display
def setup_font():
    """Setup font configuration"""
    try:
        import matplotlib.font_manager as fm

        # Try various fonts
        fonts = [
            'DejaVu Sans',
            'Arial',
            'Liberation Sans',
            'Helvetica',
            'sans-serif'
        ]

        # Get available system fonts
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # Find available font
        for font in fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"Using font: {font}")
                return font

        # If no specific font found, use default
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("Using default font")
        return None

    except Exception as e:
        print(f"Font setup failed: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

# Setup font and minus sign display
font = setup_font()
matplotlib.rcParams['axes.unicode_minus'] = False

# Set page title and layout
st.set_page_config(
    page_title="Cumulative live birth rate prediction system V1.0 for patients with polycystic ovary syndrome receiving assisted reproductive treatment",
    page_icon="🏥",
    layout="wide"
)

# Set font and minus sign display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Define global variables
global feature_names, feature_dict, variable_descriptions

# Feature names (using 15 specified variables)
feature_names_display = [
    'age', 'LDL', 'bPRL', 'bE2', 'AMH', 'S_Dose', 'T_Dose',
    'D5_FSH', 'D5_LH', 'D5_E2', 'HCG_E2', 'HCG_LH', 'Ocytes', 'BFR', 'Cycles'
]

# English feature names
feature_names_en = [
    'Female Age', 'LDL Cholesterol', 'Baseline Prolactin', 'Baseline Estradiol', 'Anti-Müllerian Hormone',
    'Gonadotropin Starting Dose', 'Total Gonadotropin Dose', 'Day 5 FSH', 'Day 5 LH', 'Day 5 Estradiol',
    'HCG Day Estradiol', 'HCG Day LH', 'Oocytes Retrieved', 'Blastocyst Formation Rate', 'Total Transfer Cycles'
]

feature_dict = dict(zip(feature_names_display, feature_names_en))

# Variable description dictionary (including 15 specified variables)
variable_descriptions = {
    'age': 'Female Age (years)',
    'LDL': 'Low Density Lipoprotein (mmol/L)',
    'bPRL': 'Baseline Prolactin (ng/mL)',
    'bE2': 'Baseline Estradiol (pg/mL)',
    'AMH': 'Anti-Müllerian Hormone (ng/mL)',
    'S_Dose': 'Initial Dose of Gonadotropin (IU)',
    'T_Dose': 'Total Dose of Gonadotropin (IU)',
    'D5_FSH': 'FSH level on day 5 of ovarian stimulation (mIU/mL)',
    'D5_LH': 'LH level on day 5 of ovarian stimulation (mIU/mL)',
    'D5_E2': 'Estradiol level on day 5 of ovarian stimulation (pg/mL)',
    'HCG_E2': 'Estradiol on HCG Day (pg/mL)',
    'HCG_LH': 'LH on HCG Day (mIU/mL)',
    'Ocytes': 'Oocytes Retrieved (count)',
    'BFR': 'Blastocyst Formation Rate (%)',
    'Cycles': 'Number of Transplantation Cycles (count)'
}

# Load XGBoost model and related files
@st.cache_resource
def load_model():
    # Load XGBoost model
    model = joblib.load('./best_xgboost_model.pkl')

    # Load scaler
    scaler = joblib.load('./scaler.pkl')

    # Load feature column names
    with open('./feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns

# Main application
def main():
    global feature_names, feature_dict, variable_descriptions

    # Sidebar title
    st.sidebar.title("Cumulative live birth rate prediction system V1.0 for patients with polycystic ovary syndrome receiving assisted reproductive treatment")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)

    # Add system description to sidebar
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is an XGBoost algorithm-based prediction system for cumulative live birth rates in PCOS patients undergoing assisted reproduction. It analyzes patients' clinical indicators and treatment process data to predict the probability of cumulative live birth.

    ## Prediction Results
    The system predicts:
    - Cumulative live birth probability
    - No cumulative live birth probability
    - Risk assessment (Low risk, Medium risk, High risk)

    ## How to Use
    1. Fill in patient clinical indicators on the main interface
    2. Click the prediction button to generate results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure patient information is entered accurately
    - All fields must be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)

    # Add variable descriptions to sidebar
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")

    # Main page title
    st.title("Cumulative live birth rate prediction system V1.0 for patients with polycystic ovary syndrome receiving assisted reproductive treatment")
    st.markdown("### XGBoost Algorithm-Based Cumulative Live Birth Rate Assessment")

    # Load model
    try:
        model, scaler, feature_columns = load_model()
        st.sidebar.success("XGBoost model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return

    # Create input form
    st.header("Patient Information Input")
    st.markdown("### Please fill in the following 15 key indicators")

    # Create tabs to organize input
    tab1, tab2, tab3, tab4 = st.tabs(["Patient Baseline Info", "Stimulation Monitoring", "Ovulation Trigger Indicators", "Embryo Testing & Transfer"])

    with tab1:
        st.subheader("Patient Baseline Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Female Age (years)", min_value=18, max_value=50, value=30)
            ldl = st.number_input("LDL Cholesterol (mmol/L)", min_value=1.0, max_value=8.0, value=2.8, step=0.1)
            bprl = st.number_input("Baseline Prolactin (ng/mL)", min_value=1.0, max_value=100.0, value=15.0, step=0.1)

        with col2:
            be2 = st.number_input("Baseline Estradiol (pg/mL)", min_value=10.0, max_value=200.0, value=40.0, step=1.0)
            amh = st.number_input("Anti-Müllerian Hormone (ng/mL)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    with tab2:
        st.subheader("Controlled Ovarian Hyperstimulation")
        col1, col2 = st.columns(2)

        with col1:
            s_dose = st.number_input("Initial dose of gonadotropin (IU)", min_value=75, max_value=450, value=225)
            t_dose = st.number_input("Total dose of gonadotropin (IU)", min_value=500, max_value=5000, value=2250)
            d5_fsh = st.number_input("FSH level on day 5 of ovarian stimulation (mIU/mL)", min_value=1.0, max_value=50.0, value=8.0, step=0.1)

        with col2:
            d5_lh = st.number_input("LH level on day 5 of ovarian stimulation (mIU/mL)", min_value=0.5, max_value=30.0, value=3.0, step=0.1)
            d5_e2 = st.number_input("Estradiol level on day 5 of ovarian stimulation (pg/mL)", min_value=50.0, max_value=2000.0, value=200.0, step=10.0)

    with tab3:
        st.subheader("Ovulation Trigger Criteria")
        col1, col2 = st.columns(2)

        with col1:
            hcg_e2 = st.number_input("Estradiol on HCG Day (pg/mL)", min_value=500.0, max_value=8000.0, value=2000.0, step=50.0)

        with col2:
            hcg_lh = st.number_input("LH on HCG Day (mIU/mL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)

    with tab4:
        st.subheader("Embryo Assessment Parameters & Transfer Cycles")
        col1, col2 = st.columns(2)

        with col1:
            ocytes = st.number_input("Oocytes Retrieved (count)", min_value=1, max_value=50, value=12)
            bfr = st.number_input("Blastocyst Formation Rate (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

        with col2:
            cycles = st.number_input("Number of Transplantation Cycles (count)", min_value=1, max_value=10, value=1)

    # Create prediction button
    predict_button = st.button("Predict Cumulative Live Birth Rate", type="primary")

    if predict_button:
        # Collect 15 input features
        features = [
            age, ldl, bprl, be2, amh, s_dose, t_dose,
            d5_fsh, d5_lh, d5_e2, hcg_e2, hcg_lh, ocytes, bfr, cycles
        ]

        # Convert to DataFrame (containing 15 feature columns)
        input_df = pd.DataFrame([features], columns=feature_columns)

        # Standardize continuous variables (all 15 variables are continuous)
        continuous_vars = ['age', 'LDL', 'bPRL', 'bE2', 'AMH', 'S_Dose', 'T_Dose',
                          'D5_FSH', 'D5_LH', 'D5_E2', 'HCG_E2', 'HCG_LH', 'Ocytes', 'BFR', 'Cycles']

        # Create a copy of input data for standardization
        input_scaled = input_df.copy()
        input_scaled[continuous_vars] = scaler.transform(input_df[continuous_vars])

        # Make prediction
        prediction = model.predict_proba(input_scaled)[0]
        no_birth_prob = prediction[0]
        birth_prob = prediction[1]

        # Display prediction results
        st.header("Cumulative Live Birth Rate Prediction Results")

        # Use progress bars to display probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("No Cumulative Live Birth Probability")
            st.progress(float(no_birth_prob))
            st.write(f"{no_birth_prob:.2%}")

        with col2:
            st.subheader("Cumulative Live Birth Probability")
            st.progress(float(birth_prob))
            st.write(f"{birth_prob:.2%}")

        # Risk assessment
        risk_level = "Low Probability" if birth_prob < 0.3 else "Medium Probability" if birth_prob < 0.7 else "High Probability"
        risk_color = "red" if birth_prob < 0.3 else "orange" if birth_prob < 0.7 else "green"

        st.markdown(f"### Cumulative Live Birth Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        


        # Add model explanation
        st.write("---")
        st.subheader("Model Explanation")

        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # Handle SHAP values format - shape (1, 10, 2) means 1 sample, 10 features, 2 classes
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # Take SHAP values for positive class (live birth class, index 1) of first sample
                shap_value = shap_values[0, :, 1]  # Shape becomes (10,)
                expected_value = explainer.expected_value[1]  # Expected value for positive class
            elif isinstance(shap_values, list):
                # If list format, take SHAP values for positive class
                shap_value = np.array(shap_values[1][0])
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_value = np.array(shap_values[0])
                expected_value = explainer.expected_value

            # Feature contribution analysis table
            st.subheader("Feature Contribution Analysis")

            # Create contribution table
            feature_values = []
            feature_impacts = []

            # Get SHAP values (no ID column now)
            for i, feature in enumerate(feature_names_display):
                # Find corresponding feature in input_df
                feature_values.append(float(input_df[feature].iloc[0]))
                # SHAP values should now be 1D array, use index directly
                impact_value = float(shap_value[i])
                feature_impacts.append(impact_value)

            shap_df = pd.DataFrame({
                'Feature': [feature_dict.get(f, f) for f in feature_names_display],
                'Value': feature_values,
                'Impact': feature_impacts
            })

            # Sort by absolute impact
            shap_df['Absolute Impact'] = shap_df['Impact'].abs()
            shap_df = shap_df.sort_values('Absolute Impact', ascending=False)

            # Display table
            st.table(shap_df[['Feature', 'Value', 'Impact']])
            
            # SHAP Waterfall Plot
            st.subheader("SHAP Waterfall Plot")
            try:
                # Create SHAP waterfall plot
                import matplotlib.font_manager as fm

                # Set English fonts
                try:
                    # Use English fonts for better compatibility
                    english_fonts = [
                        'DejaVu Sans',
                        'Arial',
                        'Liberation Sans',
                        'Helvetica',
                        'sans-serif'
                    ]
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    english_font = None
                    for font in english_fonts:
                        if font in available_fonts:
                            english_font = font
                            break

                    if english_font:
                        plt.rcParams['font.sans-serif'] = [english_font, 'DejaVu Sans']
                        plt.rcParams['font.family'] = 'sans-serif'
                    else:
                        # Use default English fonts
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                        plt.rcParams['font.family'] = 'sans-serif'

                except Exception:
                    # Font setup failed, use default fonts
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                    plt.rcParams['font.family'] = 'sans-serif'

                plt.rcParams['axes.unicode_minus'] = False

                fig_waterfall = plt.figure(figsize=(12, 8))

                # Create English feature names for better display
                english_feature_map = {
                    'Insemination': 'Treatment_Method', 'Complication': 'Complication',
                    'Years': 'Infertility_Years', 'Type': 'Infertility_Type',
                    'age': 'Female_Age', 'BMI': 'BMI', 'AMH': 'AMH', 'AFC': 'AFC',
                    'FBG': 'Fasting_Glucose', 'TC': 'Total_Cholesterol', 'TG': 'Triglycerides',
                    'HDL': 'HDL_Cholesterol', 'LDL': 'LDL_Cholesterol',
                    'bFSH': 'Baseline_FSH', 'bLH': 'Baseline_LH', 'bPRL': 'Baseline_PRL',
                    'bE2': 'Baseline_E2', 'bP': 'Baseline_P', 'bT': 'Baseline_T',
                    'D3_FSH': 'Day3_FSH', 'D3_LH': 'Day3_LH', 'D3_E2': 'Day3_E2',
                    'D5_FSH': 'Day5_FSH', 'D5_LH': 'Day5_LH', 'D5_E2': 'Day5_E2',
                    'COS': 'Stimulation_Protocol', 'S_Dose': 'Starting_Dose',
                    'T_Days': 'Treatment_Days', 'T_Dose': 'Total_Dose',
                    'HCG_LH': 'HCG_Day_LH', 'HCG_E2': 'HCG_Day_E2', 'HCG_P': 'HCG_Day_P',
                    'Ocytes': 'Retrieved_Oocytes', 'MII': 'MII_Rate', '2PN': 'Fertilization_Rate',
                    'CR': 'Cleavage_Rate', 'GVE': 'Good_Embryo_Rate',
                    'BFR': 'Blastocyst_Rate', 'Stage': 'Transfer_Stage',
                    'Cycles': 'Transfer_Cycles'
                }

                english_names = [english_feature_map.get(f, f) for f in feature_names_display]

                # Use English feature names for display
                try:
                    # Use English feature names
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=input_df.iloc[0].values,
                            feature_names=[feature_dict.get(f, f) for f in feature_names_display]
                        ),
                        max_display=15,  # Display all 15 features
                        show=False
                    )
                except Exception as display_error:
                    st.warning("Feature name display failed, using simplified English names")
                    # If display fails, use simplified English feature names
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=input_df.iloc[0].values,
                            feature_names=english_names
                        ),
                        max_display=15,
                        show=False
                    )

                # Set English fonts and fix minus sign display
                for ax in fig_waterfall.get_axes():
                    # Set axis label font size
                    ax.tick_params(labelsize=10)

                    # Fix all text fonts and minus signs
                    for text in ax.texts:
                        text_content = text.get_text()
                        # Replace unicode minus
                        if '−' in text_content:
                            text.set_text(text_content.replace('−', '-'))
                        # Set font
                        if english_font:
                            text.set_fontfamily(english_font)
                        text.set_fontsize(10)

                    # Set y-axis label font
                    for label in ax.get_yticklabels():
                        if english_font:
                            label.set_fontfamily(english_font)
                        label.set_fontsize(10)

                    # Set x-axis label font
                    for label in ax.get_xticklabels():
                        if english_font:
                            label.set_fontfamily(english_font)
                        label.set_fontsize(10)

                plt.tight_layout()
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)


            except Exception as e:
                st.error(f"Unable to generate waterfall plot: {str(e)}")
                # Use bar chart as alternative
                fig_bar = plt.figure(figsize=(10, 6))

                # Set English fonts
                try:
                    import matplotlib.font_manager as fm
                    english_fonts = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    english_font = None
                    for font in english_fonts:
                        if font in available_fonts:
                            english_font = font
                            break

                    if english_font:
                        plt.rcParams['font.sans-serif'] = [english_font, 'DejaVu Sans']
                    else:
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

                plt.rcParams['axes.unicode_minus'] = False

                sorted_idx = np.argsort(np.abs(shap_value))[-15:]  # Display all 15 features

                bars = plt.barh(range(len(sorted_idx)), shap_value[sorted_idx])

                # Set y-axis labels (feature names)
                feature_labels = [feature_dict.get(feature_names_display[i], feature_names_display[i]) for i in sorted_idx]
                plt.yticks(range(len(sorted_idx)), feature_labels)

                plt.xlabel('SHAP Value')
                plt.title('Feature Impact on Cumulative Live Birth Prediction')

                # Set different colors for positive and negative values
                for i, bar in enumerate(bars):
                    if shap_value[sorted_idx[i]] >= 0:
                        bar.set_color('lightcoral')
                    else:
                        bar.set_color('lightblue')

                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # SHAP Force Plot
            st.subheader("SHAP Force Plot")

            try:
                # Use official SHAP force plot in HTML format
                import streamlit.components.v1 as components
                import matplotlib

                # Set fonts to ensure proper minus sign display
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                matplotlib.rcParams['axes.unicode_minus'] = False

                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[feature_dict.get(f, f) for f in feature_names_display]
                )

                # Get SHAP HTML content and add CSS to fix display issues
                shap_html = f"""
                <head>
                    {shap.getjs()}
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px 10px 40px 10px;
                            overflow: visible;
                        }}
                        .force-plot {{
                            margin: 20px 0 40px 0 !important;
                            padding: 20px 0 40px 0 !important;
                        }}
                        svg {{
                            margin: 20px 0 40px 0 !important;
                        }}
                        .tick text {{
                            margin-bottom: 20px !important;
                        }}
                        .force-plot-container {{
                            min-height: 200px !important;
                            padding-bottom: 50px !important;
                        }}
                    </style>
                </head>
                <body>
                    <div class="force-plot-container">
                        {force_plot.html()}
                    </div>
                </body>
                """

                # Add more height space
                components.html(shap_html, height=400, scrolling=False)

            except Exception as e:
                st.error(f"Unable to generate HTML force plot: {str(e)}")
                st.info("Please check SHAP version compatibility")

        except Exception as e:
            st.error(f"Unable to generate SHAP explanation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("Using model feature importance as alternative")

            # Display model feature importance
            st.write("---")
            st.subheader("Feature Importance")

            # Get feature importance from XGBoost model
            try:
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': [feature_dict.get(f, f) for f in feature_names_display],
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(12, 8))

                # Set English fonts
                try:
                    import matplotlib.font_manager as fm
                    english_fonts = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    english_font = None
                    for font in english_fonts:
                        if font in available_fonts:
                            english_font = font
                            break

                    if english_font:
                        plt.rcParams['font.sans-serif'] = [english_font, 'DejaVu Sans']
                    else:
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

                plt.rcParams['axes.unicode_minus'] = False

                bars = plt.barh(range(len(importance_df)), importance_df['Importance'], color='skyblue')
                plt.yticks(range(len(importance_df)), importance_df['Feature'])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Feature Importance')

                # Set fonts
                if 'english_font' in locals() and english_font:
                    ax.set_xlabel('Importance', fontfamily=english_font)
                    ax.set_ylabel('Feature', fontfamily=english_font)
                    ax.set_title('Feature Importance', fontfamily=english_font)

                    # Set tick label fonts
                    for label in ax.get_yticklabels():
                        label.set_fontfamily(english_font)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e2:
                st.error(f"Unable to display feature importance: {str(e2)}")

if __name__ == "__main__":
    main()
