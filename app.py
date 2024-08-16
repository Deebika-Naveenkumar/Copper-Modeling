import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


st.set_page_config(layout="wide")

# Custom HTML for centered and colored header
st.markdown(
    """
    <h1 style='text-align: center; color: orange;'>
    Industrial Copper Modeling Application
    </h1>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])
with tab1:
    # Define the possible values for the dropdown menus
    status_options      = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']
    item_type_options   = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options     = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product             = ['1670798778', '1668701718', '628377', '640665', '611993', '1668701376', '164141591', '1671863738', '1332077137', '640405', '1693867550', '1665572374', '1282007633', '1668701698', '628117', '1690738206', '628112', '640400', '1671876026', '164336407', '164337175', '1668701725', '1665572032', '611728', '1721130331', '1693867563', '611733', '1690738219', '1722207579', '929423819', '1665584320', '1665584662', '1665584642']

    
    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)

        with col3:
            st.write(
                f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value. Do not give zero for thickness and quantity tons.</h5>',
                unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
         # Validate inputs
        if submit_button:
            try:
                q_tons = float(quantity_tons)
                thk = float(thickness)
                width_val = float(width)
                customer_id = float(customer)
                
                if q_tons <= 0 or thk <= 0:
                    st.write("Since log(0) or log of negative values leads to infinity, don't give zero or negative values for Quantity Tons and Thickness.")

                else:
                    import pickle

                    with open("source/model.pkl", 'rb') as file:
                        model = pickle.load(file)

                    with open("source/scaler.pkl", 'rb') as file:
                        scaler = pickle.load(file)

                    with open("source/itemtype.pkl", 'rb') as file:
                        itemtype = pickle.load(file)

                    with open("source/status.pkl", 'rb') as file:
                        status = pickle.load(file)

                    new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                            country, float(customer), int(product_ref), item_type, status]])
                    new_sample_itemtype = itemtype.transform(new_sample[:, [7]]).toarray()
                    new_sample_status = status.transform(new_sample[:, [8]]).toarray()
                    new_sample = np.concatenate((new_sample[:, 0:7], new_sample_itemtype, new_sample_status), axis=1)
                    new_sample1 = scaler.transform(new_sample)
                    new_pred = model.predict(new_sample1)[0]
                    st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

            except ValueError:
                st.write("Please enter valid numeric values.")

with tab2:
    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])

        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        # Validate inputs
        if csubmit_button:
            try:
                c_q_tons = float(cquantity_tons)
                c_thk = float(cthickness)
                c_width = float(cwidth)
                c_customer = float(ccustomer)
                c_selling = float(cselling)
                
                if c_q_tons <= 0 or c_thk <= 0:
                    st.write("Since log(0) or log of negative values leads to infinity, don't give zero or negative values for Quantity Tons and Thickness.")
                else:
                    
                    import pickle
                    with open("source/clsmodel.pkl", 'rb') as file:
                        cmodel = pickle.load(file)

                    with open("source/cscaler.pkl", 'rb') as file:
                        cscaler = pickle.load(file)

                    with open("source/citemtype.pkl", 'rb') as file:
                        citem_type = pickle.load(file)

                    # Predict the status for a new sample
                    new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                            np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(cproduct_ref),
                                            citem_type]])
                    new_sample_itemtype = citem_type.transform(new_sample[:, [8]]).toarray()
                    new_sample = np.concatenate((new_sample[:, 0:8], new_sample_itemtype), axis=1)
                    new_sample = cscaler.transform(new_sample)
                    new_pred = cmodel.predict(new_sample)
                    if new_pred == 1:
                        st.write('## :green[The Status is Won] ')
                    else:
                        st.write('## :red[The status is Lost] ')
            except ValueError:
                st.write("Please enter valid numeric values.")
