import os
import streamlit as st
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import json

# Import modelling scripts
import src.preprocessing as preprocessing
import src.scorer as scorer

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main():
    # Main container with background
    st.markdown(
        """
        <style>
        .main-container {
            background: linear-gradient(135deg, #6A82FB, #FC5C7D);
            padding: 2rem;
            border-radius: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Child container with white background and rounded corners
    st.markdown(
        """
        <div class="main-container">
            <div style="background-color: white; border-radius: 15px; padding: 2rem; text-align: center">
                <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Logo_%D0%9C%D0%A2%D0%A1_%282023%29.svg/400px-Logo_%D0%9C%D0%A2%D0%A1_%282023%29.svg.png", width=100, style="float: left"/>
                <h1>ML Ops. Homework</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Ensure the directories exist
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and allowed_file(uploaded_file.name):
        filename = secure_filename(uploaded_file.name)
        
        # Store imported file locally
        new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
        save_location = os.path.join('input', new_filename)
        
        with open(save_location, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the file
        input_df = preprocessing.import_data(save_location)
        preprocessed_df = preprocessing.run_preproc(input_df)
        submission, hist_visual, feature_importance = scorer.make_pred(preprocessed_df, save_location)
        output_filename = save_location.replace('input', 'output')
        submission.to_csv(output_filename, index=False)

        st.success(f"File predict successfully and saved to {output_filename}")

        # Provide download link for the output file
        with open(output_filename, "rb") as file:
            btn = st.download_button(
                label="Download Predict File",
                data=file,
                file_name=output_filename.split('/')[-1],
                mime='text/csv'
            )

        hist_output_path = 'output/prediction_histogram.png'
        hist_visual.savefig(hist_output_path)
        st.success(f"File prediction histogram successfully and saved to {hist_output_path}")

        # Provide download link for the histogram
        with open(hist_output_path, "rb") as file:
            btn = st.download_button(
                label="Download prediction histogram",
                data=file,
                file_name=os.path.basename(hist_output_path),
                mime="image/png"
            )
        
        feature_output_path = 'output/feature_importance.json'
        with open(feature_output_path, 'w') as json_file:
            json.dump(feature_importance, json_file, indent=4)
        
        st.success(f"File JSON with top 5 feature successfully and saved to {hist_output_path}")

        # Provide download link for feature top5
        with open(feature_output_path, "r") as file:
            btn = st.download_button(
                label="Download JSON with top 5 features",
                data=file,
                file_name=os.path.basename(feature_output_path),
                mime="application/json"
            )
        
        st.pyplot(hist_visual)
        st.json(feature_importance)

if __name__ == '__main__':
    main()
