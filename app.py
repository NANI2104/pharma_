from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained models
best_rf = joblib.load('models/best_rf_ctt_fp.pkl')
best_model_fp_fsi = joblib.load('models/best_model_fp_fsi.pkl')
best_rf_model = joblib.load('models/best_rf_model (1).pkl')
best_xgb_model = joblib.load('models/best_xgb_model (1).pkl')

# Define columns for both model sets
columns1 = pd.Index([
    'DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience',
    'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
    'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
    'New Indication_No', 'New Indication_Yes',
    'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
    'PediatricOnly_No', 'PediatricOnly_Yes'
])

columns2 = pd.Index([
    'Country_Argentina', 'Country_Australia', 'Country_Brazil', 'Country_Canada',
    'Country_China', 'Country_France', 'Country_India', 'Country_Italy',
    'Country_Japan', 'Country_South Africa', 'Country_Spain', 'Country_UK',
    'Country_USA', 'DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience',
    'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
    'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
    'New Indication_No', 'New Indication_Yes',
    'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
    'PediatricOnly_No', 'PediatricOnly_Yes'
])

countries = [
    'Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France',
    'India', 'Italy', 'Japan', 'South Africa', 'Spain', 'UK', 'USA'
]

def convert_to_array(user_input, columns):
    input_array = [0] * len(columns)
    for key, value in user_input.items():
        column_name = f"{key}_{value}"
        if column_name in columns:
            index = columns.get_loc(column_name)
            input_array[index] = 1
    return input_array

def convert_to_dataframe(user_input, columns):
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    for key, value in user_input.items():
        column_name = f"{key}_{value}"
        if column_name in columns:
            input_df.at[0, column_name] = 1
        else:
            print(f"Column '{column_name}' not found in columns")
    return input_df

def add_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    return df[columns]

@app.route('/')
def main():
    return render_template('mainpage.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_input = {
            'Country': request.form['country'],
            'DevelopmentUnit': request.form['development_unit'],
            'Phase': request.form['phase'],
            'New Indication': request.form['new_indication'],
            'Blinding': request.form['blinding'],
            'Pediatric': request.form['pediatric']
        }

        # Create separate dictionaries for CTT-FP and FP-FSI
        user_input_CTT_FP = {
            'DevelopmentUnit': user_input['DevelopmentUnit'],
            'Phase': user_input['Phase'],
            'New Indication': user_input['New Indication'],
            'Blinding': user_input['Blinding'],
            'Pediatric': user_input['Pediatric']
        }

        user_input_FP_FSI = {
            'Country': user_input['Country'],
            'DevelopmentUnit': user_input['DevelopmentUnit'],
            'Phase': user_input['Phase'],
            'New Indication': user_input['New Indication'],
            'Blinding': user_input['Blinding'],
            'Pediatric': user_input['Pediatric']
        }

        # Convert user input to arrays
        input_array1 = convert_to_array(user_input_CTT_FP, columns1)
        input_array2 = convert_to_array(user_input_FP_FSI, columns2)

        # Convert the 1D arrays to 2D arrays
        input_2d_array1 = np.array(input_array1).reshape(1, -1)
        input_2d_array2 = np.array(input_array2).reshape(1, -1)

        # Predict the weeks
        y_pred1 = best_rf.predict(input_2d_array1)[0]
        y_pred2 = best_model_fp_fsi.predict(input_2d_array2)[0]
        total_weeks = y_pred1 + y_pred2

        result = {
            'ctt_fp': f"{y_pred1:.2f}",
            'fp_fsi': f"{y_pred2:.2f}",
            'total_weeks': f"{total_weeks:.2f}"
        }

    return render_template('index.html', result=result)

@app.route('/continue')
def continue_prediction():
    return redirect(url_for('index2'))

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed")
    development_unit = request.form['developmentUnit']
    phase = request.form['phase']
    new_indication = request.form['newIndication']
    blinding = request.form['blinding']
    pediatric = request.form['pediatric']

    user_input_CTT_FP = {
        'DevelopmentUnit': development_unit,
        'Phase': phase,
        'New Indication': new_indication,
        'Blinding': blinding,
        'PediatricOnly': pediatric
    }

    results = []
    for country in countries:
        user_input_FP_FSI = {
            'Country': country,
            'DevelopmentUnit': development_unit,
            'Phase': phase,
            'New Indication': new_indication,
            'Blinding': blinding,
            'PediatricOnly': pediatric
        }

        input_df1 = convert_to_dataframe(user_input_CTT_FP, columns1)
        input_df2 = convert_to_dataframe(user_input_FP_FSI, columns2)

        input_df1 = add_missing_columns(input_df1, columns1)
        input_df2 = add_missing_columns(input_df2, columns2)

        y_pred1 = best_rf_model.predict(input_df1)
        y_pred2 = best_xgb_model.predict(input_df2)

        rounded_y_pred1 = round(y_pred1[0])
        rounded_y_pred2 = round(y_pred2[0])
        total_weeks = rounded_y_pred1 + rounded_y_pred2
        results.append((country, total_weeks))

    results.sort(key=lambda x: x[1], reverse=True)
    top_5_results = results[:5]
    print(results)
    return render_template('index2.html', results=top_5_results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
