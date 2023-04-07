from flask import Flask, request, jsonify
import streamlit as st

# Define the Flask app
app = Flask(__name__)

# Define the Flask routes
@app.route('/')
def home():
    return 'Hello, world!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Perform some processing on the data
    result = data['input'] + 5

    # Return the result as a JSON response
    return jsonify({'output': result})

# Define the Streamlit app
def main():
    st.title('My App')

    # Define the input widget
    input_value = st.number_input('Enter a number')

    # Send the input to the Flask app and get the response
    response = requests.post('http://localhost:5000/predict', json={'input': input_value})

    # Get the output from the Flask app
    output_value = response.json()['output']

    # Display the output in the Streamlit app
    st.write('Output:', output_value)

if __name__ == '__main__':
    # Start the Flask app in a separate thread
    app.run(debug=True, use_reloader=False)

    # Start the Streamlit app
    main()
