import streamlit as st
from main import process_input , execute_chain

# Set up the Streamlit app
st.title("Task Executor")

# Create a text input field for the prompt
user_input = st.text_input("Enter your prompt:", "")

user_input = {
    "prompt": user_input
}


# Create an execution button
if st.button("Execute"):
    if user_input:
        # Call the function from the external file
        result_chain = process_input(user_input)
        # Display the result
        st.write("Result:", result_chain)
        if not result_chain:
            print("No valid function chain generated.")
        else:
            print(f"Generated chain: {result_chain}")
            final_output = execute_chain(result_chain)
            print(f"\nFinal result:\n{final_output}")
    else:
        st.warning("Please enter a prompt before executing.")