import streamlit as st
import subprocess

# Set page configuration
st.set_page_config(
    page_title="Attendance Management System",
    page_icon="📊",
    layout="wide"
)

# CSS for styling the square options
st.markdown("""
    <style>
    .option-card {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 150px;
        width: 150px;
        background-color: #f0f2f6;
        border-radius: 10px;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s;
        margin: 10px;
    }
    .option-card:hover {
        transform: scale(1.05);
        background-color: #e0e2e6;
    }
    .option-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to handle redirection and execution
def handle_option_click(option):
    if option == "New User Sign In":
        st.experimental_set_query_params(page='new_user_sign_in')
        st.experimental_rerun()
    elif option == "Mark Attendance":
        st.experimental_set_query_params(page='mark_attendance')
        st.experimental_rerun()
    elif option == "Show Today's Attendance":
        st.experimental_set_query_params(page='show_todays_attendance')
        st.experimental_rerun()
    elif option == "Attendance Analytics":
        st.experimental_set_query_params(page='attendance_analytics')
        st.experimental_rerun()

# Create the options as clickable cards
options = [
    ("New User Sign In", "🔑"),
    ("Mark Attendance", "📝"),
    ("Show Today's Attendance", "📅"),
    ("Attendance Analytics", "📊")
]

# Layout for the options
st.markdown("<h1 style='text-align: center;'>Attendance Management System</h1>", unsafe_allow_html=True)
st.markdown("<div class='option-container'>", unsafe_allow_html=True)

cols = st.columns(2)
index = 0

for option, icon in options:
    if st.button(f"{icon} {option}", key=option):
        handle_option_click(option)

st.markdown("</div>", unsafe_allow_html=True)

# Handle different pages
page = st.experimental_get_query_params().get('page', [''])[0]

if page == 'new_user_sign_in':
    # Execute app.py for New User Sign In
    subprocess.Popen(["streamlit", "run", "newuser.py"])
    st.stop()
elif page == 'mark_attendance':
    subprocess.Popen(["streamlit", "run", "Mark_Attendance.py"])
    st.stop()
elif page == 'show_todays_attendance':
    st.title("Show Today's Attendance")
    st.write("This is where today's attendance will be displayed.")
elif page == 'attendance_analytics':
    st.title("Attendance Analytics")
    st.write("This is where attendance analytics will be shown.")
else:
    st.write("Select an option to get started.")
