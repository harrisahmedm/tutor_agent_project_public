import streamlit as st
from scan_student_solution import scan_student_solution
from tutor_agent import tutor_step

st.title("Tutor Agent for Student Assistance")

uploaded = st.file_uploader("Upload your solution", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded)
    parsed = scan_student_solution(uploaded.read())
    st.write("Parsed steps:", parsed)

    if st.button("Ask Tutor"):
        reply = tutor_step(parsed)
        st.write("Tutor:", reply)
