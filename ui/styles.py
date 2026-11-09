import streamlit as st


def load_styles():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
