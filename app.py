# app.py
#    miiii streamlit app
# by: Noah Syrkis

# imports
import streamlit as st


# constants
title = "miiii | mechanistic interpreability on irriducible integer identifiers"
st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")


# main function
def main():
    body_fn()


def body_fn():
    st.title(title)
    st.write()


# run main
if __name__ == "__main__":
    main()
