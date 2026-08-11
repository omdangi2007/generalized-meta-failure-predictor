import streamlit as st


def show_reasons(reasons):

    st.subheader("💡 Why this prediction?")

    for reason in reasons:

        st.markdown(
            f"""
            <div class="reason-card">
                ✅ {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )