import streamlit as st, json
from pathlib import Path
DB = Path(__file__).parents[1] / "demo_db.json"

st.header("⚙️ Admin Settings (Demo)")

if st.button("RESET DEMO DB"):
    DB.unlink(missing_ok=True)
    st.warning("Database reset. Restart the app.")
