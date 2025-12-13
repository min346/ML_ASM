# export_models.py
# Run this in the SAME folder as your trained notebook output (or adapt paths).
# Purpose: save q_tables and dq_combined_tables into pickles for Streamlit app.

import pickle

# Expect these variables already exist in your Python session:
#   q_tables
#   dq_combined_tables
#
# Example (inside notebook):
#   !python export_models.py

with open("q_tables.pkl", "wb") as f:
    pickle.dump(q_tables, f)

with open("dq_combined_tables.pkl", "wb") as f:
    pickle.dump(dq_combined_tables, f)

print("Saved: q_tables.pkl, dq_combined_tables.pkl")
