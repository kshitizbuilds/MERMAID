


import streamlit as st
import pandas as pd
import numpy as np
import io
import re

# -------------------------------------------
# Streamlit Page Setup
# -------------------------------------------
st.set_page_config(page_title="Excel Un-Merge Utility", layout="wide")
st.title("Excel Merged Cell Processor")
st.markdown("This tool prepares your workflow data by fixing two common Excel formatting issues:")
st.markdown("1. **Un-merging:** Fills down merged cells (e.g., in `CHAIN`).")
st.markdown("2. **Delimiter Cleaning:** Converts multi-line cell values (newline delimiter `\n`) into slash-separated values (`/`).")


# -------------------------------------------
# CORE LOGIC
# -------------------------------------------
@st.cache_data(show_spinner="Reading Excel file...")
def load_excel(uploaded_file):
    """Load Excel file once."""
    # CRITICAL: Use keep_default_na=False to prevent pandas from automatically converting 'N/A', 'NA', etc., 
    # which can interfere with explicit handling later if the data type is object (string).
    df = pd.read_excel(uploaded_file, keep_default_na=False) 
    
    # Ensure column names are clean and handle potential case differences
    df.columns = [str(col).strip() for col in df.columns]
    return df

def fill_down_merged_cells(df, column_name):
    """
    Fills NaN values in the specified column with the last valid observation forward.
    It aggressively converts common missing value strings (like 'N/A', 'nan', '') to pd.NA 
    before applying ffill, to prevent these strings from being propagated.
    """
    if column_name not in df.columns:
        st.warning(f"Internal error: Column '{column_name}' not found for filling.")
        return df
    
    # 1. Convert to string
    s_str = df[column_name].astype(str)
    
    # 2. Define Regex to identify common missing/empty values.
    # Pattern explanation:
    # ^\s*$ -> Matches empty string or only whitespace (e.g., "", "   ")
    # |      -> OR
    # ^(N\/A|NA|NONE|NAN|N\.A\.)$ -> Matches common missing strings exactly (after stripping), 
    #                                handling various punctuation/spellings.
    # Flags=re.IGNORECASE handles any casing (n/a, NaN, none, etc.)
    missing_pattern = r'^\s*$|^(N\/A|NA|NONE|NAN|N\.A\.)$'

    # 3. Identify indices where the string matches the missing value pattern
    # We strip() first to ensure the ^\s*$ pattern catches all whitespace-only cells.
    is_missing_str = s_str.str.strip().str.contains(missing_pattern, flags=re.IGNORECASE, regex=True, na=False)
    
    # 4. Also, explicitly check for actual pandas/numpy missing values (NaN, None, pd.NA)
    is_true_na = df[column_name].isna()
    
    # 5. Combine the checks (string match OR actual NA) and assign pd.NA 
    # This step is the fix: it turns the *string* "N/A" into a *missing value* pd.NA
    df.loc[is_missing_str | is_true_na, column_name] = pd.NA

    # 6. Use forward fill (ffill) to propagate the last valid value forward.
    # Since all missing string representations are now pd.NA, ffill will correctly skip them.
    df[column_name] = df[column_name].ffill()
    
    return df

def clean_delimiters(df, column_name):
    """Replaces newline characters (\n) with a standard slash delimiter (/) in the specified column."""
    if column_name not in df.columns:
        st.warning(f"Internal error: Column '{column_name}' not found for cleaning.")
        return df

    # Convert to string and replace newline characters (\n) with a slash (/)
    df[column_name] = df[column_name].astype(str).str.replace('\n', '/', regex=False).str.strip()
    
    return df


def to_excel_buffer(df):
    """Converts a pandas DataFrame to an Excel file stored in an in-memory buffer."""
    output = io.BytesIO()
    # Use default Excel writer engine (like openpyxl)
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False, sheet_name='Unmerged_Cleaned_Data')
    processed_data = output.getvalue()
    return processed_data


# -------------------------------------------
# UI/Interaction
# -------------------------------------------

uploaded_file = st.file_uploader("Upload Input Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Load data, preventing automatic NA conversion for better control
        df_input = load_excel(uploaded_file)
        
        # Display the first few rows of the raw data
        st.subheader("1. Raw Data Preview")
        st.dataframe(df_input.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Create a case-insensitive map of column names for safe defaulting
        col_map = {c.lower(): c for c in df_input.columns.tolist()}
        
        # 1. Determine SMART defaults for FFILL (Un-merging)
        ffill_defaults = []
        for expected in ['chain', 'batch','start time']:
            if expected in col_map:
                ffill_defaults.append(col_map[expected])
        
        # --- Multi-Column Selection for FFILL (Un-merging) ---
        column_to_ffill = st.multiselect(
            "A. Select columns with **Merged Cells** to 'fill down'. *Candidates (Chain, Batch, Successor) are pre-selected.*",
            options=df_input.columns.tolist(),
            default=ffill_defaults
        )

        # 2. Determine SMART defaults for Delimiter Cleaning by checking for '\n'
        clean_defaults = []
        df_str = df_input.astype(str) # Convert entire df to string for easy \n search
        
        for expected in ['predecessor']:
            if expected in col_map:
                col_name = col_map[expected]
                # Check if ANY cell in the column contains a newline character
                if df_str[col_name].str.contains('\n').any():
                    clean_defaults.append(col_name)

        # --- Multi-Column Selection for Delimiter Cleaning ---
        column_to_clean = st.multiselect(
            "B. Select columns with **Multi-line Dependencies** (newline delimiter) to convert to slash-separated values. *Automatically detected candidates are pre-selected.*",
            options=df_input.columns.tolist(),
            default=clean_defaults
        )

        generate_btn = st.button("🚀 Process and Generate Cleaned Excel")
        st.markdown("---")


        if generate_btn and (column_to_ffill or column_to_clean):
            with st.spinner(f"Processing..."):
                
                temp_df = df_input.copy()
                
                # Step 1: Perform the fill operation (Un-merging)
                for col in column_to_ffill:
                    temp_df = fill_down_merged_cells(temp_df, col)
                
                # Step 2: Clean Delimiters (Newline to Slash)
                for col in column_to_clean:
                    temp_df = clean_delimiters(temp_df, col)
                
                df_cleaned = temp_df

                if df_cleaned is not None:
                    
                    # Convert the DataFrame to an Excel file buffer
                    excel_buffer = to_excel_buffer(df_cleaned)
                    
                    st.success("Successfully processed and prepared your data for the graphing tool!")
                    
                    st.subheader("2. Cleaned Data Preview")
                    st.dataframe(df_cleaned.head(10), use_container_width=True)
                    
                    st.download_button(
                        label="⬇️ Download Cleaned Excel File",
                        data=excel_buffer,
                        file_name=f"cleaned_data_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        elif generate_btn and not (column_to_ffill or column_to_clean):
            st.warning("Please select at least one column for either un-merging or delimiter cleaning.")


    except Exception as e:
        # Re-raise the exception for debugging if it's not a known exception
        st.error("An unexpected error occurred during file processing. Please check the data format.")
        st.exception(e)