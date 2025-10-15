import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
from concurrent.futures import ThreadPoolExecutor
from graphviz import Digraph, Source
import plotly.express as px

# -------------------------------------------
# GLOBAL SETUP
# -------------------------------------------

# Ensure Graphviz is available (works on Windows / Streamlit Cloud)
# This is required for the Batch Dependency Diagram Generator part.
GRAPHVIZ_PATHS = [
    r"./Graphviz/bin" 
]

for path in GRAPHVIZ_PATHS:
    if os.path.exists(path):
        os.environ["PATH"] += os.pathsep + path

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------
# Streamlit Page Setup
# -------------------------------------------
st.set_page_config(page_title="Workflow Processor & Diagram Generator", layout="wide")
st.title("⚙️ Excel Workflow Processor & Diagram Generator")
st.markdown("This tool first **cleans** your Excel data, then generates a **Dependency Summary** and **Diagram**.")
st.markdown("Required Columns: **Batch**, **Predecessor**, **Successor**.")
st.markdown("---")

# -------------------------------------------
# CORE CLEANUP LOGIC (From Script 1)
# -------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning Excel file...")
def process_excel_for_workflow(uploaded_file):
    """
    Loads, cleans (un-merge, clean delimiters), and returns the processed DataFrame.
    This replaces the 'load_excel' from both original scripts.
    """
    # CRITICAL: Use keep_default_na=False to prevent pandas from auto-converting strings
    df = pd.read_excel(uploaded_file, keep_default_na=False) 
    
    # 1. Standardize column names
    df.columns = [str(col).strip() for col in df.columns]

    # Create a case-insensitive map of column names
    found_cols_lower = {col.lower(): col for col in df.columns}
    
    # Identify default columns for FFILL (Un-merging)
    ffill_cols = [
        found_cols_lower[expected] 
        for expected in ['chain', 'batch', 'start time'] 
        if expected in found_cols_lower
    ]
    
    # Identify default columns for DELIMITER CLEANING
    clean_cols = [
        found_cols_lower['predecessor']
        for expected in ['predecessor'] 
        if expected in found_cols_lower
    ]

    # --- Apply Cleaning ---
    df_cleaned = df.copy()

    # Apply Fill Down (Un-merging)
    for col in ffill_cols:
        if col in df_cleaned.columns:
            df_cleaned = fill_down_merged_cells(df_cleaned, col)

    # Apply Delimiter Cleaning (Newline to Slash)
    for col in clean_cols:
        if col in df_cleaned.columns:
            df_cleaned = clean_delimiters(df_cleaned, col)
            
    # CRITICAL: Post-cleaning aggressive NA handling for dependency logic
    for col in df_cleaned.columns:
         # Standardize Batch/Dependency columns to uppercase and clean up blanks/NAN/NONE
        if col.lower() in ['batch', 'predecessor', 'successor']:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.upper()
            df_cleaned.loc[df_cleaned[col].isin(['NAN', '', 'NONE', 'NA', 'N/A']), col] = np.nan
        else:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            df_cleaned.loc[df_cleaned[col].isin(['NAN', '', 'NONE', 'NA', 'N/A']), col] = np.nan


    return df_cleaned

def fill_down_merged_cells(df, column_name):
    """Fills NaN/missing-string values in the specified column with the last valid observation forward."""
    if column_name not in df.columns: return df
    
    s_str = df[column_name].astype(str)
    
    # Pattern to match empty/whitespace, or common missing strings
    missing_pattern = r'^\s*$|^(N\/A|NA|NONE|NAN|N\.A\.)$'

    is_missing_str = s_str.str.strip().str.contains(missing_pattern, flags=re.IGNORECASE, regex=True, na=False)
    is_true_na = df[column_name].isna()
    
    # Convert 'N/A' strings and NaNs to pd.NA
    df.loc[is_missing_str | is_true_na, column_name] = pd.NA

    # Use forward fill (ffill)
    df[column_name] = df[column_name].ffill()
    
    return df

def clean_delimiters(df, column_name):
    """Replaces newline characters (\n) with a standard slash delimiter (/) in the specified column."""
    if column_name not in df.columns: return df

    # Convert to string and replace newline characters (\n) with a slash (/)
    df[column_name] = df[column_name].astype(str).str.replace('\n', '/', regex=False).str.strip()
    
    return df

# -------------------------------------------
# CORE GENERATOR LOGIC (From Script 2) - FUNCTIONS
# -------------------------------------------
# Note: I kept the original Script 2 cached functions as they are robust.

@st.cache_data(show_spinner=False)
def get_dependencies(df, start_batch, col_map, depth=2):
    """Find all related predecessors and successors up to a depth."""
    df_clean = df.copy() 
    start_batch = start_batch.strip().upper() 

    all_related_batches = {start_batch}
    current_level = {start_batch}

    for _ in range(depth):
        next_level = set()

        # Predecessors: Find rows where the main batch is in the current level
        for _, row in df_clean[df_clean[col_map['batch']].isin(current_level)].iterrows():
            preds = str(row[col_map['predecessor']])
            if pd.notna(preds):
                # Filter out NAN/NONE/blanks
                next_level.update([p for p in preds.split('/') if p and p.upper() not in ['NAN', 'NONE']])

        # Successors (where current is in predecessor of another batch)
        for current_b in current_level:
            mask_pred = df_clean[col_map['predecessor']].astype(str).str.contains(
                r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
            )
            succs = df_clean[mask_pred][col_map['batch']].dropna().tolist()
            next_level.update(succs)

        # Successors (where current is in successor col)
        # This part seems redundant if the Predecessor column is the main link, but kept it for completeness from original code.
        for current_b in current_level:
            mask_succ = df_clean[col_map['successor']].astype(str).str.contains(
                r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
            )
            # Find the 'BATCH' names that have 'current_b' in their 'SUCCESSOR' column
            preds = df_clean[mask_succ][col_map['batch']].dropna().tolist()
            next_level.update(preds)

        new_batches = next_level - all_related_batches
        if not new_batches:
            break
        all_related_batches.update(new_batches)
        current_level = new_batches

    # Remove the placeholder missing values that might have crept in from dependencies 
    return sorted([b for b in list(all_related_batches) if b.upper() not in ['NAN', 'NONE']])

@st.cache_data(show_spinner=False)
def prepare_batch_list(df, col_map):
    """Collect all batch names from batch/predecessor/successor columns."""
    
    all_batches_raw = pd.concat([
        df[col_map['batch']].astype(str).dropna(),
        df[col_map['predecessor']].astype(str).str.split('/').explode().dropna(),
        df[col_map['successor']].astype(str).str.split('/').explode().dropna()
    ]).unique()
    
    # Final filter to remove common non-batch values (empty string, 'NAN', 'NONE')
    return sorted([
        b for b in all_batches_raw 
        if b and b.upper() not in ['NAN', 'NONE']
    ])


def calculate_dependencies(row, col_map, all_batches_df):
    """Calculates Pre/Post dependencies and counts for a single row (batch)."""
    current_batch = row[col_map['batch']]
    
    # 1. Pre Dependencies (What the current batch depends on)
    pre_deps_str = str(row.get(col_map['predecessor'], ''))
    pre_deps = [d for d in pre_deps_str.split('/') if d and d.upper() not in ['NAN', 'NONE']]
    
    # 2. Post Dependencies (What depends on the current batch)
    post_deps_str_succ = str(row.get(col_map['successor'], ''))
    post_deps_succ = [d for d in post_deps_str_succ.split('/') if d and d.upper() not in ['NAN', 'NONE']]
    
    # Successors (from the 'BATCH' column of other rows where current_batch is in PREDECESSOR)
    mask_pred = all_batches_df[col_map['predecessor']].astype(str).str.contains(
        r'(^|/)' + re.escape(current_batch) + r'(/|$)', regex=True
    )
    post_deps_pred = all_batches_df[mask_pred][col_map['batch']].dropna().tolist()
    
    # Combine and get unique list of Post Dependencies
    all_post_deps = sorted(list(set(post_deps_succ + post_deps_pred)))
    
    return pd.Series([
        '/'.join(pre_deps), 
        len(pre_deps), 
        '/'.join(all_post_deps), 
        len(all_post_deps)
    ])


def get_mapped_columns(col_map):
    """Helper to map col_map keys to standard headers for output."""
    mapping = {}
    standard_names = {
        'time': 'START TIME', 'chain': 'CHAIN', 'batch': 'BATCH', 
        'predecessor': 'PREDECESSOR', 'successor': 'SUCCESSOR', 'path': 'PATH', 
        'condition': 'CONDITION', 'param1': 'PARAMETER 1', 'param2': 'PARAMETER 2', 
        'param3': 'PARAMETER 3'
    }
    for key, name in col_map.items():
        if key in standard_names:
            mapping[standard_names[key]] = name
    return mapping


@st.cache_data(show_spinner=False)
def create_raw_dependencies_table(filtered_df, col_map):
    """Generates the DataFrame for the "Raw Dependencies Table" (Table 1)."""
    mapped_cols = get_mapped_columns(col_map)
    output_column_order = [
        'CHAIN', 'BATCH', 'PREDECESSOR', 'SUCCESSOR', 'START TIME', 
        'CONDITION', 'PATH', 'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
    ]

    final_df = filtered_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()
    output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    output_df = output_df.replace('nan', '', regex=True).replace('NONE', '', regex=True)
    
    return output_df


@st.cache_data(show_spinner=False)
def create_dependency_summary_table(filtered_df, df_full_for_lookup, col_map):
    """Generates the final DataFrame for the "SUMMARY TABLE" (Table 2)."""
    
    mapped_cols = get_mapped_columns(col_map)
    dependencies_cols = ['PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 'POST DEPENDENCY', 'NO OF POST DEPENDENCIES']
    
    temp_df = filtered_df.copy() 
    temp_df[dependencies_cols] = temp_df.apply(
        calculate_dependencies, 
        axis=1, 
        result_type='expand', 
        args=(col_map, df_full_for_lookup) # df_full_for_lookup is the fully cleaned DF
    )
    
    final_df = temp_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()

    output_column_order = [
        'CHAIN', 'BATCH', 
        'PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 
        'POST DEPENDENCY', 'NO OF POST DEPENDENCIES',  
        'START TIME', 'PATH', 'CONDITION', 
        'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
    ]

    output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    output_df = output_df.replace('nan', '', regex=True).replace('NONE', '', regex=True)
    
    return output_df


@st.cache_data(show_spinner=False)
def generate_graphviz_source(filtered_df, col_map, selected_batch, high_res=False):
    """Generate and return Graphviz source code string, including time/frequency."""
    
    time_col_name = col_map.get('time') 
    
    graph_attrs = {
        'rankdir': 'TB',
        'splines': 'curved', 
        'nodesep': '0.5', 'ranksep': '0.75',
        'fontname': 'Helvetica'
    }
    
    if high_res:
        graph_attrs['size'] = '50,50!' 
        graph_attrs['dpi'] = '300' 
    else:
        graph_attrs['size'] = '15,10!'
        
    dot = Digraph(
        comment='Batch Workflow',
        format='png',
        graph_attr=graph_attrs,
        node_attr={'fontname': 'Helvetica', 'shape': 'box', 'style': 'rounded'},
        edge_attr={'fontname': 'Helvetica'}
    )
    
    all_nodes = set()
    for col_key in ['batch', 'predecessor', 'successor']:
        col = col_map.get(col_key)
        if col and col in filtered_df.columns:
            if col_key in ['predecessor', 'successor']:
                deps = filtered_df[col].astype(str).str.split('/').explode().dropna()
                all_nodes.update(deps.unique())
            else:
                all_nodes.update(filtered_df[col].astype(str).dropna().unique())

    # Map for quick lookup of time/frequency
    time_map = filtered_df.set_index(col_map['batch'])[time_col_name].to_dict() if time_col_name and col_map.get('batch') in filtered_df.columns else {}
    
    # 1. Create Nodes
    selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— SHOW ALL BATCH —" else selected_batch

    for batch_str in all_nodes:
        if not batch_str or batch_str.upper() in ['NAN', 'NONE']:
            continue
            
        time_data = time_map.get(batch_str)
        time_label = ""
        if time_data and pd.notna(time_data):
            time_label = "\n" + str(time_data).replace("/", "\n") 
        
        label_text = batch_str + time_label
        
        node_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'white'}
        
        # Highlight the selected batch
        if selected_batch != "— Show All Batch —" and batch_str == selected_batch_upper:
            node_attrs['fillcolor'] = '#87CEFA'
            
        # Distinguish batches present in the data vs. external dependencies
        if col_map.get('batch') in filtered_df.columns and batch_str in filtered_df[col_map['batch']].values:
            dot.node(batch_str, label=label_text, **node_attrs)
        else:
            dot.node(batch_str, label=label_text, shape='box', style='rounded,filled', fillcolor='#F0F8FF')


    # 2. Create Edges
    added_edges = set() 
    
    for _, row in filtered_df.iterrows():
        batch_str = str(row[col_map['batch']])
        
        for cell_key, direction in [('predecessor', 'pred'), ('successor', 'succ')]:
            cell = col_map.get(cell_key)
            if not cell or cell not in filtered_df.columns: continue
            
            deps = str(row[cell])
            if not deps or deps.upper() in ['NAN', 'NONE']:
                continue
            
            for dep in [d for d in deps.split('/') if d and d.upper() not in ['NAN', 'NONE']]:
                if dep in all_nodes: 
                    
                    if direction == 'pred':
                        source_node = dep
                        target_node = batch_str
                    else:
                        source_node = batch_str
                        target_node = dep
                        
                    edge_key = f"{source_node}->{target_node}"
                    
                    if edge_key not in added_edges:
                        dot.edge(source_node, target_node)
                        added_edges.add(edge_key)

    return dot.source

# -------------------------------------------
# MAIN UI / EXECUTION
# -------------------------------------------

uploaded_file = st.sidebar.file_uploader("Upload Input Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Step 1: Load and Clean the Data
        # This function encapsulates all the logic from the first script
        df = process_excel_for_workflow(uploaded_file)
        
        # Determine Column Mapping (similar to start of script 2)
        found_cols_lower = {col.lower(): col for col in df.columns}
        required_cols = ['batch', 'predecessor', 'successor']
        
        col_map = {c: found_cols_lower[c] for c in required_cols if c in found_cols_lower}
        
        optional_cols_map = {
            'chain': 'chain', 
            'path': 'path', 
            'condition': 'condition',
            'param1': 'parameter 1', 
            'param2': 'parameter 2', 
            'param3': 'parameter 3'
        }
        
        # Find time/frequency column
        time_col_key = next((k for k in found_cols_lower if 'time' in k or 'frequency' in k), None)
        if time_col_key:
            col_map['time'] = found_cols_lower[time_col_key]
        else:
            col_map['time'] = None

        # Map other optional columns if they exist
        for key, name_part in optional_cols_map.items():
            if key not in col_map: 
                found_col_name = next((v for k, v in found_cols_lower.items() if name_part in k), None)
                if found_col_name:
                    col_map[key] = found_col_name
            
        # Final check for mandatory columns
        if not all(col in col_map for col in required_cols):
            st.error("The uploaded file must contain columns named **Batch**, **Predecessor**, and **Successor**.")
            st.stop()
            
        # --- UI FILTERING CONTROLS (Script 2 UI) ---
        chain_name_col = col_map.get('chain')
        selected_chain = "— All Chains —"
        
        temp_df_for_batches = df.copy() # Use the cleaned DF

        if chain_name_col:
            # Re-ffill the Chain Column if it was mapped, just in case the cleaner didn't catch it
            df[chain_name_col] = df[chain_name_col].ffill()
            temp_df_for_batches[chain_name_col] = temp_df_for_batches[chain_name_col].ffill()
            
            all_chains = sorted(df[chain_name_col].astype(str).str.strip().dropna().unique())
            selected_chain = st.sidebar.selectbox("Filter by Chain", ["— All Chains —"] + all_chains)
            
            if selected_chain != "— All Chains —":
                temp_df_for_batches = temp_df_for_batches[temp_df_for_batches[chain_name_col] == selected_chain].copy()

        all_batches = prepare_batch_list(temp_df_for_batches, col_map)
        batch_options = ["— Show All Batch —"] + all_batches
        selected_batch = st.sidebar.selectbox("Select Batch to Map Around", batch_options)

        dependency_depth = st.sidebar.slider("Diagram Dependency Depth", 1, 5, 1) 
        generate_btn = st.sidebar.button("Generate Summary & Diagram")

        st.sidebar.markdown("---")
        # Display the initial data processing result/download (from Script 1)
        st.sidebar.caption("Data Cleanup Status: **Complete**")
        
        # Download the intermediate cleaned file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer) as writer:
            df.to_excel(writer, index=False, sheet_name='Unmerged_Cleaned_Data')
        
        st.sidebar.download_button(
            label="⬇️ Download Cleaned Excel File",
            data=excel_buffer.getvalue(),
            file_name=f"CLEANED_WORKFLOW_DATA_{uploaded_file.name}",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        
        # --- GENERATION EXECUTION (Script 2 Execution) ---
        if generate_btn:
            with st.spinner("🔄 Generating Dependencies and Diagram..."):
                
                selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— Show All Batch —" else selected_batch
                
                if selected_batch != "— Show All Batch —":
                    related_batches = get_dependencies(df, selected_batch_upper, col_map, dependency_depth)
                    filtered_df = df[df[col_map['batch']].astype(str).isin(related_batches)].copy()
                    st.subheader(f"Workflow for: **{selected_batch_upper}** (Depth {dependency_depth})")
                elif selected_chain != "— All Chains —":
                    filtered_df = temp_df_for_batches.copy()
                    st.subheader(f"Full Workflow for Chain: **{selected_chain}**")
                else:
                    filtered_df = df.copy()
                    st.subheader("Full Workflow (All Batch and All Chains)")

                if filtered_df.empty:
                    st.warning("No batches or connections found. Try a different batch, a greater depth, or check your chain filter.")
                else:
                    
                    # --- TABLE 2: SUMMARY TABLE --- (Generates calculated pre/post counts)
                    summary_table_df = create_dependency_summary_table(
                        filtered_df.copy(), 
                        df,        # Pass the full cleaned DF for accurate POST DEPENDENCY lookup
                        col_map
                    )
                    
                    # --- DEPENDENCY INSIGHTS (Metrics & Chart) ---
                    st.markdown("---")
                    st.subheader("🚀 Dependency Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    total_batches = len(summary_table_df)
                    total_pre_deps = summary_table_df['NO OF PREDEPENDENCY'].sum()
                    total_post_deps = summary_table_df['NO OF POST DEPENDENCIES'].sum()
                    
                    col1.metric("Total Batches Analyzed", total_batches)
                    col2.metric("Total Pre Dependencies", total_pre_deps)
                    col3.metric("Total Post Dependencies", total_post_deps)

                    if 'NO OF POST DEPENDENCIES' in summary_table_df.columns:
                        top_dependents = summary_table_df.sort_values(
                            'NO OF POST DEPENDENCIES', ascending=False
                        ).head(10)

                        if not top_dependents.empty:
                            fig = px.bar(
                                top_dependents,
                                x='BATCH',
                                y='NO OF POST DEPENDENCIES',
                                title='Top 10 Most Critical Batches (Highest Post Dependencies)',
                                labels={'NO OF POST DEPENDENCIES': 'Number of Batches Dependent On It', 'BATCH': 'Batch Name'},
                                height=400
                            )
                            fig.update_layout(xaxis={'categoryorder':'total descending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                    st.markdown("---")
                    
                    # --- TABLE 1: RAW DEPENDENCIES TABLE ---
                    raw_deps_df = create_raw_dependencies_table(filtered_df.copy(), col_map)
                    
                    st.subheader("📋 RAW DEPENDENCIES TABLE (Source View)")
                    st.dataframe(raw_deps_df, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("📊 SUMMARY TABLE (Calculated Dependencies)")
                    st.dataframe(summary_table_df, use_container_width=True)
                    st.markdown("---")
                    
                    # --- GRAPH GENERATION ---
                    future_std = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=False)
                    dot_source_std = future_std.result()
                    
                    future_high_res = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=True)
                    dot_source_high_res = future_high_res.result()

                    st.subheader("🔗 Batch Dependency Diagram")
                    st.graphviz_chart(dot_source_std)
                    
                    # --- Download buttons for Diagrams ---
                    col_dl_std, col_dl_high = st.columns(2)
                    
                    png_data_std = Source(dot_source_std).pipe(format='png')
                    with col_dl_std:
                        st.download_button(
                            "📥 Download Standard Resolution (PNG)",
                            data=png_data_std,
                            file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_std.png",
                            mime="image/png"
                        )
                    
                    png_data_high_res = Source(dot_source_high_res).pipe(format='png')
                    with col_dl_high:
                        st.download_button(
                            "💾 Download High Resolution Big (PNG)",
                            data=png_data_high_res,
                            file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_high_res.png",
                            mime="image/png"
                        )
                    
                    with st.expander("View Filtered Raw Data & Graphviz Source"):
                        st.dataframe(filtered_df)
                        st.code(dot_source_std)

    except Exception as e:
        st.error("An error occurred during file processing or generation. Please check the data format and required columns.")
        st.exception(e)