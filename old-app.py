


# !===================== ! VERSION 3 2024-06-12 tested  =======================



# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from concurrent.futures import ThreadPoolExecutor
# from graphviz import Digraph, Source

# # Ensure Graphviz is available (works on Windows / Streamlit Cloud)
# GRAPHVIZ_PATHS = [
#     r"./Graphviz/bin" 
# ]

# for path in GRAPHVIZ_PATHS:
#     if os.path.exists(path):
#         os.environ["PATH"] += os.pathsep + path


# # -------------------------------------------
# # Streamlit Page Setup
# # -------------------------------------------
# st.set_page_config(page_title="Batch Workflow Diagram Generator", layout="wide")
# st.title("🔗 Batch Dependency Diagram")
# st.markdown("Upload your Excel file (must contain **Batch**, **Predecessor**, **Successor** columns).")

# executor = ThreadPoolExecutor(max_workers=2)

# # -------------------------------------------
# # CACHED FUNCTIONS (These use the pre-cleaned data, so they remain efficient)
# # -------------------------------------------

# @st.cache_data(show_spinner=False)
# def load_excel(uploaded_file):
#     """Load Excel file once."""
#     df = pd.read_excel(uploaded_file)
#     df.columns = [col.strip() for col in df.columns]
#     return df

# @st.cache_data(show_spinner=False)
# def get_dependencies(df, start_batch, col_map, depth=2):
#     """
#     Find all related predecessors and successors up to a depth.
#     The input DF is assumed to have already been cleaned (stripped/uppercased).
#     """
#     df_clean = df.copy() 
#     # start_batch is already stripped/uppercased by the calling function
#     start_batch = start_batch.strip().upper() 

#     all_related_batches = {start_batch}
#     current_level = {start_batch}

#     for _ in range(depth):
#         next_level = set()

#         # Predecessors: Find rows where the main batch is in the current level
#         for _, row in df_clean[df_clean[col_map['batch']].isin(current_level)].iterrows():
#             preds = str(row[col_map['predecessor']])
#             if pd.notna(preds):
#                 # The dependency columns are already stripped/uppercased in the main block
#                 next_level.update([p for p in preds.split('/') if p])

#         # Successors (where current is in predecessor of another batch)
#         for current_b in current_level:
#             # We look in the Predecessor column for the current batch
#             mask_pred = df_clean[col_map['predecessor']].astype(str).str.contains(
#                 r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
#             )
#             succs = df_clean[mask_pred][col_map['batch']].tolist()
#             next_level.update(succs)

#         # Successors (where current is in successor col)
#         for current_b in current_level:
#             # We look in the Successor column for the current batch
#             mask_succ = df_clean[col_map['successor']].astype(str).str.contains(
#                 r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
#             )
#             # The 'Batch' column entry for these rows are the predecessors/successors of 'current_b'
#             preds = df_clean[mask_succ][col_map['batch']].tolist()
#             next_level.update(preds)

#         new_batches = next_level - all_related_batches
#         if not new_batches:
#             break
#         all_related_batches.update(new_batches)
#         current_level = new_batches

#     return list(all_related_batches)

# @st.cache_data(show_spinner=False)
# def prepare_batch_list(df, col_map):
#     """Collect all batch names from batch/predecessor/successor columns."""
    
#     # All relevant columns are now already stripped/uppercased/NaN-filled by the main block
    
#     all_batches_raw = pd.concat([
#         df[col_map['batch']].astype(str).dropna(),
#         df[col_map['predecessor']].astype(str).str.split('/').explode().dropna(),
#         df[col_map['successor']].astype(str).str.split('/').explode().dropna()
#     ]).unique()
    
#     # Final filter to remove common non-batch values (empty string, 'NAN', 'NONE')
#     return sorted([
#         b for b in all_batches_raw 
#         if b and b.upper() not in ['NAN', 'NONE']
#     ])


# @st.cache_data(show_spinner=False)
# def generate_graphviz_source(filtered_df, col_map, selected_batch, high_res=False):
#     """Generate and return Graphviz source code string, including time/frequency."""
    
#     time_col_name = col_map.get('time') 
    
#     # --- Graphviz Attributes (Curvy Lines) ---
#     graph_attrs = {
#         'rankdir': 'TB',
#         'splines': 'curved', 
#         'nodesep': '0.5', 'ranksep': '0.75',
#         'fontname': 'Helvetica'
#     }
    
#     if high_res:
#         graph_attrs['size'] = '50,50!' 
#         graph_attrs['dpi'] = '300' 
#     else:
#         graph_attrs['size'] = '15,10!'
        
#     dot = Digraph(
#         comment='Batch Workflow',
#         format='png',
#         graph_attr=graph_attrs,
#         node_attr={'fontname': 'Helvetica', 'shape': 'box', 'style': 'rounded'},
#         edge_attr={'fontname': 'Helvetica'}
#     )
#     # ----------------------------------------------------------------
    
#     # Get all nodes that should be in the graph (Batch column values + all dependencies)
#     all_nodes = set()
#     for col in [col_map['batch'], col_map['predecessor'], col_map['successor']]:
#         if col in [col_map['predecessor'], col_map['successor']]:
#             deps = filtered_df[col].astype(str).str.split('/').explode().dropna()
#             all_nodes.update(deps.unique())
#         else:
#             all_nodes.update(filtered_df[col].astype(str).dropna().unique())

#     # Map for quick lookup of time/frequency
#     time_map = filtered_df.set_index(col_map['batch'])[time_col_name].to_dict() if time_col_name else {}
    
#     # 1. Create Nodes
#     selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— SHOW ALL BATCH —" else selected_batch

#     for batch_str in all_nodes:
#         if not batch_str or batch_str.upper() in ['NAN', 'NONE']:
#             continue
            
#         time_data = time_map.get(batch_str)
#         time_label = ""
#         if time_data and pd.notna(time_data):
#             # This column was not uppercased, so we show it as is
#             time_label = "\n" + str(time_data).replace("/", "\n") 
        
#         label_text = batch_str + time_label
        
#         node_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'white'}
        
#         # Check against the cleaned/uppercased selected batch
#         if selected_batch != "— Show All Batch —" and batch_str == selected_batch_upper:
#             node_attrs['fillcolor'] = '#87CEFA'
            
#         if batch_str in filtered_df[col_map['batch']].values:
#             dot.node(batch_str, label=label_text, **node_attrs)
#         else:
#             dot.node(batch_str, label=label_text, shape='box', style='rounded,filled', fillcolor='#F0F8FF')


#     # 2. Create Edges
#     added_edges = set() 
    
#     for _, row in filtered_df.iterrows():
#         batch_str = str(row[col_map['batch']])
        
#         for cell, direction in [(col_map['predecessor'], 'pred'), (col_map['successor'], 'succ')]:
#             deps = str(row[cell])
#             if not deps or deps.upper() == 'NAN':
#                 continue
            
#             # Dependencies are already clean
#             for dep in [d for d in deps.split('/') if d]:
#                 if dep in all_nodes: 
                    
#                     if direction == 'pred':
#                         source_node = dep
#                         target_node = batch_str
#                     else:
#                         source_node = batch_str
#                         target_node = dep
                        
#                     edge_key = f"{source_node}->{target_node}"
                    
#                     if edge_key not in added_edges:
#                         dot.edge(source_node, target_node)
#                         added_edges.add(edge_key)

#     return dot.source

# # -------------------------------------------
# # MAIN UI
# # -------------------------------------------

# uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# if uploaded_file:
#     try:
#         df = load_excel(uploaded_file)
#         found_cols_lower = {col.lower(): col for col in df.columns}
#         required_cols = ['batch', 'predecessor', 'successor']

#         if not all(col in found_cols_lower for col in required_cols):
#             st.error("Excel must contain columns: **Batch**, **Predecessor**, **Successor**")
#             st.stop()

#         col_map = {c: found_cols_lower[c] for c in required_cols}
        
#         # --- LOGIC: Identify Time/Frequency Column ---
#         time_col_key = next((k for k in found_cols_lower if 'time' in k or 'frequency' in k), None)
#         if time_col_key:
#             col_map['time'] = found_cols_lower[time_col_key]
#             st.sidebar.info(f"Using time/frequency column: **{col_map['time']}**")
#         else:
#             col_map['time'] = None
#             st.sidebar.warning("Could not find a 'time' or 'frequency' column. Diagram nodes will only show batch names.")
#         # ------------------------------------------------
        
        
#         # --- AGGRESSIVE CLEANING FOR DEPENDENCY AND TIME COLUMNS ---
#         # Do this cleaning once, up front, on the whole DF.
#         for col_key, col_name in col_map.items():
#             if col_name is not None:
#                 # 1. Convert to string
#                 df[col_name] = df[col_name].astype(str)
                
#                 # 2. Uppercase and strip for reliable matching in get_dependencies
#                 if col_key in ['batch', 'predecessor', 'successor']:
#                     df[col_name] = df[col_name].str.strip().str.upper()
#                 # Time column is not uppercased, but is stripped and converted to nan for consistency
#                 elif col_key == 'time':
#                     df[col_name] = df[col_name].str.strip()
                
#                 # 3. Replace 'NAN', 'NONE', and empty strings with NaN for proper filtering/ffill
#                 df.loc[df[col_name].isin(['NAN', '', 'NONE']), col_name] = np.nan
#         # --------------------------------------------------


#         chain_name_col = found_cols_lower.get('chain')
#         selected_chain = "— All Chains —"
        
#         # Create a temporary DF for the batch list dropdown based on chain filter
#         temp_df_for_batches = df.copy()

#         if chain_name_col:
#             # --- CRITICAL FIX 2: Clean and Ffill the Chain Column ---
            
#             # 1. Strip the chain column string (Crucial if the Excel cell has "  ChainName  ")
#             df[chain_name_col] = df[chain_name_col].astype(str).str.strip()
#             temp_df_for_batches[chain_name_col] = temp_df_for_batches[chain_name_col].astype(str).str.strip()
            
#             # 2. Convert empty strings (from the strip) to NaN
#             df.loc[df[chain_name_col] == '', chain_name_col] = np.nan
#             temp_df_for_batches.loc[temp_df_for_batches[chain_name_col] == '', chain_name_col] = np.nan
            
#             # 3. Forward-fill the chain column (This fills the blank rows)
#             df[chain_name_col] = df[chain_name_col].ffill()
#             temp_df_for_batches[chain_name_col] = temp_df_for_batches[chain_name_col].ffill()
#             # ------------------------------------------------------
            
#             # Now, generate the list of chains from the *cleaned* column
#             all_chains = sorted(df[chain_name_col].astype(str).str.strip().dropna().unique())
#             selected_chain = st.sidebar.selectbox("Filter by Chain", ["— All Chains —"] + all_chains)
            
#             if selected_chain != "— All Chains —":
#                 # Filter the temp_df using the *cleaned* selected chain value
#                 temp_df_for_batches = temp_df_for_batches[temp_df_for_batches[chain_name_col] == selected_chain].copy()

#         # Generate batch list using the now clean/filtered data
#         all_batches = prepare_batch_list(temp_df_for_batches, col_map)
#         batch_options = ["— Show All Batch —"] + all_batches
#         selected_batch = st.sidebar.selectbox("Select Batch", batch_options)

#         dependency_depth = st.sidebar.slider("Dependency Depth", 1, 5, 1) 
#         generate_btn = st.sidebar.button("Generate Diagram")

#         if generate_btn:
#             with st.spinner("🔄 Generating diagram..."):
                
#                 # Normalize selected batch for lookup
#                 selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— Show All Batch —" else selected_batch
                
#                 if selected_batch != "— Show All Batch —":
#                     # Case 1: Specific Batch is selected
                    
#                     related_batches = get_dependencies(df, selected_batch_upper, col_map, dependency_depth)
#                     filtered_df = df[df[col_map['batch']].astype(str).isin(related_batches)].dropna(subset=[col_map['batch']])
                    
#                     st.subheader(f"Workflow for: **{selected_batch_upper}** (Depth {dependency_depth}) - Includes Cross-Chain Dependencies")

#                 elif selected_chain != "— All Chains —":
#                     # Case 2: Chain is selected (filtered by ffill)
#                     filtered_df = temp_df_for_batches.copy()
#                     st.subheader(f"Full Workflow for Chain: **{selected_chain}** (Internal View)")
                    
#                 else:
#                     # Case 3: All
#                     filtered_df = df.copy()
#                     st.subheader("Full Workflow (All Batch and All Chains)")

#                 if filtered_df.empty:
#                     st.warning("No connections found. Try a different batch or depth, or check your chain filter.")
#                 else:
#                     # 1. Generate standard Graphviz source for display and standard download
#                     future_std = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=False)
#                     dot_source_std = future_std.result()
                    
#                     # 2. Generate high-res Graphviz source for high-res download
#                     future_high_res = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=True)
#                     dot_source_high_res = future_high_res.result()

#                     st.graphviz_chart(dot_source_std)
                    
#                     # --- Column layout for the two download buttons ---
#                     col1, col2 = st.columns(2)
                    
#                     # Standard Download
#                     png_data_std = Source(dot_source_std).pipe(format='png')
#                     with col1:
#                         st.download_button(
#                             "📥 Download Standard Resolution (PNG)",
#                             data=png_data_std,
#                             file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_std.png",
#                             mime="image/png"
#                         )
                    
#                     # High Resolution Download
#                     png_data_high_res = Source(dot_source_high_res).pipe(format='png')
#                     with col2:
#                         st.download_button(
#                             "💾 Download High Resolution Big (PNG)",
#                             data=png_data_high_res,
#                             file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_high_res.png",
#                             mime="image/png"
#                         )
#                     # --------------------------------------------------

#                     with st.expander("View Filtered Data & Graphviz Source"):
#                         st.dataframe(filtered_df)
#                         st.code(dot_source_std)

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         st.exception(e)


# ! VERSION 4 TESTED

# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from concurrent.futures import ThreadPoolExecutor
# from graphviz import Digraph, Source

# # Ensure Graphviz is available (works on Windows / Streamlit Cloud)
# GRAPHVIZ_PATHS = [
#     r"./Graphviz/bin" 
# ]

# for path in GRAPHVIZ_PATHS:
#     if os.path.exists(path):
#         os.environ["PATH"] += os.pathsep + path


# # -------------------------------------------
# # Streamlit Page Setup
# # -------------------------------------------
# st.set_page_config(page_title="Batch Workflow Diagram Generator", layout="wide")
# st.title("🔗 Batch Dependency Summary & Diagram Generator")
# st.markdown("Upload your Excel file (must contain **Batch**, **Predecessor**, **Successor** columns).")

# executor = ThreadPoolExecutor(max_workers=2)

# # -------------------------------------------
# # CACHED FUNCTIONS
# # -------------------------------------------

# @st.cache_data(show_spinner=False)
# def load_excel(uploaded_file):
#     """Load Excel file once."""
#     df = pd.read_excel(uploaded_file)
#     df.columns = [col.strip() for col in df.columns]
#     return df

# @st.cache_data(show_spinner=False)
# def get_dependencies(df, start_batch, col_map, depth=2):
#     """
#     Find all related predecessors and successors up to a depth.
#     (Used for filtering the graph data)
#     """
#     df_clean = df.copy() 
#     start_batch = start_batch.strip().upper() 

#     all_related_batches = {start_batch}
#     current_level = {start_batch}

#     for _ in range(depth):
#         next_level = set()

#         # Predecessors: Find rows where the main batch is in the current level
#         for _, row in df_clean[df_clean[col_map['batch']].isin(current_level)].iterrows():
#             preds = str(row[col_map['predecessor']])
#             if pd.notna(preds):
#                 next_level.update([p for p in preds.split('/') if p])

#         # Successors (where current is in predecessor of another batch)
#         for current_b in current_level:
#             mask_pred = df_clean[col_map['predecessor']].astype(str).str.contains(
#                 r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
#             )
#             succs = df_clean[mask_pred][col_map['batch']].tolist()
#             next_level.update(succs)

#         # Successors (where current is in successor col)
#         for current_b in current_level:
#             mask_succ = df_clean[col_map['successor']].astype(str).str.contains(
#                 r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
#             )
#             preds = df_clean[mask_succ][col_map['batch']].tolist()
#             next_level.update(preds)

#         new_batches = next_level - all_related_batches
#         if not new_batches:
#             break
#         all_related_batches.update(new_batches)
#         current_level = new_batches

#     return list(all_related_batches)

# @st.cache_data(show_spinner=False)
# def prepare_batch_list(df, col_map):
#     """Collect all batch names from batch/predecessor/successor columns."""
    
#     all_batches_raw = pd.concat([
#         df[col_map['batch']].astype(str).dropna(),
#         df[col_map['predecessor']].astype(str).str.split('/').explode().dropna(),
#         df[col_map['successor']].astype(str).str.split('/').explode().dropna()
#     ]).unique()
    
#     # Final filter to remove common non-batch values (empty string, 'NAN', 'NONE')
#     return sorted([
#         b for b in all_batches_raw 
#         if b and b.upper() not in ['NAN', 'NONE']
#     ])


# def calculate_dependencies(row, col_map, all_batches_df):
#     """
#     Calculates Pre/Post dependencies and counts for a single row (batch).
#     """
#     current_batch = row[col_map['batch']]
    
#     # 1. Pre Dependencies (What the current batch depends on)
#     pre_deps_str = str(row.get(col_map['predecessor'], ''))
#     pre_deps = [d for d in pre_deps_str.split('/') if d and d.upper() not in ['NAN', 'NONE']]
    
#     # 2. Post Dependencies (What depends on the current batch)
#     post_deps_str_succ = str(row.get(col_map['successor'], ''))
#     post_deps_succ = [d for d in post_deps_str_succ.split('/') if d and d.upper() not in ['NAN', 'NONE']]
    
#     # Successors (from the 'BATCH' column of other rows where current_batch is in PREDECESSOR)
#     mask_pred = all_batches_df[col_map['predecessor']].astype(str).str.contains(
#         r'(^|/)' + re.escape(current_batch) + r'(/|$)', regex=True
#     )
#     post_deps_pred = all_batches_df[mask_pred][col_map['batch']].tolist()
    
#     # Combine and get unique list of Post Dependencies
#     all_post_deps = sorted(list(set(post_deps_succ + post_deps_pred)))
    
#     return pd.Series([
#         '/'.join(pre_deps), 
#         len(pre_deps), 
#         '/'.join(all_post_deps), 
#         len(all_post_deps)
#     ])


# def get_mapped_columns(col_map):
#     """Helper to map col_map keys to standard headers for output."""
#     mapping = {}
#     for key, name in col_map.items():
#         if key == 'time':
#             mapping['START TIME'] = name
#         elif key == 'chain':
#              mapping['CHAIN'] = name
#         elif key == 'batch':
#              mapping['BATCH'] = name
#         elif key == 'predecessor':
#              mapping['PREDECESSOR'] = name
#         elif key == 'successor':
#              mapping['SUCCESSOR'] = name
#         elif key == 'path':
#              mapping['PATH'] = name
#         elif key == 'condition':
#              mapping['CONDITION'] = name
#         elif key == 'param1':
#              mapping['PARAMETER 1'] = name
#         elif key == 'param2':
#              mapping['PARAMETER 2'] = name
#         elif key == 'param3':
#              mapping['PARAMETER 3'] = name
#     return mapping


# @st.cache_data(show_spinner=False)
# def create_raw_dependencies_table(filtered_df, col_map):
#     """
#     Generates the DataFrame for the "Raw Dependencies Table" (Table 1).
#     Shows the raw PREDECESSOR and SUCCESSOR values from the source file.
#     """
    
#     mapped_cols = get_mapped_columns(col_map)
    
#     output_column_order = [
#         'CHAIN', 'BATCH', 'PREDECESSOR', 'SUCCESSOR', 'START TIME', 
#         'CONDITION', 'PATH', 'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
#     ]

#     # Rename and select columns
#     final_df = filtered_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()

#     # Filter to only columns that were successfully mapped
#     output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    
#     # Clean up the output: Replace 'nan' strings with empty string for a cleaner look
#     output_df = output_df.replace('nan', '', regex=True)
    
#     return output_df


# @st.cache_data(show_spinner=False)
# def create_dependency_summary_table(filtered_df, df_full_for_lookup, col_map):
#     """
#     Generates the final DataFrame for the "SUMMARY TABLE" (Table 2).
#     Shows the calculated PRE/POST DEPENDENCY list and count.
#     """
    
#     mapped_cols = get_mapped_columns(col_map)
    
#     # Apply the calculation
#     dependencies_cols = ['PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 'POST DEPENDENCY', 'NO OF POST DEPENDENCIES']
    
#     # Apply the dependency calculation across all rows in the filtered data
#     temp_df = filtered_df.copy() # Work on a copy
#     temp_df[dependencies_cols] = temp_df.apply(
#         calculate_dependencies, 
#         axis=1, 
#         result_type='expand', 
#         args=(col_map, df_full_for_lookup)
#     )
    
#     # Rename the raw columns to the output names
#     final_df = temp_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()

#     # Define the final requested order
#     output_column_order = [
#         'CHAIN', 'BATCH', 
#         'PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 
#         'POST DEPENDENCY', 'NO OF POST DEPENDENCIES',  
#         'START TIME', 'PATH', 'CONDITION', 
#         'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
#     ]

#     # Filter to only columns that were successfully mapped or calculated
#     output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    
#     # Clean up the output: Replace 'nan' strings with empty string for a cleaner look
#     output_df = output_df.replace('nan', '', regex=True)
    
#     return output_df


# @st.cache_data(show_spinner=False)
# def generate_graphviz_source(filtered_df, col_map, selected_batch, high_res=False):
#     """Generate and return Graphviz source code string, including time/frequency."""
    
#     time_col_name = col_map.get('time') 
    
#     # --- Graphviz Attributes (Curvy Lines) ---
#     graph_attrs = {
#         'rankdir': 'TB',
#         'splines': 'curved', 
#         'nodesep': '0.5', 'ranksep': '0.75',
#         'fontname': 'Helvetica'
#     }
    
#     if high_res:
#         graph_attrs['size'] = '50,50!' 
#         graph_attrs['dpi'] = '300' 
#     else:
#         graph_attrs['size'] = '15,10!'
        
#     dot = Digraph(
#         comment='Batch Workflow',
#         format='png',
#         graph_attr=graph_attrs,
#         node_attr={'fontname': 'Helvetica', 'shape': 'box', 'style': 'rounded'},
#         edge_attr={'fontname': 'Helvetica'}
#     )
#     # ----------------------------------------------------------------
    
#     # Get all nodes that should be in the graph (Batch column values + all dependencies)
#     all_nodes = set()
#     for col_key in ['batch', 'predecessor', 'successor']:
#         col = col_map.get(col_key)
#         if col and col in filtered_df.columns:
#             if col_key in ['predecessor', 'successor']:
#                 deps = filtered_df[col].astype(str).str.split('/').explode().dropna()
#                 all_nodes.update(deps.unique())
#             else:
#                 all_nodes.update(filtered_df[col].astype(str).dropna().unique())

#     # Map for quick lookup of time/frequency
#     time_map = filtered_df.set_index(col_map['batch'])[time_col_name].to_dict() if time_col_name and col_map.get('batch') in filtered_df.columns else {}
    
#     # 1. Create Nodes
#     selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— SHOW ALL BATCH —" else selected_batch

#     for batch_str in all_nodes:
#         if not batch_str or batch_str.upper() in ['NAN', 'NONE']:
#             continue
            
#         time_data = time_map.get(batch_str)
#         time_label = ""
#         if time_data and pd.notna(time_data):
#             time_label = "\n" + str(time_data).replace("/", "\n") 
        
#         label_text = batch_str + time_label
        
#         node_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'white'}
        
#         if selected_batch != "— Show All Batch —" and batch_str == selected_batch_upper:
#             node_attrs['fillcolor'] = '#87CEFA'
            
#         if col_map.get('batch') in filtered_df.columns and batch_str in filtered_df[col_map['batch']].values:
#             dot.node(batch_str, label=label_text, **node_attrs)
#         else:
#             dot.node(batch_str, label=label_text, shape='box', style='rounded,filled', fillcolor='#F0F8FF')


#     # 2. Create Edges
#     added_edges = set() 
    
#     for _, row in filtered_df.iterrows():
#         batch_str = str(row[col_map['batch']])
        
#         for cell_key, direction in [('predecessor', 'pred'), ('successor', 'succ')]:
#             cell = col_map.get(cell_key)
#             if not cell or cell not in filtered_df.columns: continue
            
#             deps = str(row[cell])
#             if not deps or deps.upper() == 'NAN':
#                 continue
            
#             for dep in [d for d in deps.split('/') if d]:
#                 if dep in all_nodes: 
                    
#                     if direction == 'pred':
#                         source_node = dep
#                         target_node = batch_str
#                     else:
#                         source_node = batch_str
#                         target_node = dep
                        
#                     edge_key = f"{source_node}->{target_node}"
                    
#                     if edge_key not in added_edges:
#                         dot.edge(source_node, target_node)
#                         added_edges.add(edge_key)

#     return dot.source

# # -------------------------------------------
# # MAIN UI
# # -------------------------------------------

# uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# if uploaded_file:
#     try:
#         df = load_excel(uploaded_file)
#         found_cols_lower = {col.lower(): col for col in df.columns}
#         required_cols = ['batch', 'predecessor', 'successor']
        
#         # Mapping for all required and optional columns
#         col_map = {c: found_cols_lower[c] for c in required_cols if c in found_cols_lower}
        
#         optional_cols_map = {
#             'chain': 'chain', 
#             'time': 'time', 
#             'path': 'path', 
#             'condition': 'condition',
#             'param1': 'parameter 1', 
#             'param2': 'parameter 2', 
#             'param3': 'parameter 3'
#         }
        
#         # Find time/frequency column
#         time_col_key = next((k for k in found_cols_lower if 'time' in k or 'frequency' in k), None)
#         if time_col_key:
#             col_map['time'] = found_cols_lower[time_col_key]
#         else:
#             col_map['time'] = None

#         # Map other optional columns if they exist
#         for key, name_part in optional_cols_map.items():
#             if key not in col_map: 
#                 found_col_name = next((v for k, v in found_cols_lower.items() if name_part in k), None)
#                 if found_col_name:
#                     col_map[key] = found_col_name
            
#         # Check mandatory columns
#         if not all(col in col_map for col in required_cols):
#             st.error("Excel must contain columns: **Batch**, **Predecessor**, **Successor**")
#             st.stop()
            
#         # --- AGGRESSIVE CLEANING ---
#         for col_key, col_name in col_map.items():
#             if col_name is not None:
#                 df[col_name] = df[col_name].astype(str).str.strip()
                
#                 if col_key in ['batch', 'predecessor', 'successor']:
#                     df[col_name] = df[col_name].str.upper()
                
#                 df.loc[df[col_name].isin(['NAN', '', 'NONE']), col_name] = np.nan
#         # ---------------------------

        
#         chain_name_col = col_map.get('chain')
#         selected_chain = "— All Chains —"
        
#         temp_df_for_batches = df.copy()

#         if chain_name_col:
#             # Ffill the Chain Column for filtering (Chain name is not uppercased)
#             df[chain_name_col] = df[chain_name_col].ffill()
#             temp_df_for_batches[chain_name_col] = temp_df_for_batches[chain_name_col].ffill()
            
#             all_chains = sorted(df[chain_name_col].astype(str).str.strip().dropna().unique())
#             selected_chain = st.sidebar.selectbox("Filter by Chain", ["— All Chains —"] + all_chains)
            
#             if selected_chain != "— All Chains —":
#                 temp_df_for_batches = temp_df_for_batches[temp_df_for_batches[chain_name_col] == selected_chain].copy()

#         all_batches = prepare_batch_list(temp_df_for_batches, col_map)
#         batch_options = ["— Show All Batch —"] + all_batches
#         selected_batch = st.sidebar.selectbox("Select Batch", batch_options)

#         dependency_depth = st.sidebar.slider("Diagram Dependency Depth", 1, 5, 1) 
#         generate_btn = st.sidebar.button("Generate Summary & Diagram")

#         if generate_btn:
#             with st.spinner("🔄 Generating data..."):
                
#                 selected_batch_upper = selected_batch.strip().upper() if selected_batch != "— Show All Batch —" else selected_batch
                
#                 if selected_batch != "— Show All Batch —":
#                     related_batches = get_dependencies(df, selected_batch_upper, col_map, dependency_depth)
#                     filtered_df = df[df[col_map['batch']].astype(str).isin(related_batches)].copy()
#                     st.subheader(f"Workflow for: **{selected_batch_upper}** (Depth {dependency_depth})")
#                 elif selected_chain != "— All Chains —":
#                     filtered_df = temp_df_for_batches.copy()
#                     st.subheader(f"Full Workflow for Chain: **{selected_chain}**")
#                 else:
#                     filtered_df = df.copy()
#                     st.subheader("Full Workflow (All Batch and All Chains)")

#                 if filtered_df.empty:
#                     st.warning("No connections found. Try a different batch or depth, or check your chain filter.")
#                 else:
                    
#                     # --- TABLE 1: RAW DEPENDENCIES TABLE (The "Previous Table") ---
#                     raw_deps_df = create_raw_dependencies_table(filtered_df.copy(), col_map)
                    
#                     st.markdown("---")
#                     st.subheader("📋 RAW DEPENDENCIES TABLE (Source View)")
#                     st.dataframe(raw_deps_df, use_container_width=True)
                    
#                     # --- TABLE 2: SUMMARY TABLE ---
#                     summary_table_df = create_dependency_summary_table(
#                         filtered_df.copy(), 
#                         df,                # Pass the full cleaned DF for accurate POST DEPENDENCY lookup
#                         col_map
#                     )
                    
#                     st.markdown("---")
#                     st.subheader("📊 SUMMARY TABLE (Calculated Dependencies)")
#                     st.dataframe(summary_table_df, use_container_width=True)
#                     st.markdown("---")
                    
#                     # --- GRAPH GENERATION ---
#                     future_std = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=False)
#                     dot_source_std = future_std.result()
                    
#                     future_high_res = executor.submit(generate_graphviz_source, filtered_df, col_map, selected_batch_upper, high_res=True)
#                     dot_source_high_res = future_high_res.result()

#                     st.subheader("🔗 Batch Dependency Diagram")
#                     st.graphviz_chart(dot_source_std)
                    
#                     # --- Download buttons ---
#                     col1, col2 = st.columns(2)
                    
#                     png_data_std = Source(dot_source_std).pipe(format='png')
#                     with col1:
#                         st.download_button(
#                             "📥 Download Standard Resolution (PNG)",
#                             data=png_data_std,
#                             file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_std.png",
#                             mime="image/png"
#                         )
                    
#                     png_data_high_res = Source(dot_source_high_res).pipe(format='png')
#                     with col2:
#                         st.download_button(
#                             "💾 Download High Resolution Big (PNG)",
#                             data=png_data_high_res,
#                             file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_high_res.png",
#                             mime="image/png"
#                         )
                    
#                     with st.expander("View Filtered Raw Data & Graphviz Source"):
#                         st.dataframe(filtered_df)
#                         st.code(dot_source_std)

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         st.exception(e)


# ! VERSION 5 UNTESTED -- EXTRA FEATURE

import os
import streamlit as st
import pandas as pd
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from graphviz import Digraph, Source
import plotly.express as px # New Import for Charts

# Ensure Graphviz is available (works on Windows / Streamlit Cloud)
GRAPHVIZ_PATHS = [
    r"./Graphviz/bin" 
]

for path in GRAPHVIZ_PATHS:
    if os.path.exists(path):
        os.environ["PATH"] += os.pathsep + path


# -------------------------------------------
# Streamlit Page Setup
# -------------------------------------------
st.set_page_config(page_title="Batch Workflow Diagram Generator", layout="wide")
st.title("🔗 Batch Dependency Summary & Diagram Generator")
st.markdown("Upload your Excel file (must contain **Batch**, **Predecessor**, **Successor** columns).")

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------
# CACHED FUNCTIONS
# -------------------------------------------

@st.cache_data(show_spinner=False)
def load_excel(uploaded_file):
    """Load Excel file once."""
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    return df

@st.cache_data(show_spinner=False)
def get_dependencies(df, start_batch, col_map, depth=2):
    """
    Find all related predecessors and successors up to a depth.
    (Used for filtering the graph data)
    """
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
                next_level.update([p for p in preds.split('/') if p])

        # Successors (where current is in predecessor of another batch)
        for current_b in current_level:
            mask_pred = df_clean[col_map['predecessor']].astype(str).str.contains(
                r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
            )
            succs = df_clean[mask_pred][col_map['batch']].tolist()
            next_level.update(succs)

        # Successors (where current is in successor col)
        for current_b in current_level:
            mask_succ = df_clean[col_map['successor']].astype(str).str.contains(
                r'(^|/)' + re.escape(current_b) + r'(/|$)', regex=True
            )
            preds = df_clean[mask_succ][col_map['batch']].tolist()
            next_level.update(preds)

        new_batches = next_level - all_related_batches
        if not new_batches:
            break
        all_related_batches.update(new_batches)
        current_level = new_batches

    return list(all_related_batches)

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
    """
    Calculates Pre/Post dependencies and counts for a single row (batch).
    """
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
    post_deps_pred = all_batches_df[mask_pred][col_map['batch']].tolist()
    
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
    for key, name in col_map.items():
        if key == 'time':
            mapping['START TIME'] = name
        elif key == 'chain':
             mapping['CHAIN'] = name
        elif key == 'batch':
             mapping['BATCH'] = name
        elif key == 'predecessor':
             mapping['PREDECESSOR'] = name
        elif key == 'successor':
             mapping['SUCCESSOR'] = name
        elif key == 'path':
             mapping['PATH'] = name
        elif key == 'condition':
             mapping['CONDITION'] = name
        elif key == 'param1':
             mapping['PARAMETER 1'] = name
        elif key == 'param2':
             mapping['PARAMETER 2'] = name
        elif key == 'param3':
             mapping['PARAMETER 3'] = name
    return mapping


@st.cache_data(show_spinner=False)
def create_raw_dependencies_table(filtered_df, col_map):
    """
    Generates the DataFrame for the "Raw Dependencies Table" (Table 1).
    Shows the raw PREDECESSOR and SUCCESSOR values from the source file.
    """
    
    mapped_cols = get_mapped_columns(col_map)
    
    output_column_order = [
        'CHAIN', 'BATCH', 'PREDECESSOR', 'SUCCESSOR', 'START TIME', 
        'CONDITION', 'PATH', 'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
    ]

    # Rename and select columns
    final_df = filtered_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()

    # Filter to only columns that were successfully mapped
    output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    
    # Clean up the output: Replace 'nan' strings with empty string for a cleaner look
    output_df = output_df.replace('nan', '', regex=True)
    
    return output_df


@st.cache_data(show_spinner=False)
def create_dependency_summary_table(filtered_df, df_full_for_lookup, col_map):
    """
    Generates the final DataFrame for the "SUMMARY TABLE" (Table 2).
    Shows the calculated PRE/POST DEPENDENCY list and count.
    """
    
    mapped_cols = get_mapped_columns(col_map)
    
    # Apply the calculation
    dependencies_cols = ['PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 'POST DEPENDENCY', 'NO OF POST DEPENDENCIES']
    
    # Apply the dependency calculation across all rows in the filtered data
    temp_df = filtered_df.copy() # Work on a copy
    temp_df[dependencies_cols] = temp_df.apply(
        calculate_dependencies, 
        axis=1, 
        result_type='expand', 
        args=(col_map, df_full_for_lookup)
    )
    
    # Rename the raw columns to the output names
    final_df = temp_df.rename(columns={v: k for k, v in mapped_cols.items() if v}).copy()

    # Define the final requested order
    output_column_order = [
        'CHAIN', 'BATCH', 
        'PRE DEPENDENCY', 'NO OF PREDEPENDENCY', 
        'POST DEPENDENCY', 'NO OF POST DEPENDENCIES',  
        'START TIME', 'PATH', 'CONDITION', 
        'PARAMETER 1', 'PARAMETER 2', 'PARAMETER 3'
    ]

    # Filter to only columns that were successfully mapped or calculated
    output_df = final_df[[col for col in output_column_order if col in final_df.columns]].copy()
    
    # Clean up the output: Replace 'nan' strings with empty string for a cleaner look
    output_df = output_df.replace('nan', '', regex=True)
    
    return output_df


@st.cache_data(show_spinner=False)
def generate_graphviz_source(filtered_df, col_map, selected_batch, high_res=False):
    """Generate and return Graphviz source code string, including time/frequency."""
    
    time_col_name = col_map.get('time') 
    
    # --- Graphviz Attributes (Curvy Lines) ---
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
    # ----------------------------------------------------------------
    
    # Get all nodes that should be in the graph (Batch column values + all dependencies)
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
        
        if selected_batch != "— Show All Batch —" and batch_str == selected_batch_upper:
            node_attrs['fillcolor'] = '#87CEFA'
            
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
            if not deps or deps.upper() == 'NAN':
                continue
            
            for dep in [d for d in deps.split('/') if d]:
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
# MAIN UI
# -------------------------------------------

uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = load_excel(uploaded_file)
        found_cols_lower = {col.lower(): col for col in df.columns}
        required_cols = ['batch', 'predecessor', 'successor']
        
        # Mapping for all required and optional columns
        col_map = {c: found_cols_lower[c] for c in required_cols if c in found_cols_lower}
        
        optional_cols_map = {
            'chain': 'chain', 
            'time': 'time', 
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
            
        # Check mandatory columns
        if not all(col in col_map for col in required_cols):
            st.error("Excel must contain columns: **Batch**, **Predecessor**, **Successor**")
            st.stop()
            
        # --- AGGRESSIVE CLEANING ---
        for col_key, col_name in col_map.items():
            if col_name is not None:
                df[col_name] = df[col_name].astype(str).str.strip()
                
                if col_key in ['batch', 'predecessor', 'successor']:
                    df[col_name] = df[col_name].str.upper()
                
                df.loc[df[col_name].isin(['NAN', '', 'NONE']), col_name] = np.nan
        # ---------------------------

        
        chain_name_col = col_map.get('chain')
        selected_chain = "— All Chains —"
        
        temp_df_for_batches = df.copy()

        if chain_name_col:
            # Ffill the Chain Column for filtering (Chain name is not uppercased)
            df[chain_name_col] = df[chain_name_col].ffill()
            temp_df_for_batches[chain_name_col] = temp_df_for_batches[chain_name_col].ffill()
            
            all_chains = sorted(df[chain_name_col].astype(str).str.strip().dropna().unique())
            selected_chain = st.sidebar.selectbox("Filter by Chain", ["— All Chains —"] + all_chains)
            
            if selected_chain != "— All Chains —":
                temp_df_for_batches = temp_df_for_batches[temp_df_for_batches[chain_name_col] == selected_chain].copy()

        all_batches = prepare_batch_list(temp_df_for_batches, col_map)
        batch_options = ["— Show All Batch —"] + all_batches
        selected_batch = st.sidebar.selectbox("Select Batch", batch_options)

        dependency_depth = st.sidebar.slider("Diagram Dependency Depth", 1, 5, 1) 
        generate_btn = st.sidebar.button("Generate Summary & Diagram")

        if generate_btn:
            with st.spinner("🔄 Generating data..."):
                
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
                    st.warning("No connections found. Try a different batch or depth, or check your chain filter.")
                else:
                    
                    # --- TABLE 2: SUMMARY TABLE --- (Generate this first as it contains dependency counts)
                    summary_table_df = create_dependency_summary_table(
                        filtered_df.copy(), 
                        df,                # Pass the full cleaned DF for accurate POST DEPENDENCY lookup
                        col_map
                    )
                    
                    # --- NEW FEATURE: STATS AND CHART ---
                    
                    st.markdown("---")
                    st.subheader("🚀 Dependency Insights")
                    
                    # 1. Summary Metrics (Top Row)
                    col1, col2, col3 = st.columns(3)
                    
                    total_batches = len(summary_table_df)
                    total_pre_deps = summary_table_df['NO OF PREDEPENDENCY'].sum()
                    total_post_deps = summary_table_df['NO OF POST DEPENDENCIES'].sum()
                    
                    col1.metric("Total Batches Analyzed", total_batches)
                    col2.metric("Total Pre Dependencies", total_pre_deps)
                    col3.metric("Total Post Dependencies", total_post_deps)

                    
                    # 2. Top 10 Dependent Batches Chart
                    
                    # Ensure we have the necessary column
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
                        else:
                            st.info("No batches found with dependencies in the current filter.")
                    
                    st.markdown("---")
                    
                    # --- TABLE 1: RAW DEPENDENCIES TABLE (The "Previous Table") ---
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
                    
                    # --- Download buttons ---
                    col1, col2 = st.columns(2)
                    
                    png_data_std = Source(dot_source_std).pipe(format='png')
                    with col1:
                        st.download_button(
                            "📥 Download Standard Resolution (PNG)",
                            data=png_data_std,
                            file_name=f"batch_workflow_{selected_batch_upper.replace('— SHOW ALL BATCH —', 'all')}_std.png",
                            mime="image/png"
                        )
                    
                    png_data_high_res = Source(dot_source_high_res).pipe(format='png')
                    with col2:
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
        st.error(f"An error occurred: {e}")
        # Only show the full exception in debug mode/for developer
        # st.exception(e)