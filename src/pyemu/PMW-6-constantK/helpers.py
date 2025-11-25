import os
import pandas as pd
import flopy.utils.binaryfile as bf
import numpy as np

def process_heads_csv(model_name="MLAC-model"):
    import flopy.utils.binaryfile as bf
    """
    Reads the MODFLOW binary heads file, extracts heads for specified
    observation wells at desired times, interpolating between time steps
    if necessary, and saves the data to heads.csv.
    """

    path = ['.\\3d_injection_modeling_PMW-3\\', '.\\3d_injection_modeling_PMW-6\\']
    wells = ['PMW-3', 'PMW-6']
    desired_order = [['time', 'HEAD_PW-1', 'HEAD_PMW-6', 'HEAD_PMW-10', 'HEAD_PMW-14',
                        'HEAD_PW-2', 'HEAD_PMW-9', 'HEAD_PZ-1', 'HEAD_PMW-4', 'HEAD_PMW-5',
                        'HEAD_OW24P', 'HEAD_OW23P', 'HEAD_OW17P'], 
                ['time', 'HEAD_PW-1', 'HEAD_PMW-3', 'HEAD_PMW-10', 'HEAD_PMW-14',
                        'HEAD_PW-2', 'HEAD_PMW-9', 'HEAD_PZ-1', 'HEAD_PMW-4', 'HEAD_PMW-5',
                        'HEAD_OW24P', 'HEAD_OW23P', 'HEAD_OW17P']]

    for path, well, desired_order in zip(path, wells, desired_order): 


        model_ws = path

        file_path = os.path.join(model_ws, 'well_location_and_times.csv')

        # --- Load observation info from the CSV file ---
        try:
            obs_df = pd.read_csv(file_path)

        except FileNotFoundError:
            raise FileNotFoundError("well_locations.csv not found. Did you run create_model.py?")

        # Check if the .hds file exists before attempting to read
        hds_filepath = os.path.join(model_ws, f'{model_name}.hds')
        if not os.path.exists(hds_filepath):
            raise FileNotFoundError(f"Binary heads file not found: {hds_filepath}")

        hds_file = bf.HeadFile(hds_filepath)
        model_output_times = np.array(hds_file.get_times())

        # Get the unique well names and locations
        well_locations = obs_df[['well_name', 'layer', 'row', 'col']].drop_duplicates()

        # Prepare results storage
        interpolated_results = []

        for index, obs in obs_df.iterrows():
            well_name = obs['well_name']
            layer, row, col = obs['layer'], obs['row'], obs['col']
            target_time = obs['time']

            # Find the time steps in the model output that bracket the target_time
            # Use np.searchsorted for efficient finding of insertion points
            idx = np.searchsorted(model_output_times, target_time)

            # Handle edge cases: target_time is before the first or after the last model output time
            if idx == 0:
                # Target time is at or before the first model output time.
                # Use the first time step's head.
                # You might want to raise a warning or handle this differently (e.g., extrapolation if allowed).
                t1 = model_output_times[0]
                heads_data_t1 = hds_file.get_data(totim=t1)[layer, row, col]
                interpolated_head = heads_data_t1
                print(f"Warning: Target time {target_time} is before or at first model time {t1}. Using head at {t1}.")

            elif idx == len(model_output_times):
                # Target time is at or after the last model output time.
                # Use the last time step's head.
                # You might want to raise a warning or handle this differently.
                t1 = model_output_times[-1]
                heads_data_t1 = hds_file.get_data(totim=t1)[layer, row, col]
                interpolated_head = heads_data_t1
                print(f"Warning: Target time {target_time} is at or after last model time {t1}. Using head at {t1}.")

            else:
                # Target time is between two model output times or exactly at one.
                t1 = model_output_times[idx - 1]
                t2 = model_output_times[idx]

                if target_time == t1:
                    # Exact match for the lower bound
                    heads_data_t1 = hds_file.get_data(totim=t1)[layer, row, col]
                    interpolated_head = heads_data_t1
                elif target_time == t2:
                    # Exact match for the upper bound
                    heads_data_t2 = hds_file.get_data(totim=t2)[layer, row, col]
                    interpolated_head = heads_data_t2
                else:
                    # Interpolate between t1 and t2
                    heads_data_t1 = hds_file.get_data(totim=t1)[layer, row, col]
                    heads_data_t2 = hds_file.get_data(totim=t2)[layer, row, col]

                    # Linear interpolation formula
                    interpolated_head = heads_data_t1 + (heads_data_t2 - heads_data_t1) * \
                                        (target_time - t1) / (t2 - t1)
            
            interpolated_results.append({
                'time': target_time,
                'well_name': f'HEAD_{well_name}',
                'head': interpolated_head
            })

        # Create and pivot the DataFrame
        results_df = pd.DataFrame(interpolated_results)

        # Pivot the data to have columns for each well name
        output_df = results_df.pivot(index='time', columns='well_name', values='head')

        # Sort by time, reset index for clean CSV, and ensure consistent output column order
        output_df = output_df.sort_index().reset_index()

        # Select the columns in the specified order
        # Note: If a column is in desired_order but not in output_df, it will be added as NaN.
        # If a column is in output_df but not in desired_order, it will be dropped.
        try:
            output_df = output_df[desired_order]
        except KeyError as e:
            print(f"Warning: A column was not found during reordering: {e}")
            # Continue saving, but without the specified order if it fails
        
        # Write to CSV
        output_df.to_csv(f'./{well}_heads.csv', index=False, float_format='%.16E')
        print(f"{well}_heads.csv created with interpolated head data at specified times.")

def replace_forward_run(filename, search_code, new_code):
    with open(filename, 'r') as file: 
        lines = file.readlines()

    modified_lines = []
    found = False

    for line in lines: 
        if search_code == line: 
            modified_lines.append(new_code + '\n' if not new_code.endswith('\n') else new_code)
            found = True
        else: 
            modified_lines.append(line)

    if found: 
        with open(filename, 'w') as file: 
            file.writelines(modified_lines)