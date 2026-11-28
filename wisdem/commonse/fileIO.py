import os
import warnings
#import pickle
import openmdao
import numpy as np
import pandas as pd
import scipy.io as sio
from openmdao.utils.mpi import MPI

def get_variable_list(prob, rank_0=False):

    # Get all OpenMDAO inputs and outputs into a dictionary
    input_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, out_stream=None) # is_indep_var=True
    # If MPI, share input dictionary from rank 0 to all other ranks, which would otherwise be empty
    if MPI and rank_0 == False:
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0) 
    for k in range(len(input_dict)):
        input_dict[k][1]["type"] = "input"

    inter_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, out_stream=None) # is_indep_var=False
    # If MPI, share intermediate dictionary from rank 0 to all other ranks, which would otherwise be empty
    if MPI and rank_0 == False:
        inter_dict = MPI.COMM_WORLD.bcast(inter_dict, root=0)
    for k in range(len(inter_dict)):
        inter_dict[k][1]["type"] = "intermediate"

    #var_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, out_stream=None)
    #for k in range(len(var_dict)):
    #    var_dict[k][1]["type"] = "output"

    out_dict = prob.model.list_outputs(prom_name=True, units=True, desc=True, out_stream=None)
    # If MPI, share output dictionary from rank 0 to all other ranks, which would otherwise be empty
    if MPI and rank_0 == False:
        out_dict = MPI.COMM_WORLD.bcast(out_dict, root=0)
    for k in range(len(out_dict)):
        out_dict[k][1]["type"] = "output"
        
    var_dict = input_dict.copy()
    var_dict.extend(inter_dict)
    var_dict.extend(out_dict)
    return input_dict, out_dict, var_dict


def variable_dict2df(var_dict):
    data = {}
    data["variables"] = []
    data["type"] = []
    data["units"] = []
    data["values"] = []
    data["description"] = []
    for k in range(len(var_dict)):
        unit_str = var_dict[k][1]["units"]
        if unit_str is None:
            unit_str = ""

        iname = var_dict[k][1]["prom_name"]
        itype = var_dict[k][1]["type"]
        if iname in data["variables"]:
            iprev = data["variables"].index( iname )
            if itype == "output":
                data["type"][iprev] = itype
            continue

        data["variables"].append(iname)
        data["type"].append(itype)
        data["units"].append(unit_str)
        data["values"].append(var_dict[k][1]["val"])
        data["description"].append(var_dict[k][1]["desc"])

    # Simplify stored value data type
    for k in range(len(data["values"])):
        if type(data["values"][k]) == type(np.array([])):
            if data["values"][k].size == 1:
                data["values"][k] = data["values"][k][0]
            else:
                data["values"][k] = data["values"][k].tolist()
        elif isinstance(data["values"][k], list) and len(data["values"][k]) == 1:
            data["values"][k] = data["values"][k][0]
                
    return pd.DataFrame(data)


def save_data(fname, prob, npz_file=True, mat_file=False, xls_file=True):
    # Get the variables
    _, _, var_dict = get_variable_list(prob, rank_0 = True)
    df = variable_dict2df(var_dict)
    
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Reduce to variables we can save for matlab or python
    array_dict = {}
    for k in range(len(var_dict)):
        unit_str = var_dict[k][1]["units"]
        if unit_str is None or unit_str in ["Unavailable","n/a"]:
            unit_str = ""
        elif len(unit_str) > 0:
            unit_str = "_" + unit_str

        iname = var_dict[k][1]["prom_name"] + unit_str
        value = var_dict[k][1]["val"]

        if iname in array_dict:
            continue

        if type(value) in [type(np.array([])), type(0.0), type(0), np.float64, np.int64]:
            array_dict[iname] = value
        elif type(value) == type(True):
            array_dict[iname] = np.bool_(value)
        elif type(value) == type(""):
            array_dict[iname] = np.str_(value)
        elif type(value) == type([]):
            temp_val = np.empty(len(value), dtype=object)
            temp_val[:] = value[:]
            array_dict[iname] = temp_val
        # else:
        #    print(var_dict[k])

    # Pickle the full archive so that we can load it back in if we need
    df.to_pickle(froot + ".pkl")
        
    # Save to numpy compatible
    if npz_file:
        kwargs = {key: array_dict[key] for key in array_dict.keys()}
        np.savez_compressed(froot + ".npz", **kwargs)

    # Save to matlab compatible
    if mat_file:
        sio.savemat(froot + ".mat", array_dict, long_field_names=True)

    if xls_file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cell contents too long.*",
                category=UserWarning
            )
            df.to_excel(froot + ".xlsx", index=False)
        df.to_csv(froot + ".csv", index=False)


    
        
def transfer_data(prob_from, prob_to, prefix_append=None, prefix_remove=None):

    if prefix_append is None:
        prefix_append = ''
        
    if prefix_remove is None:
        prefix_remove = ''

    # Get FROM data as DF
    if isinstance(prob_from, pd.DataFrame):
        var_df_from = prob_from
    elif isinstance(prob_from, openmdao.core.problem.Problem):
        _, _, var_dict_from = get_variable_list(prob_from)
        var_df_from = variable_dict2df(var_dict_from)
    
    # Get TO data as DF
    if isinstance(prob_to, pd.DataFrame):
        var_df_to = prob_to
    else:
        _, _, var_dict_to = get_variable_list(prob_to)
        var_df_to = variable_dict2df(var_dict_to)
    valid_vars_to = list(var_df_to["variables"])

    # Set the variables
    for k in range(len(var_df_from)):
        # Guess at the proper variable name
        invar_name = var_df_from["variables"].iloc[k]
        invar_units = var_df_from["units"].iloc[k]
        local_name = prefix_append + invar_name.replace(prefix_remove,"")
        local_name2 = local_name.replace(f"_{invar_units}", "")

        # Restore the type of the input data
        ival = var_df_from["values"].iloc[k]
        if isinstance(ival, (np.ndarray, float, int, list, dict)):
            pass
        elif isinstance(ival, str) and (ival.startswith('[') or ival.startswith('{')):
            ival = eval(ival)
        else:
            try:
                ival = float(ival)
            except Exception:
                pass

        # Store the value
        if local_name in valid_vars_to:
            prob_to[local_name] = ival
        elif local_name2 in valid_vars_to:
            prob_to[local_name2] = ival

    return prob_to


def load_data(fname, prob, prefix_append=None, prefix_remove=None):

    # Extracting from npz
    def npz2df(obj):
        data = {}
        data["variables"] = []
        data["units"] = []
        data["values"] = []
        for k in range(len(wt_obj.files)):
            iname = obj.files[k]
            iunit = iname.split('_')[-1]
            ival = obj[iname]
            data["variables"].append(iname)
            data["units"].append(iunit)
            data["values"].append(ival)
            
        mydf = pd.DataFrame(data)
        return mydf

    # Load in the difference filetypes
    if isinstance(fname, str) and fname.endswith(".xlsx"):
        mydf = pd.read_excel(fname)
        
    elif isinstance(fname, str) and fname.endswith(".csv"):
        mydf = pd.read_csv(fname)
        
    elif isinstance(fname, str) and fname.endswith(".npz"):
        wt_obj = np.load(fname, allow_pickle=True)
        mydf = npz2df(wt_obj)
        
    elif isinstance(fname, str) and fname.endswith(".pkl"):
        mydf = pd.read_pickle(fname)

    elif isinstance(fname, np.lib.npyio.NpzFile):
        mydf = npz2df(fname)

    elif isinstance(fname, pd.DataFrame) or isinstance(fname, openmdao.core.problem.Problem):
        mydf = fname

    else:
        raise Exception(f"Unknown file type, {fname}.  Expected xlsx or csv or npz or pkl")
    
    return transfer_data(mydf, prob, prefix_append=prefix_append, prefix_remove=prefix_remove)
