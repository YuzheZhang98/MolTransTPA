import json
import pandas as pd

def load_list(file_path):
    """
    Load a list from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        list: The loaded list.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_solvents(file_path):
    """
    Load solvents from a csv file.
    
    Args:
        file_path (str): The path to the csv file.
        
    Returns:
        dict: The loaded solvents.
    """
    slow_solvent_list = []
    name_list = []
    data = pd.read_csv(file_path)
    for index, row in data.iterrows():
        if row["Solvent"] in name_list:
            continue
        else:
            solvent = {
                "Solvent": row["Solvent"],
                "ET(30)": row["ET(30)"],
                "dielectic constant": row["dielectic constant"],
                "dipole moment": row["dipole moment"]
            }
        slow_solvent_list.append(solvent)

    fast_solvent_dict = {}
    for entry in slow_solvent_list:
        solvent = entry["Solvent"]
        fast_solvent_dict[solvent] = {
            "ET(30)": entry["ET(30)"],
            "dielectic": entry["dielectic constant"],
            "dipole": entry["dipole moment"]
        }
    with open("solvents_list.json", "w") as file:
        json.dump(fast_solvent_dict, file, indent=4)
    return fast_solvent_dict

if __name__ == "__main__":
    solvents_dict = load_solvents("solvent_params.csv")

    formatted_list = []
    data_list = load_list("TPAML.json")

    for entry in data_list:
        if entry["Solvent"] == "NaN" or entry["Solvent"] == "\uff1f" or entry["Solvent"] == "None":
            continue
        temp_sol = entry["Solvent"]
        if temp_sol == "Ethyl acetate":
            temp_sol = "EA"
        solvent_params = solvents_dict[temp_sol]
        if entry["smiles"].find("I") != -1 or \
            entry["smiles"].find("P") != -1 or \
                entry["smiles"].find("Si") != -1:
            continue
        if type(entry["wavelength"]) == list:
            for i in range(len(entry["wavelength"])):
                if entry["wavelength"][i] < 600 or entry["wavelength"][i] > 1000:
                    continue
                new_entry = {
                    "smiles": entry["smiles"],
                    "ET(30)": solvent_params["ET(30)"],
                    "dielectic constant": solvent_params["dielectic"],
                    "dipole moment": solvent_params["dipole"],
                    "wavelength": entry["wavelength"][i],
                    "TPACS": entry["TPACS"][i],
                }
                formatted_list.append(new_entry)
        else:
            new_entry = {
                "smiles": entry["smiles"],
                "Solvent": entry["Solvent"],
                "ET(30)": solvent_params["ET(30)"],
                "dielectic constant": solvent_params["dielectic"],
                "dipole moment": solvent_params["dipole"],
                "wavelength": entry["wavelength"],
                "TPACS": entry["TPACS"],
            }
            formatted_list.append(new_entry)
    print(len(formatted_list))
    with open("formatted_data.json", "w") as file:
        json.dump(formatted_list, file, indent=4)
