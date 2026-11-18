import pandas as pd
import numpy as np


def filter_data(filepath):
    # Carreguem el dataset com un Dataframe de Pandas
    df = pd.read_parquet(filepath)

    # Filtrem les tasques: en seleccionem les més interessants
    df = df[
        # Tasques que ocupen solament 1 node
        (df["num_nodes_alloc"] == 1)    &

        # Tasques que s'executen entre 0 i 1800 segons
        (df["run_time"] > 0)            &
        (df["run_time"] <= 1800)        &
        
        # Tasques que requereixen el còmput d'almenys 1 core
        (df["num_cores_req"] > 0)
    ].copy()


    # Calculem la potència mitjana de cada tasca
    def mean_power(values):
        # Si no té valors, retorna NULL
        if (values is None) or (len(values)==0):
            return np.nan
        return np.mean(values)
    
    df["mean_power"] = df["node_power_consumption"].apply(mean_power)

    # Filtrem els registres que tenen valors no nuls i positius de "power"
    df = df.dropna(subset=["mean_power"])
    df = df[df["mean_power"] > 0]

    # Seleccionem solament 5 tasques per fer proves
    tasks = df.head(5).copy()

    # Convertim els temps d'execució en blocs de 5 minuts (300 segons)
    tasks["time_units"] = np.ceil(tasks["run_time"]/300).astype(int)
    tasks = tasks.reset_index(drop=True)

    return tasks