import time
import pandas as pd
from textwrap import wrap
from typing import Any, Dict, List, TypedDict, Callable

import numpy as np # type: ignore
import matplotlib.pyplot as plt

class DataParam(TypedDict):
    label: str
    params: Dict[str, Any]

class DataParamsPackage(TypedDict):
    x_label: str
    y_label: str
    data_params: List[DataParam]
    functions: List[Callable[[List[Any]], Any]]

class FunctionTimingResults(TypedDict):
    label: str
    attempt: List[int]
    time: List[float]

def wrap_label (label: str):
    return '\n'.join(wrap(label, 40))
        
def time_comparison (data_params_package: DataParamsPackage, get_data: Callable[[Dict[str, Any]], Any], runs: int = 1):

    results: List[FunctionTimingResults] = []

    functions = data_params_package["functions"]

    functions_count = len(functions)
    data_sets_count = len(data_params_package["data_params"])
    
    for func in functions:

        functionTimingResults: FunctionTimingResults = {"attempt": [], "time": [], "label": func.__name__}

        for data_index, data_params in enumerate(data_params_package["data_params"]):

            # get the data to be passed to the func
            func_data = get_data(data_params["params"])

            # cumulative time
            cumulative_time = 0
            
            # time the executation of the func for as many runs as we have
            
            for i in range(runs):
                
                start = time.time()
                func(*func_data)
                end = time.time()

                cumulative_time = cumulative_time + (end - start)
            
            # append the results fo the execution
            functionTimingResults["attempt"].append(data_index)
            functionTimingResults["time"].append(cumulative_time / runs)

        # append all the timing data for this function
        results.append(functionTimingResults)

    # Shared plot configs
    seconds_label = f"{data_params_package['y_label']} (avg over {runs} runs)"
        
    # Do a bar chart if there is only one data set
    if (data_sets_count == 1):
        
        fig, ax = plt.subplots()

        results_data = pd.DataFrame({
            "labels": [wrap_label(func.__name__) for func in functions],
            "values": [r["time"][0] for r in results]
        });
                
        y_pos = np.arange(len(results_data["values"]))

        ax.barh(y_pos, results_data["values"], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results_data["labels"])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(seconds_label)
        
        plt.show()
            
    # Otherwise normal line plot
    else:
        
        for result in results:
            plt.plot(result["attempt"], result["time"], label=result["label"])
        
        plt.legend(loc="upper left")
        plt.ylabel(seconds_label)
        plt.xlabel(data_params_package["x_label"])
        
        plt.show()
        
    
        
        
