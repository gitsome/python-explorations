import time
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

def time_comparison (data_params_package: DataParamsPackage, get_data: Callable[[Dict[str, Any]], Any]):

    results: List[FunctionTimingResults] = []

    functions = data_params_package["functions"]

    for func in functions:

        functionTimingResults: FunctionTimingResults = {"attempt": [], "time": [], "label": func.__name__}

        for data_index, data_params in enumerate(data_params_package["data_params"]):

            # get the data to be passed to the func
            func_data = get_data(data_params["params"])

            # time the executation of the func
            start = time.time()
            func(*func_data)
            end = time.time()

            # append the results fo the execution
            functionTimingResults["attempt"].append(data_index)
            functionTimingResults["time"].append(end - start)

        # append all the timing data for this function
        results.append(functionTimingResults)

    for result in results:
        plt.plot(result["attempt"], result["time"], label=result["label"])

    plt.xlabel(data_params_package["x_label"])
    plt.ylabel(data_params_package["y_label"])
    plt.legend(loc="upper left")
    plt.show()
