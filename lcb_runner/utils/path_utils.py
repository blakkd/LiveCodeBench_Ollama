import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    path = f"cache/{model_repr}/{scenario}_{n}_.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{cot_suffix}_eval_all.json"
    return path
