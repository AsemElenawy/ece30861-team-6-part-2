import multiprocessing
import os
import src.url_class as url_class
import metric_caller
from src.json_output import build_model_output

if __name__ == '__main__':
    verbosity = 1
    logfile = "log.txt"
    github_token = ""

    os.environ['API_KEY'] = ""


    #Running URL FILE
    project_groups: list[url_class.ProjectGroup] = url_class.parse_project_file("input.txt")

    # Preload available functions from the metrics directory used by your project
    log_q = multiprocessing.Queue()
    available_functions = metric_caller.load_available_functions("metrics", log_q, verbosity)

    for group in project_groups:
       # Guard against None optional fields to satisfy Pylance
        if group.model is None:
            continue

        model_ns = group.model.namespace
        model_repo = group.model.repo
        model_rev = group.model.rev

        code_link = group.code.link if group.code else ""
        dataset_repo = group.dataset.repo if group.dataset else ""

        #size = get_model_size(i.model.namespace, i.model.repo, i.model.rev)

        input_dict = {
            "repo_owner": model_ns,
            "repo_name": model_repo,
            "verbosity": verbosity,
            "log_queue": logfile,
            "model_size_bytes": 1,
            "github_str": code_link,
            "dataset_name": dataset_repo,
            "github_token": github_token,
        }

        scores, latency = metric_caller.run_concurrently_from_file(
            "./tasks.txt",  # tasks_filename
            input_dict,     # all_args_dict
            available_functions,  # available_functions
            logfile         # log_file
        )

        build_model_output(f"{model_ns}/{model_repo}", "model", scores, latency)