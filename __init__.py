import sys
import git


def get_git_root(path):
    try:
        repo = git.Repo(path, search_parent_directories=True)
        return repo.working_tree_dir
    except git.exc.InvalidGitRepositoryError:
        return None


ROOT_DIR = get_git_root(".")
sys.path.append(ROOT_DIR)
