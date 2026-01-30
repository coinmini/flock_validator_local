
import os
from loguru import logger

def is_latest_version(repo_path: str):
    """
    Check if the current branch is up-to-date with the remote main branch.
    Parameters:
    - repo_path (str or Path): The path to the git repository.
    """
    IS_DOCKER_CONTAINER = os.getenv("IS_DOCKER_CONTAINER", False)
    if IS_DOCKER_CONTAINER:
        logger.info("Skip checking the latest version in docker container")
        logger.info(
            "Please make sure you are using the latest version of the docker image."
        )
        return
    
    import git  # only import git in non-docker container environment because it is not installed in docker image
    try:
        repo = git.Repo(repo_path)
        origin = repo.remotes.origin
        origin.fetch()

        local_commit = repo.commit("main")
        remote_commit = repo.commit("origin/main")

        if local_commit.hexsha != remote_commit.hexsha:
            logger.error(
                "The local code is not up to date with the main branch.Pls update your version"
            )
            raise
    except git.exc.InvalidGitRepositoryError:
        logger.error("This is not a git repository.")
        raise
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise