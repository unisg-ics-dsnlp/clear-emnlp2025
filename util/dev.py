# utilities for dev environment, should not be needed / used for prod related stuff

def generate_env_sample(env_file_path: str, env_example_path: str) -> None:
    """
    Generates .env.example file from .env file. This file contains only the keys but no values.
    The .env file is in the .gitignore and for obvious reasons should not be committed to the repo.

    :param env_file_path: path to .env file
    :param env_example_path: path to .env.example file
    """
    # reads .env file
    with open(env_file_path, 'r') as f:
        lines = f.readlines()

    out = ''
    for line in lines:
        is_key = '=' in line
        line = line.split('=')[0]
        if is_key:
            line += '='
        if '\n' not in line:
            line += '\n'  # prevents empty lines from causing two newlines
        out += line
    with open(env_example_path, 'w') as f:
        f.write(out)


if __name__ == '__main__':
    generate_env_sample('../.env', '../.env.example')
