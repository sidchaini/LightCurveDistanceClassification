import re
import os


def remove_stderr_from_notebook(input_path, output_path):
    # Load the notebook
    with open(input_path, "r", encoding="utf-8") as fp:
        nb = fp.read()

    # Define the regex pattern
    pattern = r'\{\s*"name":\s*"stderr",.*?\},'

    # Replace the pattern with an empty string and count the matches
    modified_contents, num_matches = re.subn(pattern, "", nb, flags=re.DOTALL)
    # print(f"{num_matches} matches were found and removed.")

    if num_matches > 0:
        # Write the modified contents back to the file or a new file
        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write(modified_contents)

    return num_matches


if __name__ == "__main__":
    for root, dirs, files in os.walk("."):
        if ".ipynb_checkpoints" in root:
            continue
        for file in files:
            if file.endswith(".ipynb"):
                input_path = os.path.join(root, file)
                output_path = input_path  # Overwrite the same file
                num_changes = remove_stderr_from_notebook(input_path, output_path)
                if num_changes > 0:
                    print(f"{num_changes} changes were made in {input_path[2:]}")
