import subprocess

def run_command(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(1)
    return result.stdout.strip()

def main():
    # Prompt user for a commit message
    commit_message = input("Enter commit message: ").strip()
    if not commit_message:
        print("Commit message cannot be empty.")
        return

    # Run Git commands
    print("Adding changes...")
    run_command("git add .")

    print("Committing changes...")
    run_command(f'git commit -m "{commit_message}"')

    print("Pushing to remote...")
    run_command("git push origin main")

    print("✅ Changes pushed successfully!")

if __name__ == "__main__":
    main()
