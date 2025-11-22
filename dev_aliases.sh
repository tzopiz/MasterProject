# Backend Aliases
# Uses subshell () to avoid changing current directory
alias ib="(cd Backend && swift build)"
alias ir="(cd Backend && swift run App serve --hostname 0.0.0.0 --port 8080)"

# ML Service Aliases
# Note: 'source' needs to run in the current shell to activate venv, 
# but running the app blocks the shell anyway.
alias im="(source MLService/venv/bin/activate && cd MLService && python app.py)"

# Workflow Aliases
alias citg="./convert_issue.sh"
