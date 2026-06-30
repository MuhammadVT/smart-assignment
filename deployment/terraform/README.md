# Terraform (optional)

Empty by default. Add Terraform configs here only if your deployment
needs to provision GCP resources beyond what `adk deploy` manages
automatically (e.g. a dedicated service account, VPC, Cloud SQL for a
custom session/memory service, etc).

For most agents, `deployment/deploy.py` (which wraps `adk deploy
agent_engine`) is sufficient and this directory can stay empty or be
removed.
